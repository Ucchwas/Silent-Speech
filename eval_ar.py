#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style AR evaluation for EMG->Text with a frozen LLM.
- Loads your training checkpoint (adapter/char_embed/LM head/soft prompt).
- Builds context = [soft_prompt?, textual_prompt, EMG adapter embeddings].
- Runs autoregressive decoding (greedy or beam) on the character vocab.
- Writes predictions to an .xlsx and prints corpus WER.

Usage:
    python eval_ar.py \        --ckpt artifacts/checkpoint.pt \        --data_dir data/val_emg \        --normalizer_path artifacts/emg_norm.pkl \        --beam 4 \        --strip_suffix _silent \        --out_xlsx predictions_ar.xlsx
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Repo-style imports
    from scripts.data_utils import TextTransform
    from scripts.dataset_emg import make_loader
    from models.emg_adapter import EMGAdapterV2 as EMGAdapter
except Exception:
    # Flat-file fallback
    from data_utils import TextTransform
    from dataset_emg import make_loader
    from emg_adapter import EMGAdapterV2 as EMGAdapter

from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


# ------------------ Model wrapper ------------------
class EMGTextModel(nn.Module):
    """
    Frozen LLM + your EMG adapter + character embedding/LM head.
    Exposes:
      - encode_context(prompt_embeds, emg) -> (B, S, D)
      - step(context, in_ids) -> logits over vocab for next char
    """
    def __init__(self, base_llm, hidden_size: int, vocab_size: int, pad_idx: int, n_soft_prompt: int = 0):
        super().__init__()
        self.llm = base_llm
        for p in self.llm.parameters():
            p.requires_grad = False

        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        self.char_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.adapter = EMGAdapter(in_dim=112, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Saved in ckpt (optional for AR math but kept for parity)
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

        self.n_soft = int(n_soft_prompt) if n_soft_prompt else 0
        if self.n_soft > 0:
            self.soft_prompt = nn.Parameter(torch.randn(1, self.n_soft, hidden_size) * 0.02)
        else:
            self.register_parameter("soft_prompt", None)

    def encode_context(self, prompt_embeds: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        """
        Return fixed context embedding sequence: [soft_prompt?, prompt_embeds, emg_emb].
            prompt_embeds: (B, P, D)
            emg:           (B, T, 112)
        """
        emg_emb = self.adapter(emg)  # (B, T', D)
        parts = []
        if self.soft_prompt is not None:
            parts.append(self.soft_prompt.expand(emg_emb.size(0), -1, -1))  # (B, S, D)
        parts.append(prompt_embeds)
        parts.append(emg_emb)
        return torch.cat(parts, dim=1)  # (B, P'+T', D)

    def step(self, context: torch.Tensor, in_ids: torch.Tensor) -> torch.Tensor:
        """
        One AR step: returns logits over vocab for the next char given (context + tokens-so-far).
            context: (B, S, D) fixed per utterance
            in_ids : (B, L) BOS + generated so far
        """
        tok_emb = self.char_embed(in_ids)                # (B, L, D)
        inputs_embeds = torch.cat([context, tok_emb], 1) # (B, S+L, D)
        outputs = self.llm(inputs_embeds=inputs_embeds)
        last_h = outputs.last_hidden_state[:, -1, :]     # (B, D)
        return self.lm_head(last_h)                      # (B, V)


# ------------------ Decoding ------------------
def greedy_decode(model: EMGTextModel, context: torch.Tensor, max_new_tokens: int,
                  bos_idx: int, eos_idx: int) -> List[int]:
    ids = [bos_idx]
    for _ in range(max_new_tokens):
        in_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model.step(context, in_ids)  # (1, V)
        next_id = int(logits.argmax(-1).item())
        ids.append(next_id)
        if next_id == eos_idx:
            break
    return ids


def beam_search_decode(model: EMGTextModel, context: torch.Tensor, max_new_tokens: int,
                       bos_idx: int, eos_idx: int, beam: int = 4, lp_alpha: float = 0.7) -> List[int]:
    """
    Length-penalized beam search on character vocab.
    Returns the best token id sequence (including BOS and maybe EOS).
    """
    BeamItem = Tuple[float, List[int]]  # (logprob, ids)
    beams: List[BeamItem] = [(0.0, [bos_idx])]

    for _ in range(max_new_tokens):
        cand: List[BeamItem] = []
        for logp, ids in beams:
            if ids[-1] == eos_idx:
                cand.append((logp, ids))  # keep finished beams
                continue
            in_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            logits = model.step(context, in_ids)  # (1, V)
            logprobs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
            topk_logp, topk_idx = torch.topk(logprobs, k=beam)
            for add_logp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                cand.append((logp + add_logp, ids + [int(idx)]))

        def score(item):
            lprob, ids = item
            L = max(1, len(ids) - 1)  # exclude BOS
            # GNMT-style length penalty
            return lprob / (((5 + L) ** lp_alpha) / ((5 + 1) ** lp_alpha))

        beams = sorted(cand, key=score, reverse=True)[:beam]
        if all(seq[-1] == eos_idx for _, seq in beams):
            break

    best = max(beams, key=lambda x: x[0])
    return best[1]


def ids_to_text(tt: TextTransform, ids: List[int]) -> str:
    # Strip BOS, stop at EOS, map via BASE_CHARS
    out = []
    for i in ids[1:]:
        if i == tt.EOS_IDX:
            break
        if 0 <= i < len(tt.BASE_CHARS):
            out.append(tt.BASE_CHARS[i])
    return "".join(out).strip()


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    import jiwer
    from unidecode import unidecode
    # Align with TextTransform.clean: unidecode + lower + remove punctuation
    t = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    refs = [t(unidecode(r)) for r in refs]
    hyps = [t(unidecode(h)) for h in hyps]
    return jiwer.wer(refs, hyps)


# ------------------ Main ------------------
def main():
    import argparse

    # Defaults so `python eval_ar.py` works with no args
    DEFAULTS = dict(
        ckpt="artifacts/checkpoint.pt",
        data_dir="data/val_emg",
        normalizer_path="artifacts/emg_norm.pkl",
        prompt="Transcribe the silent speech from EMG: ",
        batch_size=1,            # keep 1 for AR decoding
        beam=4,                  # paper-style decoding
        max_new_tokens=64,
        strip_suffix="_silent",  # set "" if you don't need it
        out_xlsx="predictions_ar.xlsx",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULTS["ckpt"], help="Path to training checkpoint (.pt)")
    ap.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"], help="Dir with *.npy + *.json pairs")
    ap.add_argument("--normalizer_path", type=str, default=DEFAULTS["normalizer_path"], help="Path to saved normalizer.pkl")
    ap.add_argument("--prompt", type=str, default=DEFAULTS["prompt"])
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"], help="Keep 1 for AR decoding")
    ap.add_argument("--beam", type=int, default=DEFAULTS["beam"], help="Beam width; set 1 to use greedy")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULTS["max_new_tokens"])
    ap.add_argument("--strip_suffix", type=str, default=DEFAULTS["strip_suffix"], help="If .npy filenames have extra suffix (e.g., _silent)")
    ap.add_argument("--out_xlsx", type=str, default=DEFAULTS["out_xlsx"])
    args = ap.parse_args()


    # ----- Load checkpoint & backbone LLM -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    llm_name = ckpt.get("llm_name", ckpt.get("config", {}).get("model_name_or_path", "meta-llama/Llama-3.2-3B"))
    base_llm = AutoModel.from_pretrained(llm_name).to(DEVICE)
    hidden = base_llm.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Build EMGTextModel and load heads/adapter -----
    TT = TextTransform()
    model = EMGTextModel(base_llm, hidden_size=hidden, vocab_size=TT.VOCAB_SIZE,
                         pad_idx=TT.PAD_IDX, n_soft_prompt=0).to(DEVICE)

    sd = ckpt["state_dict"]
    model.char_embed.load_state_dict(sd["char_embed"])
    model.adapter.load_state_dict(sd["adapter"])
    model.lm_head.load_state_dict(sd["lm_head"])
    if "ctc_head" in sd:
        try:
            model.ctc_head.load_state_dict(sd["ctc_head"])
        except Exception:
            pass
    if sd.get("soft_prompt", None) is not None:
        with torch.no_grad():
            sp = sd["soft_prompt"].to(DEVICE)
            model.soft_prompt = nn.Parameter(sp)
            model.n_soft = sp.shape[1]

    # ----- Data loader -----
    loader = make_loader(args.data_dir, args.normalizer_path,
                         batch_size=args.batch_size, shuffle=False, num_workers=0,
                         strip_suffix=args.strip_suffix)

    # ----- Prompt embeddings (once per batch) -----
    def make_prompt_embeds(bsz: int) -> torch.Tensor:
        ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        emb = model.llm.get_input_embeddings()(ids)  # (1, P, D)
        return emb.expand(bsz, -1, -1)               # (B, P, D)

    # ----- Decode & collect -----
    names, refs, hyps = [], [], []
    for batch in loader:
        emg = batch["emg"].to(DEVICE)  # (1, T, 112)
        ref = batch["txts"][0]
        names.append(batch["names"][0])
        refs.append(ref)

        prompt_embeds = make_prompt_embeds(emg.size(0))
        context = model.encode_context(prompt_embeds, emg)  # (1, S, D)

        if args.beam and args.beam > 1:
            ids = beam_search_decode(model, context, args.max_new_tokens,
                                     TT.BOS_IDX, TT.EOS_IDX, beam=args.beam)
        else:
            ids = greedy_decode(model, context, args.max_new_tokens,
                                TT.BOS_IDX, TT.EOS_IDX)
        hyp = ids_to_text(TT, ids)
        hyps.append(hyp)

    # ----- Save predictions and print WER -----
    import pandas as pd
    df = pd.DataFrame({"file": names, "reference": refs, "prediction": hyps})
    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)

    w = compute_wer(refs, hyps)
    print(f"Corpus WER (AR decode, beam={args.beam}): {w:.4f}")
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
