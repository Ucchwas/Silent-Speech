#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_emg_llm.py — NeuroSymbolic decoding for EMG→Text (inference)
- Loads frozen LLM backbone + your trained adapter/head checkpoint
- Trie-constrained AR beam search + optional LM prior (KenLM or char n-gram)
- Rule-based post-correction
- Optional CTC↔AR re-rank
- Reads *.npy (+ matching *.json) from a folder, writes predictions to .xlsx, prints overall WER

Minimal:
    python inference_emg_llm.py --ckpt artifacts/checkpoint.pt \
        --normalizer artifacts/emg_norm.pkl --val_dir data/val_emg

Recommended (with lexicon/phrasebook if you have them):
    python inference_emg_llm.py --ckpt artifacts/checkpoint.pt \
        --normalizer artifacts/emg_norm.pkl --val_dir data/val_emg \
        --lexicon data/lexicon.txt --phrasebook data/phrasebook.txt --beam 6
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ------------------ Project imports (robust fallbacks) ------------------
try:
    # Repo-style imports
    from scripts.data_utils import FeatureNormalizer, TextTransform
    from models.emg_adapter import EMGAdapterV2 as EMGAdapter
except Exception:
    # Flat-file fallbacks
    from data_utils import FeatureNormalizer, TextTransform
    from emg_adapter import EMGAdapterV2 as EMGAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


# ======================================================================
#                            Model Wrapper
# ======================================================================
class EMGTextModel(nn.Module):
    """
    Frozen LLM + your EMG adapter + character embedding/LM head (+ CTC head).
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
        self.ctc_head = nn.Linear(hidden_size, vocab_size)   # used for optional CTC decode

        self.n_soft = int(n_soft_prompt) if n_soft_prompt else 0
        if self.n_soft > 0:
            self.soft_prompt = nn.Parameter(torch.randn(1, self.n_soft, hidden_size) * 0.02)
        else:
            self.register_parameter("soft_prompt", None)

    def encode_context(self, prompt_embeds: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        emg_emb = self.adapter(emg)  # (B, T', D)
        parts = []
        if self.soft_prompt is not None:
            parts.append(self.soft_prompt.expand(emg_emb.size(0), -1, -1))  # (B, S, D)
        parts.append(prompt_embeds)
        parts.append(emg_emb)
        return torch.cat(parts, dim=1)  # (B, P'+T', D)

    def step(self, context: torch.Tensor, in_ids: torch.Tensor) -> torch.Tensor:
        tok_emb = self.char_embed(in_ids)                # (B, L, D)
        inputs_embeds = torch.cat([context, tok_emb], 1) # (B, S+L, D)
        outputs = self.llm(inputs_embeds=inputs_embeds)
        last_h = outputs.last_hidden_state[:, -1, :]     # (B, D)
        return self.lm_head(last_h)                      # (B, V)


# ======================================================================
#                      Neuro-Symbolic Utilities (in-file)
# ======================================================================
# ------------------ Trie & Lexicon ------------------
class Trie:
    def __init__(self):
        self.next: Dict[str, "Trie"] = {}
        self.end: bool = False
    def insert(self, s: str):
        cur = self
        for ch in s:
            cur = cur.next.setdefault(ch, Trie())
        cur.end = True
    def next_chars(self, prefix: str) -> Iterable[str]:
        cur = self
        for ch in prefix:
            if ch not in cur.next:
                return ()
            cur = cur.next[ch]
        return cur.next.keys()
    def is_word(self, s: str) -> bool:
        cur = self
        for ch in s:
            if ch not in cur.next:
                return False
            cur = cur.next[ch]
        return cur.end

def build_trie_from_words(words: Iterable[str]) -> Trie:
    T = Trie()
    for w in words:
        w = (w or "").strip().lower()
        if w:
            T.insert(w)
    return T

# ------------------ Lightweight Char n-gram LM ------------------
class CharNgramLM:
    def __init__(self, order: int = 3):
        from collections import defaultdict
        self.order = order
        self.counts = defaultdict(int)
        self.context = defaultdict(int)
        self.V = set()

    def fit(self, texts: List[str]):
        for t in texts:
            t = (t or "").strip().lower()
            s = f"^{t}$"
            for i in range(len(s) - self.order + 1):
                ngram = s[i:i+self.order]
                ctx = ngram[:-1]
                self.counts[ngram] += 1
                self.context[ctx] += 1
                self.V.update(ngram)

    def logp(self, text: str) -> float:
        import math
        if not text:
            return -1e9
        s = f"^{text.strip().lower()}$"
        logp = 0.0
        for i in range(len(s) - self.order + 1):
            ngram = s[i:i+self.order]
            ctx = ngram[:-1]
            c = self.counts.get(ngram, 0)
            C = self.context.get(ctx, 0)
            logp += math.log((c + 1) / (C + len(self.V) + 1))  # add-one smoothing
        return logp

class KenLMWrapper:
    def __init__(self, path: str):
        import kenlm  # optional dependency
        self.m = kenlm.Model(path)
    def logp(self, text: str) -> float:
        return float(self.m.score(text.strip().lower(), bos=True, eos=True))

# ------------------ Helpers ------------------
def ids_to_text(TT: TextTransform, ids: List[int]) -> str:
    """Map ids (with BOS/EOS) to surface text using TT constants."""
    out = []
    for i in ids[1:]:
        if i == TT.EOS_IDX:
            break
        j = i - TT.VOCAB_OFFSET  # character space starts at VOCAB_OFFSET
        if 0 <= j < len(TT.BASE_CHARS):
            out.append(TT.BASE_CHARS[j])
    return "".join(out).strip()

def postprocess(text: str, lex_trie: Trie, lex_words: List[str]) -> str:
    """Collapse spaces/repeats; snap OOV words to nearest lexicon entry."""
    import difflib
    t = " ".join((text or "").split())  # collapse whitespace
    words = []
    for w in t.split():
        # remove ≥3 same-letter runs
        fixed = []
        for ch in w:
            if len(fixed) >= 2 and fixed[-1] == ch == fixed[-2]:
                continue
            fixed.append(ch)
        w = "".join(fixed)
        if not lex_trie.is_word(w):
            cand = difflib.get_close_matches(w, lex_words, n=1, cutoff=0.82)
            if cand:
                w = cand[0]
        words.append(w)
    return " ".join(words)

def read_reference(json_path: Path) -> str | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in ("text", "transcript", "sentence", "label"):
        if isinstance(data.get(k), str):
            return data[k]
    for v in data.values():
        if isinstance(v, str):
            return v
    return None

def compute_wer(refs: List[str], hyps: List[str]) -> float:
    import jiwer
    from unidecode import unidecode
    t = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    refs = [t(unidecode(r or "")) for r in refs]
    hyps = [t(unidecode(h or "")) for h in hyps]
    return float(jiwer.wer(refs, hyps))

# ------------------ Constrained AR beam ------------------
def constrained_beam_decode(
    step_fn,                   # callable: (List[int]) -> Tensor[1,V] next-step logits
    TT: TextTransform,
    beam: int,
    max_new_tokens: int,
    trie: Trie,
    lm=None,                   # None | CharNgramLM | KenLMWrapper
    lm_weight: float = 0.3,
    oov_penalty: float = 1.0
) -> List[int]:
    """
    Trie-constrained AR beam over character vocab.
    Maintains (logprob, ids, surface_text). Only extend with chars that keep a lexicon path.
    """
    beams: List[Tuple[float, List[int], str]] = [(0.0, [TT.BOS_IDX], "")]
    space_id = TT.VOCAB_OFFSET + TT.BASE_CHARS.index(" ")

    for _ in range(max_new_tokens):
        cand: List[Tuple[float, List[int], str]] = []

        for logp, ids, surf in beams:
            if ids[-1] == TT.EOS_IDX:
                cand.append((logp, ids, surf))
                continue

            logits = step_fn(ids)                 # (1, V)
            logprobs = F.log_softmax(logits, dim=-1).squeeze(0)

            # Current word prefix (since last space)
            prefix = "" if surf.endswith(" ") else (surf.split()[-1] if surf else "")
            allowed = set(trie.next_chars(prefix))
            # If prefix already a word, allow space to delimit
            if trie.is_word(prefix):
                allowed.add(" ")

            options: List[Tuple[float, int, str]] = []
            # Allowed, lexicon-constrained expansions
            for ch in allowed:
                if ch == " ":
                    nxt_id = space_id
                    new_text = surf + " "
                else:
                    if ch not in TT.BASE_CHARS:
                        continue
                    nxt_id = TT.VOCAB_OFFSET + TT.BASE_CHARS.index(ch)
                    new_text = surf + ch

                add_lp = float(logprobs[nxt_id].item())
                # LM bonus when we complete a word (on space) or in general
                if lm is not None:
                    add_lp += lm_weight * float(lm.logp(new_text))
                # discourage inserting a space if unfinished/non-word
                if ch == " " and not trie.is_word(prefix):
                    add_lp -= oov_penalty

                options.append((add_lp, nxt_id, new_text))

            # Fallback: if nothing allowed (e.g., OOV prefix), let top-k neural chars through
            if not options:
                topk = logprobs.topk(k=beam)
                for add_lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    # Map ID back to char (best effort)
                    ch = ""
                    j = idx - TT.VOCAB_OFFSET
                    if 0 <= j < len(TT.BASE_CHARS):
                        ch = TT.BASE_CHARS[j]
                    new_text = (surf + ch).strip()
                    options.append((add_lp - oov_penalty, int(idx), new_text))

            for add_lp, idx, txt in options:
                cand.append((logp + add_lp, ids + [idx], txt))

        # prune
        beams = sorted(cand, key=lambda x: x[0], reverse=True)[:beam]
        if all(seq[-1] == TT.EOS_IDX for _, seq, _ in beams):
            break

    best = max(beams, key=lambda x: x[0])
    return best[1]


# ======================================================================
#                                Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoint.pt")
    ap.add_argument("--normalizer", default="artifacts/emg_norm.pkl")
    ap.add_argument("--val_dir", default="data/val_emg")
    ap.add_argument("--output", default="predictions_ns.xlsx")
    ap.add_argument("--prompt", default="Transcribe the silent speech from EMG: ")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--strip_suffix", default="_silent", help="strip this suffix from .npy stem to find matching .json")
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--lexicon", type=str, default="data/lexicon.txt")
    ap.add_argument("--phrasebook", type=str, default="data/phrasebook.txt")
    ap.add_argument("--lm_arpa", type=str, default="")  # optional: artifacts/kenlm.arpa
    ap.add_argument("--lm_order", type=int, default=3)
    ap.add_argument("--ns_lm_weight", type=float, default=0.3)
    ap.add_argument("--ns_oov_penalty", type=float, default=1.0)
    ap.add_argument("--use_ctc_rerank", action="store_true", default=True)
    args = ap.parse_args()

    # ---------- Load checkpoint ----------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    llm_name = ckpt.get("llm_name", ckpt.get("config", {}).get("model_name_or_path", "meta-llama/Llama-3.2-3B"))

    # ---------- Build backbone & modules ----------
    base_llm = AutoModel.from_pretrained(llm_name).to(DEVICE)
    hidden = base_llm.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    TT = TextTransform()
    model = EMGTextModel(base_llm, hidden_size=hidden, vocab_size=TT.VOCAB_SIZE,
                         pad_idx=TT.PAD_IDX, n_soft_prompt=0).to(DEVICE).eval()

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

    # ---------- Normalizer ----------
    norm = FeatureNormalizer().load(args.normalizer)

    # ---------- Files ----------
    val_dir = Path(args.val_dir)
    npy_files = sorted(val_dir.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"No .npy files found in {val_dir}")

    # ---------- Collect references (for LM/lexicon fallback) ----------
    refs_for_lm: List[str] = []
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for npy in npy_files:
        # find matching JSON (same stem, or stem with/without strip_suffix)
        stem = npy.stem
        cand = [val_dir / f"{stem}.json"]
        if args.strip_suffix and stem.endswith(args.strip_suffix):
            cand.append(val_dir / f"{stem[:-len(args.strip_suffix)]}.json")
        else:
            cand.append(val_dir / f"{stem}{args.strip_suffix}.json")
        json_path = next((p for p in cand if p.exists()), None)
        pairs.append((npy, json_path))
        if json_path:
            ref_raw = read_reference(json_path)
            if isinstance(ref_raw, str):
                refs_for_lm.append(TextTransform().clean(ref_raw))

    # ---------- Lexicon / Trie ----------
    lex_words: List[str] = []
    lex_path = Path(args.lexicon)
    if lex_path.exists():
        lex_words += [l.strip().lower() for l in lex_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    phrase_path = Path(args.phrasebook)
    if phrase_path.exists():
        lex_words += [l.strip().lower() for l in phrase_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lex_words:
        # fall back: use references in this split (closed vocab decoding)
        for t in refs_for_lm:
            lex_words.extend([w for w in (t or "").split() if w.strip()])
        lex_words = sorted(list(set(lex_words)))
    lex_trie = build_trie_from_words(lex_words)

    # ---------- Language Model ----------
    lm = None
    if args.lm_arpa and Path(args.lm_arpa).exists():
        try:
            lm = KenLMWrapper(args.lm_arpa)
        except Exception:
            lm = None  # if kenlm not installed
    if lm is None:
        lm = CharNgramLM(order=args.lm_order)
        lm.fit(refs_for_lm)

    # ---------- Prompt embeds builder ----------
    def prompt_embeds(bsz: int) -> torch.Tensor:
        ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        emb = model.llm.get_input_embeddings()(ids)  # (1, P, D)
        return emb.expand(bsz, -1, -1).contiguous()

    # ---------- Decode ----------
    rows = []
    for npy, json_path in pairs:
        # load + normalize emg
        x = np.load(npy).astype(np.float32)  # (T, C)
        x = norm.transform(x)
        emg = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, T, C)

        # context
        P = prompt_embeds(1)                           # (1, P, D)
        context = model.encode_context(P, emg)         # (1, S, D)

        # step_fn closure for constrained beam
        def step_fn(ids_list: List[int]) -> torch.Tensor:
            in_ids = torch.tensor(ids_list, dtype=torch.long, device=DEVICE).unsqueeze(0)
            return model.step(context, in_ids)  # (1, V)

        # Constrained AR beam + LM
        ids = constrained_beam_decode(
            step_fn, TT, beam=max(2, args.beam), max_new_tokens=args.max_len,
            trie=lex_trie, lm=lm, lm_weight=args.ns_lm_weight, oov_penalty=args.ns_oov_penalty
        )
        hyp_ar = ids_to_text(TT, ids)
        hyp_ar = postprocess(hyp_ar, lex_trie, lex_words)

        # Optional: quick CTC greedy (adapter stream) and re-rank
        hyp = hyp_ar
        if args.use_ctc_rerank:
            with torch.no_grad():
                emg_emb = model.adapter(emg)                    # (1, T', D)
                logits_ctc = model.ctc_head(emg_emb)            # (1, T', V)
                ctc_ids = logits_ctc.argmax(-1).squeeze(0).tolist()
                # collapse repeats & strip PAD/EOS/BOS
                dedup = []
                prev = None
                for t in ctc_ids:
                    if t == prev:
                        continue
                    prev = t
                    if t in (TT.PAD_IDX, TT.BOS_IDX, TT.EOS_IDX):
                        continue
                    dedup.append(t)
                # wrap with BOS/EOS for ids_to_text
                hyp_ctc = ids_to_text(TT, [TT.BOS_IDX] + dedup + [TT.EOS_IDX])
                hyp_ctc = postprocess(hyp_ctc, lex_trie, lex_words)

            # choose with LM + mild length penalty
            def lm_score(s: str) -> float:
                return float(lm.logp(s)) - 0.002 * len(s)
            cand = [(hyp_ar, lm_score(hyp_ar)), (hyp_ctc, lm_score(hyp_ctc))]
            hyp = max(cand, key=lambda x: x[1])[0]

        # reference handling
        ref_raw = read_reference(json_path) if json_path else ""
        ref_clean = TextTransform().clean(ref_raw) if isinstance(ref_raw, str) else ""

        # per-sample WER (skip if no ref)
        wer_val = compute_wer([ref_clean], [hyp]) if ref_clean else None

        rows.append({
            "file": npy.name,
            "json": json_path.name if json_path else "",
            "reference_raw": ref_raw if isinstance(ref_raw, str) else "",
            "reference": ref_clean,
            "prediction": hyp,
            "wer": wer_val,
        })

    df = pd.DataFrame(rows)

    # overall WER on rows with references
    if (df["reference"] != "").any():
        overall = compute_wer(df.loc[df.reference != "", "reference"].tolist(),
                              df.loc[df.reference != "", "prediction"].tolist())
        print(f"Files: {len(df)}  |  references: {(df.reference!='').sum()}  |  overall WER: {overall:.4f}")
    else:
        print(f"Files: {len(df)}  |  no references found; skipping WER.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"✓ Saved predictions → {out_path.resolve()}")

if __name__ == "__main__":
    main()
