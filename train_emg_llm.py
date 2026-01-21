#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_emg_llm.py — Train EMG→Text with frozen LLM backbone and NeuroSymbolic regularizer.

Trains:
  - EMG adapter (maps (B,T,112)->(B,T',D))
  - Character embedding (teacher forcing)
  - LM head to character vocab
  - CTC head on adapter stream (auxiliary)
  - (Optional) learnable soft prompt
  - (Optional) Char n-gram LM regularizer (neuro-symbolic prior)

Saves best checkpoint to CONFIG["save_to"] based on lowest validation WER.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# -------- robust imports (repo-style → flat fallback) --------
try:
    from models.emg_adapter import EMGAdapterV2 as EMGAdapter
    from scripts.data_utils import TextTransform
    from scripts.dataset_emg import make_loader
except Exception:
    from emg_adapter import EMGAdapterV2 as EMGAdapter
    from data_utils import TextTransform
    from dataset_emg import make_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================ Config ============================
CONFIG: Dict = {
    "model_name_or_path": "meta-llama/Llama-3.2-3B",
    "train_dir": "data/train_emg",
    "val_dir": "data/val_emg",
    "normalizer_path": "artifacts/emg_norm.pkl",
    "strip_suffix": "_silent",
    "batch_size": 8,
    "epochs": 300,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "save_to": "artifacts/checkpoint.pt",
    "prompt": "Transcribe the silent speech from EMG: ",
    "char_dropout": 0.3,
    "ctc_weight": 0.30,      # weight for CTC auxiliary loss
    "n_soft_prompt": 8,      # number of learnable soft-prompt tokens
    # --- NeuroSymbolic regularizer (set weight=0 to disable) ---
    "lm_reg_weight": 0.05,   # 0.0 disables the NS regularizer
    "lm_reg_order": 3,       # char n-gram order (3 is safe)
}


# ================== Lightweight Char n-gram LM ==================
class CharNgramLM:
    """
    Simple add-one-smoothed character n-gram LM used as a symbolic prior.
    Provides conditional distributions P(next_char | context) for regularization.
    """
    def __init__(self, order: int = 3):
        from collections import defaultdict
        self.order = int(order)
        self.counts = defaultdict(int)   # ngram -> count
        self.context = defaultdict(int)  # (n-1)-gram -> count
        self.V = set()

    def fit(self, texts: List[str]) -> "CharNgramLM":
        for t in texts:
            t = (t or "").strip().lower()
            s = f"^{t}$"  # boundary markers
            for i in range(len(s) - self.order + 1):
                ngram = s[i:i+self.order]
                ctx = ngram[:-1]
                self.counts[ngram] += 1
                self.context[ctx] += 1
                self.V.update(ngram)
        return self

    def cond_dist(self, context: str, base_chars: str) -> torch.Tensor:
        """
        Return a probability distribution over base_chars for next char given context.
        Uses add-one smoothing. Output is a 1D float tensor length=len(base_chars).
        """
        import math
        ctx = (context or "").lower()
        if self.order > 1:
            ctx = f"^{ctx}" if len(ctx) < (self.order - 1) else ctx[-(self.order - 1):]
        # denominator: context count + |V|
        C = self.context.get(ctx, 0)
        V = max(1, len(self.V))
        probs = []
        for ch in base_chars:
            ngram = (ctx + ch) if self.order == 1 else (ctx + ch)
            c = self.counts.get(ngram, 0)
            probs.append((c + 1) / (C + V + 1))
        p = torch.tensor(probs, dtype=torch.float32, device=DEVICE)
        p = p / (p.sum() + 1e-12)
        return p


# ============================ Model ============================
class EMGTextModel(nn.Module):
    """
    Frozen LLaMA backbone; we train:
      - EMG adapter
      - char embedding (teacher forcing)
      - small LM head to vocab
      - CTC head on adapter output (aux)
      - learnable soft prompt (optional)
    """
    def __init__(self, base_llm, hidden_size: int, vocab_size: int, pad_idx: int,
                 char_dropout: float = 0.0, n_soft_prompt: int = 0):
        super().__init__()
        self.llm = base_llm
        for p in self.llm.parameters():
            p.requires_grad = False

        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        self.char_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.char_dropout = nn.Dropout(char_dropout) if char_dropout > 0 else nn.Identity()
        self.adapter = EMGAdapter(in_dim=112, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # CTC head on adapter stream (aux)
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

        # learnable soft prompt
        self.n_soft = int(n_soft_prompt) if n_soft_prompt else 0
        if self.n_soft > 0:
            self.soft_prompt = nn.Parameter(torch.randn(1, self.n_soft, hidden_size) * 0.02)
        else:
            self.register_parameter("soft_prompt", None)

    def forward(self, prompt_embeds, emg, in_ids):
        """
        prompt_embeds: (B, P, D)
        emg          : (B, T, 112)
        in_ids       : (B, L)  (BOS + prev chars)
        Returns:
          logits_ce  : (B, L, V)   CE decoding logits
          logits_ctc : (B, T', V)  CTC logits on adapter stream
          Tprime     : int         length of adapter time-axis
        """
        B = emg.size(0)

        # Adapter stream
        emg_emb = self.adapter(emg)                 # (B, T', D)
        Tprime = emg_emb.size(1)
        logits_ctc = self.ctc_head(emg_emb)         # (B, T', V)

        # Build total prompt (soft + textual)
        if self.soft_prompt is not None:
            soft = self.soft_prompt.expand(B, -1, -1)         # (B, S, D)
            prompt_total = torch.cat([soft, prompt_embeds], dim=1)
        else:
            prompt_total = prompt_embeds

        # Teacher forcing path through frozen LLM
        char_in = self.char_embed(in_ids)           # (B, L, D)
        char_in = self.char_dropout(char_in)

        inputs = torch.cat([prompt_total, emg_emb, char_in], dim=1)  # (B, P+S+T'+L, D)
        out = self.llm(inputs_embeds=inputs, output_hidden_states=False)
        h = out.last_hidden_state                   # (B, P+S+T'+L, D)

        L = in_ids.size(1)
        char_h = h[:, -L:, :]                       # (B, L, D)
        logits_ce = self.lm_head(char_h)            # (B, L, V)

        return logits_ce, logits_ctc, Tprime


# ============================ Helpers ============================
def prepare_prompt_embeds(tokenizer, base_llm, prompt: str, batch_size: int) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    embeds = base_llm.get_input_embeddings()(ids)  # (1, P, D)
    return embeds.expand(batch_size, -1, -1).contiguous()


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    from jiwer import wer
    return float(wer(refs, hyps))


def ids_to_text_batch(TT: TextTransform, ids_list: List[List[int]]) -> List[str]:
    out = []
    for ids in ids_list:
        # remove -100 pads
        ids = [i for i in ids if i != -100]
        out.append(TT.ids_to_text(ids))
    return out


# ============================= Train =============================
def main():
    cfg = CONFIG
    Path(Path(cfg["save_to"]).parent).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader = make_loader(cfg["train_dir"], cfg["normalizer_path"],
                               batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=0, strip_suffix=cfg["strip_suffix"])
    val_loader = make_loader(cfg["val_dir"], cfg["normalizer_path"],
                             batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=0, strip_suffix=cfg["strip_suffix"])

    # Base LLaMA (no LM head)
    base_llm = AutoModel.from_pretrained(cfg["model_name_or_path"]).to(DEVICE)
    hidden = base_llm.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Modules
    TT = TextTransform()
    model = EMGTextModel(
        base_llm, hidden_size=hidden, vocab_size=TT.VOCAB_SIZE,
        pad_idx=TT.PAD_IDX, char_dropout=cfg["char_dropout"],
        n_soft_prompt=cfg["n_soft_prompt"]
    ).to(DEVICE)

    # Optim & sched
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=4)

    # CTC criterion
    criterion_ctc = nn.CTCLoss(blank=TT.PAD_IDX, zero_infinity=True)

    # ----- Build NS prior (char n-gram LM) from training refs -----
    lm_prior = None
    if cfg["lm_reg_weight"] > 0:
        texts_for_lm: List[str] = []
        # One pass over training set (batch_size=1 to avoid mixing)
        lm_loader = make_loader(cfg["train_dir"], cfg["normalizer_path"],
                                batch_size=1, shuffle=False, num_workers=0,
                                strip_suffix=cfg["strip_suffix"])
        for batch in lm_loader:
            texts_for_lm.append(batch["txts"][0])
        lm_prior = CharNgramLM(order=cfg["lm_reg_order"]).fit(texts_for_lm)

    # Helper: NS regularizer per batch
    def ns_lm_regularizer(logits_ce: torch.Tensor, in_ids: torch.Tensor, out_ids: torch.Tensor) -> torch.Tensor:
        """
        logits_ce: (B, L, V)
        in_ids   : (B, L) BOS + previous chars
        out_ids  : (B, L) target chars (with -100 at pads)
        Returns a scalar tensor (mean over valid positions) penalizing divergence from LM prior.
        """
        if lm_prior is None or cfg["lm_reg_weight"] <= 0:
            return torch.zeros((), device=DEVICE)

        B, L, V = logits_ce.shape
        # Precompute mapping: model char-space indices in [0..len(BASE_CHARS)-1]
        # model vocab indices j -> char index k (or -1 for specials)
        char_index = torch.full((V,), -1, device=DEVICE, dtype=torch.long)
        for k, ch in enumerate(TT.BASE_CHARS):
            j = TT.VOCAB_OFFSET + k
            if j < V:
                char_index[j] = k

        total = torch.zeros((), device=DEVICE)
        count = 0

        log_probs = logits_ce.log_softmax(dim=-1)  # (B,L,V)

        for i in range(B):
            # Build running context string from in_ids[i]
            # Convert ids to chars (skip PAD/BOS/EOS)
            prefix_chars = []
            for t in range(L):
                # Skip if this position is padding on output side
                if int(out_ids[i, t].item()) == -100:
                    continue

                # Build context from current prefix_chars (last order-1 chars)
                ctx = "".join(prefix_chars)

                # LM prior distribution over next char (tensor len=|BASE_CHARS|)
                p_char = lm_prior.cond_dist(ctx, TT.BASE_CHARS)  # (K,)
                # Expand to model vocab size
                p_vocab = torch.zeros((V,), dtype=torch.float32, device=DEVICE)
                # map char probs into vocab positions
                for k in range(len(TT.BASE_CHARS)):
                    j = TT.VOCAB_OFFSET + k
                    if j < V:
                        p_vocab[j] = p_char[k]
                # small epsilon to avoid log(0) issues downstream, then renormalize
                p_vocab = (p_vocab + 1e-8) / (p_vocab.sum() + 1e-8 * V)

                # CE(p_lm || p_model) = - sum p_lm * log p_model
                total = total + (- (p_vocab * log_probs[i, t]).sum())
                count += 1

                # Update context with ground-truth next char at this step
                # (teacher-forcing prefix): append target char to context
                tgt_id = int(out_ids[i, t].item())
                if tgt_id >= TT.VOCAB_OFFSET:
                    k = tgt_id - TT.VOCAB_OFFSET
                    if 0 <= k < len(TT.BASE_CHARS):
                        prefix_chars.append(TT.BASE_CHARS[k])

        if count == 0:
            return torch.zeros((), device=DEVICE)
        return total / count

    best = {"wer": 1e9, "epoch": -1}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        total_loss = 0.0

        for batch in pbar:
            emg     = batch["emg"].to(DEVICE)        # (B, T, C)
            in_ids  = batch["in_ids"].to(DEVICE)     # (B, L)
            out_ids = batch["out_ids"].to(DEVICE)    # (B, L) with -100 at pads

            prompt_embeds = prepare_prompt_embeds(tokenizer, base_llm, cfg["prompt"], emg.size(0))
            logits_ce, logits_ctc, Tprime = model(prompt_embeds, emg, in_ids)

            # ---- CE loss ----
            B, L, V = logits_ce.shape
            loss_ce = F.cross_entropy(
                logits_ce.view(B * L, V),
                out_ids.view(B * L),
                ignore_index=-100,
                reduction="mean"
            )

            # ---- CTC loss on adapter stream ----
            logp_ctc = logits_ctc.log_softmax(-1).transpose(0, 1)  # (T', B, V)
            input_lengths = torch.full((B,), Tprime, dtype=torch.long, device=DEVICE)

            target_seqs = []
            target_lengths = []
            for i in range(B):
                tgt = out_ids[i][out_ids[i] != -100]
                target_seqs.append(tgt)
                target_lengths.append(int(tgt.numel()))
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=DEVICE)
            target = torch.cat(target_seqs, dim=0) if target_seqs else torch.tensor([], dtype=torch.long, device=DEVICE)

            if target.numel() == 0:
                loss_ctc = torch.zeros((), device=DEVICE)
            else:
                loss_ctc = criterion_ctc(logp_ctc, target, input_lengths, target_lengths)

            # ---- NeuroSymbolic LM regularizer (optional) ----
            loss_ns = ns_lm_regularizer(logits_ce, in_ids, out_ids)

            loss = loss_ce + cfg["ctc_weight"] * loss_ctc + cfg["lm_reg_weight"] * loss_ns

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             ce=f"{loss_ce.item():.3f}",
                             ctc=f"{loss_ctc.item():.3f}",
                             ns=f"{loss_ns.item():.3f}" if cfg["lm_reg_weight"] > 0 else "0.000")

        # ---------- Eval ----------
        model.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                emg     = batch["emg"].to(DEVICE)
                in_ids  = batch["in_ids"].to(DEVICE)
                out_ids = batch["out_ids"].to(DEVICE)

                prompt_embeds = prepare_prompt_embeds(tokenizer, base_llm, cfg["prompt"], emg.size(0))
                logits_ce, _, _ = model(prompt_embeds, emg, in_ids)
                pred_ids = logits_ce.argmax(dim=-1).tolist()

                # Build refs/hyps with mask-aligned lengths
                for i in range(len(pred_ids)):
                    ref_ids = out_ids[i][out_ids[i] != -100].tolist()
                    hyp_ids = pred_ids[i][: len(ref_ids)]
                    refs.append(TT.ids_to_text(ref_ids))
                    hyps.append(TT.ids_to_text(hyp_ids))

        val_wer = compute_wer(refs, hyps)
        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch}: val WER = {val_wer:.4f} (avg train loss {avg_train_loss:.4f})")
        sched.step(val_wer)

        # Save best
        if val_wer <= best["wer"] or epoch == 1:
            best = {"wer": val_wer, "epoch": epoch}
            ckpt = {
                "config": cfg,
                "state_dict": {
                    "char_embed": model.char_embed.state_dict(),
                    "adapter": model.adapter.state_dict(),
                    "lm_head": model.lm_head.state_dict(),
                    "soft_prompt": (model.soft_prompt.data if model.soft_prompt is not None else None),
                    "ctc_head": model.ctc_head.state_dict(),
                },
                "llm_name": cfg["model_name_or_path"],
                "text_meta": {
                    "BASE_CHARS": TT.BASE_CHARS,
                    "BOS_IDX": TT.BOS_IDX,
                    "EOS_IDX": TT.EOS_IDX,
                    "PAD_IDX": TT.PAD_IDX,
                    "VOCAB_SIZE": TT.VOCAB_SIZE,
                },
            }
            out = Path(cfg["save_to"])
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out)
            print(f"  ✓ saved best checkpoint → {out} (WER={val_wer:.4f})")

    print(f"\nBest WER {best['wer']:.4f} @ epoch {best['epoch']}")

if __name__ == "__main__":
    main()
