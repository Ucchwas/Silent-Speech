#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ar.py — NeuroSymbolic decoding for EMG→Text
- Frozen LLM backbone, trained adapter + heads (from your checkpoint)
- Trie-constrained AR beam search + optional LM prior (KenLM or light char n-gram)
- Rule-based post-correction
- Optional CTC↔AR re-rank
- Writes predictions to .xlsx and prints corpus WER

Run (minimal):
    python eval_ar.py --ckpt artifacts/checkpoint.pt --data_dir data/val_emg \
        --normalizer_path artifacts/emg_norm.pkl

Recommended (with lexicon & phrasebook if you have them):
    python eval_ar.py --ckpt artifacts/checkpoint.pt --data_dir data/val_emg \
        --normalizer_path artifacts/emg_norm.pkl \
        --lexicon data/lexicon.txt --phrasebook data/phrasebook.txt --beam 6
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Project imports (robust fallbacks) ------------------
try:
    # Repo-style imports
    from scripts.data_utils import TextTransform
    from scripts.dataset_emg import make_loader
    from models.emg_adapter import EMGAdapterV2 as EMGAdapter
except Exception:
    # Flat-file fallbacks
    from data_utils import TextTransform
    from dataset_emg import make_loader
    from emg_adapter import EMGAdapterV2 as EMGAdapter

from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


# ======================================================================
#                            Model Wrapper
# ======================================================================
class EMGTextModel(nn.Module):
    """
    Frozen LLM + your EMG adapter + character embedding/LM head (+ CTC head).
    Exposes:
      - encode_context(prompt_embeds, emg) -> (B, S, D) fixed per utterance
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
        # character space starts at VOCAB_OFFSET
        j = i - TT.VOCAB_OFFSET
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

def compute_wer(refs: List[str], hyps: List[str]) -> float:
    import jiwer
    from unidecode import unidecode
    t = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    refs = [t(unidecode(r or "")) for r in refs]
    hyps = [t(unidecode(h or "")) for h in hyps]
    return float(jiwer.wer(refs, hyps))

# ------------------ Constrained AR beam ------------------
def constrained_beam_decode(
    step_fn,
    TT: TextTransform,
    beam: int,
    max_new_tokens: int,
    trie: Trie,
    lm=None,
    lm_weight: float = 0.3,
    oov_penalty: float = 1.0,
    *,
    word_bonus: float = 0.0,
    length_penalty_alpha: float = 0.6,
    min_len_for_eos: int = 8,
) -> List[int]:
    """
    Trie-constrained AR beam over character vocab with:
      - explicit EOS option at word boundaries,
      - GNMT-style length normalization for pruning & final pick,
      - 'word_bonus' when inserting a space after a valid lexicon word.
    """
    beams: List[Tuple[float, List[int], str]] = [(0.0, [TT.BOS_IDX], "")]
    space_id = TT.VOCAB_OFFSET + TT.BASE_CHARS.index(" ")

    def norm_score(entry: Tuple[float, List[int], str]) -> float:
        lp, seq, _ = entry
        L = max(1, len(seq) - 1)  # generated tokens excluding BOS
        denom = ((5 + L) ** length_penalty_alpha) / ((5 + 1) ** length_penalty_alpha)
        return lp / denom

    for _ in range(max_new_tokens):
        cand: List[Tuple[float, List[int], str]] = []
        for logp, ids, surf in beams:
            # already finished
            if ids[-1] == TT.EOS_IDX:
                cand.append((logp, ids, surf))
                continue

            logits = step_fn(ids)                      # (1, V)
            logprobs = F.log_softmax(logits, dim=-1).squeeze(0)

            # prefix since last space
            prefix = "" if surf.endswith(" ") else (surf.split()[-1] if surf else "")
            allowed = set(trie.next_chars(prefix))
            if trie.is_word(prefix):
                allowed.add(" ")  # allow delimiter when current prefix already forms a word

            options: List[Tuple[float, int, str]] = []

            # 1) lexicon-constrained expansions (characters + word-delimiting space)
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

                # LM prior
                if lm is not None:
                    add_lp += lm_weight * float(lm.logp(new_text))

                # discourage inserting space if current prefix is not a full word yet
                if ch == " " and not trie.is_word(prefix):
                    add_lp -= oov_penalty

                # reward finishing a valid word (space after a lexicon word)
                if ch == " " and trie.is_word(prefix):
                    add_lp += word_bonus

                options.append((add_lp, nxt_id, new_text))

            # 2) explicit EOS at word boundary, but not too early
            if (len(surf) >= min_len_for_eos) and (surf.endswith(" ") or trie.is_word(prefix)):
                add_lp = float(logprobs[TT.EOS_IDX].item())
                # small nudge to avoid ultra-short outputs
                add_lp -= 0.002 * len(surf)
                if lm is not None and surf:
                    add_lp += lm_weight * float(lm.logp(surf))
                options.append((add_lp, TT.EOS_IDX, surf))

            # 3) soft fallback when the trie gives no next char (e.g., OOV prefix)
            if not options:
                topk = logprobs.topk(k=beam)
                for add_lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    j = idx - TT.VOCAB_OFFSET
                    ch = TT.BASE_CHARS[j] if 0 <= j < len(TT.BASE_CHARS) else ""
                    new_text = (surf + ch).strip()
                    options.append((add_lp - oov_penalty, int(idx), new_text))

            for add_lp, idx, txt in options:
                cand.append((logp + add_lp, ids + [idx], txt))

        # prune with length-normalized score
        beams = sorted(cand, key=norm_score, reverse=True)[:beam]
        if all(seq[-1] == TT.EOS_IDX for _, seq, _ in beams):
            break

    # pick best with the same normalization used during pruning
    best = max(beams, key=norm_score)
    return best[1]

def greedy_decode(step_fn, TT, max_new_tokens: int, min_len_for_eos: int = 1):
    ids = [TT.BOS_IDX]
    for _ in range(max_new_tokens):
        logits = step_fn(ids)
        nxt = int(torch.argmax(logits, dim=-1)[0].item())
        if (len(ids) - 1) >= min_len_for_eos and nxt == TT.EOS_IDX:
            ids.append(nxt); break
        ids.append(nxt)
    if ids[-1] != TT.EOS_IDX:
        ids.append(TT.EOS_IDX)
    return ids

def beam_decode(step_fn, TT, beam: int, max_new_tokens: int,
                length_penalty_alpha: float = 0.6, min_len_for_eos: int = 1):
    def norm_score(lp, L):
        denom = ((5 + L) ** length_penalty_alpha) / ((5 + 1) ** length_penalty_alpha)
        return lp / denom
    beams = [(0.0, [TT.BOS_IDX], "")]
    for _ in range(max_new_tokens):
        cand = []
        for logp, ids, surf in beams:
            if ids[-1] == TT.EOS_IDX:
                cand.append((logp, ids, surf)); continue
            logits = step_fn(ids)
            logprobs = F.log_softmax(logits, dim=-1).squeeze(0)
            # expand over all base chars
            for j, ch in enumerate(TT.BASE_CHARS):
                idx = TT.VOCAB_OFFSET + j
                cand.append((logp + float(logprobs[idx].item()), ids + [idx], surf + ch))
            # allow EOS after a minimum length
            if len(surf) >= min_len_for_eos:
                cand.append((logp + float(logprobs[TT.EOS_IDX].item()),
                             ids + [TT.EOS_IDX], surf))
        beams = sorted(cand, key=lambda x: norm_score(x[0], len(x[1]) - 1), reverse=True)[:beam]
        if all(seq[-1] == TT.EOS_IDX for _, seq, _ in beams):
            break
    best = max(beams, key=lambda x: norm_score(x[0], len(x[1]) - 1))
    return best[1]


# ======================================================================
#                                Main
# ======================================================================
def main():
    # ---------- Defaults ----------
    DEFAULTS = dict(
        ckpt="artifacts/checkpoint.pt",
        data_dir="data/val_emg",
        normalizer_path="artifacts/emg_norm.pkl",
        prompt="Transcribe the silent speech from EMG: ",
        batch_size=1,              # keep 1 for AR decoding
        beam=6,                    # wider beam helps with constraints
        max_new_tokens=64,
        strip_suffix="_silent",
        out_xlsx="predictions_ar_ns.xlsx",
        lexicon="data/lexicon.txt",
        phrasebook="data/phrasebook.txt",
        lm_arpa="",                # optional: artifacts/kenlm.arpa
        lm_order=3,
        ns_lm_weight=0.3,
        ns_oov_penalty=1.0,
        use_ctc_rerank=True,
        word_bonus=0.35,               
        length_penalty_alpha=0.6,
        min_len_for_eos=8,
    )

    # ---------- Args ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULTS["ckpt"])
    ap.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"])
    ap.add_argument("--normalizer_path", type=str, default=DEFAULTS["normalizer_path"])
    ap.add_argument("--prompt", type=str, default=DEFAULTS["prompt"])
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--beam", type=int, default=DEFAULTS["beam"])
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULTS["max_new_tokens"])
    ap.add_argument("--strip_suffix", type=str, default=DEFAULTS["strip_suffix"])
    ap.add_argument("--out_xlsx", type=str, default=DEFAULTS["out_xlsx"])
    ap.add_argument("--lexicon", type=str, default=DEFAULTS["lexicon"])
    ap.add_argument("--phrasebook", type=str, default=DEFAULTS["phrasebook"])
    ap.add_argument("--lm_arpa", type=str, default=DEFAULTS["lm_arpa"])
    ap.add_argument("--lm_order", type=int, default=DEFAULTS["lm_order"])
    ap.add_argument("--ns_lm_weight", type=float, default=DEFAULTS["ns_lm_weight"])
    ap.add_argument("--ns_oov_penalty", type=float, default=DEFAULTS["ns_oov_penalty"])
    ap.add_argument("--use_ctc_rerank", action="store_true", default=True)
    ap.add_argument("--no_ctc_rerank", action="store_false", dest="use_ctc_rerank")
    ap.add_argument("--word_bonus", type=float, default=DEFAULTS["word_bonus"])
    ap.add_argument("--length_penalty_alpha", type=float, default=DEFAULTS["length_penalty_alpha"])
    ap.add_argument("--min_len_for_eos", type=int, default=DEFAULTS["min_len_for_eos"])
    ap.add_argument("--disable_ns", action="store_true",
                help="Pure model decoding: no lexicon/trie, no LM, no postprocess, no CTC re-rank.")


    args = ap.parse_args()
    if args.disable_ns:
        args.use_ctc_rerank = False


    # ---------- Load checkpoint & backbone ----------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    llm_name = ckpt.get("llm_name", ckpt.get("config", {}).get("model_name_or_path", "meta-llama/Llama-3.2-3B"))
    base_llm = AutoModel.from_pretrained(llm_name).to(DEVICE)
    hidden = base_llm.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Build EMGTextModel and load heads/adapter ----------
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
    model.eval()

    # ---------- Data loaders ----------
    # 1) pre-scan to collect references for LM/lexicon when files not provided
    loader_scan = make_loader(args.data_dir, args.normalizer_path,
                              batch_size=1, shuffle=False, num_workers=0,
                              strip_suffix=args.strip_suffix)
    refs_for_lm: List[str] = []
    names_scan: List[str] = []
    for batch in loader_scan:
        refs_for_lm.append(batch["txts"][0])
        names_scan.append(batch["names"][0])

    # 2) actual decoding loader
    loader = make_loader(args.data_dir, args.normalizer_path,
                         batch_size=args.batch_size, shuffle=False, num_workers=0,
                         strip_suffix=args.strip_suffix)

    # ---------- Build prompt embeds ----------
    def make_prompt_embeds(bsz: int) -> torch.Tensor:
        ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        emb = model.llm.get_input_embeddings()(ids)  # (1, P, D)
        return emb.expand(bsz, -1, -1)              # (B, P, D)

    # ---------- Lexicon / Trie ----------
    lex_words: List[str] = []
    if not args.disable_ns:
        lex_path = Path(args.lexicon)
        if lex_path.exists():
            lex_words += [l.strip().lower() for l in lex_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        phrase_path = Path(args.phrasebook)
        if phrase_path.exists():
            lex_words += [l.strip().lower() for l in phrase_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lex_words:
            raise ValueError(
                "No lexicon/phrasebook provided. Build them from TRAIN only:\n"
                "  python make_lexicon.py --src_dirs data/train_emg "
                "--out_lexicon data/lexicon.txt --out_phrasebook data/phrasebook.txt\n"
                "…then pass: --lexicon data/lexicon.txt --phrasebook data/phrasebook.txt"
            )
        lex_trie = build_trie_from_words(lex_words)
    else:
        lex_trie = None  # not used

    # ---------- Language Model ----------
    lm = None
    if not args.disable_ns:
        if args.lm_arpa and Path(args.lm_arpa).exists():
            try:
                lm = KenLMWrapper(args.lm_arpa)
            except Exception:
                lm = None
        if lm is None:
            lm = CharNgramLM(order=args.lm_order)
            lm.fit(refs_for_lm)

    # ---------- Decode & collect ----------
    names, refs, hyps = [], [], []

    for batch in loader:
        emg = batch["emg"].to(DEVICE)     # (B=1, T, C)
        ref = batch["txts"][0]
        name = batch["names"][0]
        names.append(name)
        refs.append(ref)

        prompt_embeds = make_prompt_embeds(emg.size(0))
        context = model.encode_context(prompt_embeds, emg)  # (1, S, D)

        # step_fn closure for constrained beam
        def step_fn(ids_list: List[int]) -> torch.Tensor:
            in_ids = torch.tensor(ids_list, dtype=torch.long, device=DEVICE).unsqueeze(0)
            return model.step(context, in_ids)  # (1, V)

        # ---------- NS OFF: pure model decoding ----------
        if args.disable_ns:
            if args.beam <= 1:
                ids = greedy_decode(step_fn, TT, max_new_tokens=args.max_new_tokens, min_len_for_eos=1)
            else:
                ids = beam_decode(step_fn, TT,
                                beam=max(2, args.beam),
                                max_new_tokens=args.max_new_tokens,
                                length_penalty_alpha=args.length_penalty_alpha,
                                min_len_for_eos=1)
            hyp = ids_to_text(TT, ids)
            hyps.append(hyp)
            continue  # skip NS path below
        
        # Constrained AR beam + LM
        ids = constrained_beam_decode(
            step_fn, TT,
            beam=max(2, args.beam),
            max_new_tokens=args.max_new_tokens,
            trie=lex_trie,
            lm=lm,
            lm_weight=args.ns_lm_weight,
            oov_penalty=args.ns_oov_penalty,
            # NEW:
            word_bonus=args.word_bonus,
            length_penalty_alpha=args.length_penalty_alpha,
            min_len_for_eos=args.min_len_for_eos,
        )

        hyp_ar = ids_to_text(TT, ids)

        # Rule-based post-correction
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

            # === AR + LM joint re-ranking (uses the same step_fn as the beam) ===
            def ar_logprob(text: str) -> float:
                if not text:
                    return -1e9
                # text -> char ids (no BOS/EOS inside; we add them)
                try:
                    char_ids = [TT.VOCAB_OFFSET + TT.BASE_CHARS.index(ch) for ch in text]
                except ValueError:
                    # skip unknown chars safely
                    char_ids = [TT.VOCAB_OFFSET + TT.BASE_CHARS.index(ch) for ch in text if ch in TT.BASE_CHARS]
                seq = [TT.BOS_IDX] + char_ids + [TT.EOS_IDX]

                tot = 0.0
                for i in range(1, len(seq)):
                    logits = step_fn(seq[:i])                              # (1, V)
                    tot += F.log_softmax(logits, dim=-1)[0, seq[i]].item() # AR token logprob
                return float(tot)

            def joint_score(s: str) -> float:
                lm_part = float(lm.logp(s)) if lm is not None else 0.0
                return ar_logprob(s) + args.ns_lm_weight * lm_part

            cand = [(hyp_ar, joint_score(hyp_ar)), (hyp_ctc, joint_score(hyp_ctc))]
            hyp  = max(cand, key=lambda x: x[1])[0]

        hyps.append(hyp)

    # ---------- Save predictions & print WER ----------
    import pandas as pd
    df = pd.DataFrame({"file": names, "reference": refs, "prediction": hyps})
    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)

    w = compute_wer(refs, hyps)
    print(f"Corpus WER (NS-constrained AR, beam={args.beam}, CTC_rerank={args.use_ctc_rerank}): {w:.4f}")
    print(f"Wrote: {out.resolve()}")

if __name__ == "__main__":
    main()
