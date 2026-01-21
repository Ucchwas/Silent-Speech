#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/ns_utils.py — Reusable Neuro-Symbolic utilities for EMG→Text

Provides:
  - Trie and lexicon loaders
  - Lightweight char n-gram LM (and optional KenLM wrapper)
  - Constrained AR beam decoder over character vocab
  - Rule-based post-correction (OOV snapping, repeat/space cleanup)
  - Helpers: ids_to_text, compute_wer, read_reference

These utilities are compatible with your current pipeline:
  * TextTransform is expected to expose: BASE_CHARS (str), PAD_IDX, BOS_IDX, EOS_IDX, VOCAB_OFFSET, VOCAB_SIZE
  * Decoder step_fn should accept a list[int] token history and return a (1,V) tensor of logits
"""
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict
from pathlib import Path
import json
import math
import difflib

import torch
import torch.nn.functional as F


# ======================================================================
#                                Trie
# ======================================================================
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


def load_lexicon(path: str, extra: Optional[str] = None) -> Tuple[Trie, List[str]]:
    """Load lexicon and optional phrasebook; return (trie, word_list)."""
    words: List[str] = []
    p = Path(path)
    if p.exists():
        words += [l.strip().lower() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if extra:
        q = Path(extra)
        if q.exists():
            words += [l.strip().lower() for l in q.read_text(encoding="utf-8").splitlines() if l.strip()]
    # uniq, stable
    words = sorted(list(set(words)))
    return build_trie_from_words(words), words


# ======================================================================
#                         Character n-gram LM
# ======================================================================
class CharNgramLM:
    """
    Add-one smoothed character n-gram LM usable both for:
      - scoring whole strings via logp(text)
      - providing conditional P(next_char | context) via cond_dist
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
            s = f"^{t}$"
            for i in range(len(s) - self.order + 1):
                ngram = s[i:i+self.order]
                ctx = ngram[:-1]
                self.counts[ngram] += 1
                self.context[ctx] += 1
                self.V.update(ngram)
        return self

    def _ctx_tail(self, ctx: str) -> str:
        """Return the last (order-1) chars of ctx, left-padded with '^' if needed."""
        ctx = (ctx or "").lower()
        k = self.order - 1
        if k <= 0:
            return ""
        if len(ctx) >= k:
            return ctx[-k:]
        return ("^" * (k - len(ctx))) + ctx

    def cond_dist(self, context: str, base_chars: str) -> torch.Tensor:
        """
        Probability distribution over base_chars for next char given context.
        Uses add-one smoothing. Returns (K,) float tensor on CPU.
        """
        ctx = self._ctx_tail(context)
        C = self.context.get(ctx, 0)
        Vsz = max(1, len(self.V))
        probs = []
        for ch in base_chars:
            ngram = ctx + ch
            c = self.counts.get(ngram, 0)
            probs.append((c + 1) / (C + Vsz + 1))
        p = torch.tensor(probs, dtype=torch.float32)
        p = p / (p.sum() + 1e-12)
        return p

    def logp(self, text: str) -> float:
        if not text:
            return -1e9
        s = f"^{text.strip().lower()}$"
        logp = 0.0
        for i in range(len(s) - self.order + 1):
            ngram = s[i:i+self.order]
            ctx = ngram[:-1]
            c = self.counts.get(ngram, 0)
            C = self.context.get(ctx, 0)
            Vsz = max(1, len(self.V))
            logp += math.log((c + 1) / (C + Vsz + 1))
        return float(logp)


class KenLMWrapper:
    """Optional: requires `pip install kenlm` and an .arpa model file."""
    def __init__(self, path: str):
        import kenlm  # type: ignore
        self.m = kenlm.Model(path)

    def logp(self, text: str) -> float:
        return float(self.m.score(text.strip().lower(), bos=True, eos=True))


# ======================================================================
#                           Post-processing
# ======================================================================
def postprocess(text: str, lex_trie: Trie, lex_words: List[str]) -> str:
    """
    Deterministic cleanups:
      - collapse whitespace
      - collapse 3+ repeated letters
      - snap OOV words to the nearest lexicon word (edit distance via difflib)
    """
    t = " ".join((text or "").split())
    out = []
    for w in t.split():
        # collapse long repeats (e.g., "hellooo" -> "helloo")
        fixed = []
        for ch in w:
            if len(fixed) >= 2 and fixed[-1] == ch == fixed[-2]:
                continue
            fixed.append(ch)
        w2 = "".join(fixed)
        if not lex_trie.is_word(w2):
            cand = difflib.get_close_matches(w2, lex_words, n=1, cutoff=0.82)
            if cand:
                w2 = cand[0]
        out.append(w2)
    return " ".join(out)


# ======================================================================
#                         Constrained AR Beam
# ======================================================================
def constrained_beam_decode(
    step_fn,                   # callable: (List[int]) -> Tensor[1,V] next-step logits
    TT,                        # TextTransform-like with special IDs & BASE_CHARS
    beam: int,
    max_new_tokens: int,
    trie: Trie,
    lm=None,                   # None | CharNgramLM | KenLMWrapper
    lm_weight: float = 0.3,
    oov_penalty: float = 1.0,
    allow_eos_anytime: bool = True
) -> List[int]:
    """
    Trie-constrained AR beam over character vocab (with optional LM prior).

    We maintain beams: (logprob, token_ids, surface_text).
    Allowed expansions at each step are characters that keep the current word
    as a prefix in the trie, plus a space if the current prefix already forms a word.
    If no expansions are possible (OOV prefix), we fall back to neural top-k.
    EOS can always be taken (configurable).
    """
    beams: List[Tuple[float, List[int], str]] = [(0.0, [TT.BOS_IDX], "")]
    space_id = TT.VOCAB_OFFSET + TT.BASE_CHARS.index(" ")
    V = TT.VOCAB_SIZE

    for _ in range(max_new_tokens):
        candidates: List[Tuple[float, List[int], str]] = []

        for logp, ids, surf in beams:
            if ids[-1] == TT.EOS_IDX:
                candidates.append((logp, ids, surf))
                continue

            logits = step_fn(ids)                # (1, V)
            logprobs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

            # Always allow EOS to prevent run-ons (optional)
            if allow_eos_anytime and TT.EOS_IDX < V:
                lp_eos = float(logprobs[TT.EOS_IDX].item())
                candidates.append((logp + lp_eos, ids + [TT.EOS_IDX], surf))

            # Determine allowed char expansions
            prefix = "" if surf.endswith(" ") else (surf.split()[-1] if surf else "")
            allowed = set(trie.next_chars(prefix))
            if trie.is_word(prefix):
                allowed.add(" ")

            options: List[Tuple[float, int, str]] = []
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
                if lm is not None:
                    add_lp += lm_weight * float(lm.logp(new_text))
                if ch == " " and not trie.is_word(prefix):
                    add_lp -= oov_penalty
                options.append((add_lp, nxt_id, new_text))

            # Fallback if no lexicon-consistent expansion
            if not options:
                k = min(beam, int(logprobs.numel()))
                topk = logprobs.topk(k=k)
                for add_lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    # map idx back to char (best-effort)
                    j = int(idx) - TT.VOCAB_OFFSET
                    ch = TT.BASE_CHARS[j] if 0 <= j < len(TT.BASE_CHARS) else ""
                    new_text = (surf + ch).strip()
                    options.append((float(add_lp) - oov_penalty, int(idx), new_text))

            for add_lp, idx, txt in options:
                candidates.append((logp + add_lp, ids + [idx], txt))

        # prune
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam]
        if all(seq[-1] == TT.EOS_IDX for _, seq, _ in beams):
            break

    best = max(beams, key=lambda x: x[0])
    return best[1]


# ======================================================================
#                               Helpers
# ======================================================================
def ids_to_text(TT, ids: List[int]) -> str:
    """
    Convert token ids (including BOS/EOS) to surface text
    using TT.BASE_CHARS and TT.VOCAB_OFFSET.
    """
    out = []
    for i in ids[1:]:
        if i == TT.EOS_IDX:
            break
        j = i - TT.VOCAB_OFFSET
        if 0 <= j < len(TT.BASE_CHARS):
            out.append(TT.BASE_CHARS[j])
    return "".join(out).strip()


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    import jiwer
    from unidecode import unidecode
    t = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    refs = [t(unidecode(r or "")) for r in refs]
    hyps = [t(unidecode(h or "")) for h in hyps]
    return float(jiwer.wer(refs, hyps))


def read_reference(json_path: Optional[Path]) -> Optional[str]:
    if not json_path or not Path(json_path).exists():
        return None
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in ("text", "transcript", "sentence", "label"):
        if isinstance(data.get(k), str):
            return data[k]
    for v in data.values():
        if isinstance(v, str):
            return v
    return None


__all__ = [
    "Trie",
    "build_trie_from_words",
    "load_lexicon",
    "CharNgramLM",
    "KenLMWrapper",
    "postprocess",
    "constrained_beam_decode",
    "ids_to_text",
    "compute_wer",
    "read_reference",
]
