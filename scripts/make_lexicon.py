#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_lexicon.py — Build lexicon/phrasebook for NeuroSymbolic decoding.

Scans training JSON transcripts, cleans text with TextTransform, and writes:
  - data/lexicon.txt     (unique words meeting frequency/length thresholds)
  - data/phrasebook.txt  (optional manual additions only, or kept if exists)

Usage (typical):
  python scripts/make_lexicon.py \
      --src_dirs data/train_emg \
      --out_lexicon data/lexicon.txt \
      --out_phrasebook data/phrasebook.txt \
      --min_freq 1 --min_len 1

Optional extras:
  --add_words extra_words.txt           # appended to lexicon
  --phrasebook_add my_phrasebook.txt    # appended to phrasebook
  --dump_texts artifacts/train_texts.txt  # save cleaned refs (one per line)
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, List

# Robust imports (repo-style → flat fallback)
try:
    from scripts.data_utils import TextTransform
except Exception:
    from data_utils import TextTransform


def read_reference(json_path: Path) -> Optional[str]:
    """Read a reference string from a JSON sidecar."""
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Common keys, then any string value as fallback
    for k in ("text", "transcript", "sentence", "label"):
        v = data.get(k)
        if isinstance(v, str):
            return v
    for v in data.values():
        if isinstance(v, str):
            return v
    return None


def iter_jsons(src_dirs: Iterable[str]) -> Iterable[Path]:
    """Yield all *.json paths under the given source directories."""
    for d in src_dirs:
        p = Path(d)
        if not p.exists():
            continue
        # Prefer sidecars next to *_silent.npy, but accept any *.json
        # 1) sidecars for *_silent.npy
        for npy in p.glob("*.npy"):
            j1 = npy.with_suffix(".json")
            if j1.exists():
                yield j1
            stem = npy.stem
            # also try stripping/adding _silent
            if stem.endswith("_silent"):
                j2 = p / f"{stem[:-7]}.json"
                if j2.exists():
                    yield j2
            else:
                j3 = p / f"{stem}_silent.json"
                if j3.exists():
                    yield j3
        # 2) any other *.json
        for j in p.glob("*.json"):
            yield j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dirs", nargs="+", default=["data/train_emg"],
                    help="Directories containing *.npy + *.json pairs (use TRAIN only to avoid leakage).")
    ap.add_argument("--out_lexicon", default="data/lexicon.txt")
    ap.add_argument("--out_phrasebook", default="data/phrasebook.txt")
    ap.add_argument("--min_freq", type=int, default=1, help="Minimum count to include a word.")
    ap.add_argument("--min_len", type=int, default=1, help="Minimum length to include a word.")
    ap.add_argument("--max_words", type=int, default=200000, help="Cap number of lexicon entries.")
    ap.add_argument("--add_words", type=str, default="", help="Optional file with extra words to append to lexicon.")
    ap.add_argument("--phrasebook_add", type=str, default="", help="Optional file with extra phrasebook entries.")
    ap.add_argument("--dump_texts", type=str, default="", help="Optional path to save cleaned references.")
    args = ap.parse_args()

    TT = TextTransform()

    # Collect and clean all references
    texts: List[str] = []
    seen_jsons = set()
    for jpath in iter_jsons(args.src_dirs):
        # Avoid duplicates if discovered multiple ways
        if jpath in seen_jsons:
            continue
        seen_jsons.add(jpath)
        raw = read_reference(jpath)
        if isinstance(raw, str):
            texts.append(TT.clean(raw))

    if not texts:
        raise SystemExit(f"No references found under: {args.src_dirs}")

    # Optionally dump cleaned refs
    if args.dump_texts:
        out_texts = Path(args.dump_texts)
        out_texts.parent.mkdir(parents=True, exist_ok=True)
        out_texts.write_text("\n".join(texts) + "\n", encoding="utf-8")

    # Word frequency
    cnt = Counter()
    for t in texts:
        for w in t.split():
            if len(w) >= args.min_len:
                cnt[w] += 1

    # Build lexicon list with thresholds
    lex_words = [w for w, c in cnt.items() if c >= args.min_freq]
    # Keep within TextTransform charset to avoid unreachable tokens
    allowed_chars = set(TT.BASE_CHARS)
    lex_words = [w for w in lex_words if all(ch in allowed_chars for ch in w)]
    # Sort by (-freq, word) and cap
    lex_words.sort(key=lambda w: (-cnt[w], w))
    if args.max_words and args.max_words > 0:
        lex_words = lex_words[: args.max_words]

    # Append optional extra words
    if args.add_words:
        extra_path = Path(args.add_words)
        if extra_path.exists():
            extras = [l.strip().lower() for l in extra_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            # filter to allowed chars
            extras = [w for w in extras if all(ch in allowed_chars for ch in w)]
            lex_words = list(dict.fromkeys(lex_words + extras))  # dedupe preserving order

    # Write lexicon
    out_lex = Path(args.out_lexicon)
    out_lex.parent.mkdir(parents=True, exist_ok=True)
    out_lex.write_text("\n".join(lex_words) + ("\n" if lex_words else ""), encoding="utf-8")

    # Build/append phrasebook (manual additions only; we don't auto-guess proper nouns after cleaning)
    phrase_entries: List[str] = []
    if args.phrasebook_add:
        pb_path = Path(args.phrasebook_add)
        if pb_path.exists():
            phrase_entries = [l.strip().lower() for l in pb_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_pb = Path(args.out_phrasebook)
    out_pb.parent.mkdir(parents=True, exist_ok=True)
    # If a phrasebook already exists, preserve its existing lines
    if out_pb.exists():
        existing = [l.strip().lower() for l in out_pb.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        existing = []
    merged_pb = list(dict.fromkeys(existing + phrase_entries))
    out_pb.write_text("\n".join(merged_pb) + ("\n" if merged_pb else ""), encoding="utf-8")

    # Console summary
    total_refs = len(texts)
    total_tokens = sum(len(t.split()) for t in texts)
    print(f"✓ Built lexicon: {len(lex_words)} words  |  refs: {total_refs}  |  tokens: {total_tokens}")
    print(f"→ {out_lex.resolve()}")
    print(f"→ {out_pb.resolve()}")

if __name__ == "__main__":
    main()
