#!/usr/bin/env python3
"""
Turn texts from predictions.xlsx into WAVs using Coqui TTS.

Examples:
  # Single-speaker model (recommended):
  python scripts/tts_from_predictions_coqui.py \
      --xlsx predictions.xlsx \
      --out_dir artifacts/audio_preds \
      --col prediction \
      --model tts_models/en/ljspeech/vits

  # Multi-speaker example:
  # (after first run, script prints available speakers)
  python scripts/tts_from_predictions_coqui.py \
      --xlsx predictions.xlsx \
      --out_dir artifacts/audio_preds_vctk \
      --col prediction \
      --model tts_models/en/vctk/vits --speaker_idx 22
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from TTS.api import TTS


def safe_stem(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:120] or "utt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="predictions.xlsx",
                    help="Path to predictions.xlsx")
    ap.add_argument("--out_dir", default="artifacts/audio_preds",
                    help="Directory to write WAV files")
    ap.add_argument("--col", default="prediction",
                    help="Which column to synthesize: prediction|reference|reference_raw")
    ap.add_argument("--model", default="tts_models/en/ljspeech/vits",
                    help="Coqui TTS model name (use a one-package model; no vocoder arg)")
    ap.add_argument("--speaker", default=None,
                    help="Speaker name (for multi-speaker models)")
    ap.add_argument("--speaker_idx", type=int, default=None,
                    help="Speaker index (for multi-speaker models)")
    ap.add_argument("--gpu", action="store_true",
                    help="Use GPU if available")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not in {args.xlsx}. "
                         f"Available columns: {list(df.columns)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = bool(args.gpu and torch.cuda.is_available())
    tts = TTS(model_name=args.model, progress_bar=False, gpu=use_gpu)

    # Show available speakers for convenience (multi-speaker models)
    try:
        if getattr(tts, "speakers", None):
            print(f"Model has {len(tts.speakers)} speakers. "
                  f"Pass --speaker or --speaker_idx. First 10: {tts.speakers[:10]}")
    except Exception:
        pass

    n = 0
    for i, row in df.iterrows():
        text = str(row.get(args.col, "") or "").strip()
        if not text:
            continue

        base = row["file"] if "file" in df.columns and isinstance(row["file"], str) else f"row_{i}"
        wav = out_dir / f"{safe_stem(Path(base).stem)}__{args.col}.wav"

        synth_kwargs = {}
        # For multi-speaker models, pass one of these if provided
        if args.speaker is not None:
            synth_kwargs["speaker"] = args.speaker
        if args.speaker_idx is not None:
            synth_kwargs["speaker_idx"] = args.speaker_idx

        tts.tts_to_file(text=text, file_path=str(wav), **synth_kwargs)
        n += 1

    if n == 0:
        print("No non-empty texts found to synthesize.")
    else:
        print(f"✓ Synthesized {n} WAV file(s) → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
