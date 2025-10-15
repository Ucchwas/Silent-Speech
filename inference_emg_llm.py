#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# --- make project root importable no matter where you run this from ---
import sys
ROOT = Path(__file__).resolve().parents[1]   # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.emg_adapter import EMGAdapterV2 as EMGAdapter          
from scripts.data_utils import FeatureNormalizer, TextTransform  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- model helpers ----------
def load_checkpoint(ckpt_path: str | Path):
    return torch.load(ckpt_path, map_location="cpu")


def build_modules(ckpt):
    base_llm = AutoModel.from_pretrained(ckpt["llm_name"]).to(DEVICE)
    hidden = base_llm.config.hidden_size

    TT = TextTransform()
    char_embed = nn.Embedding(TT.VOCAB_SIZE, hidden, padding_idx=TT.PAD_IDX)
    adapter = EMGAdapter(in_dim=112, hidden_size=hidden)
    lm_head = nn.Linear(hidden, TT.VOCAB_SIZE)

    char_embed.load_state_dict(ckpt["state_dict"]["char_embed"])
    adapter.load_state_dict(ckpt["state_dict"]["adapter"])
    lm_head.load_state_dict(ckpt["state_dict"]["lm_head"])

    for m in (adapter, char_embed, lm_head):
        m.to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt["llm_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return base_llm, tokenizer, adapter, char_embed, lm_head, TT


def prompt_embeds(tokenizer, base_llm, prompt: str, bsz: int):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    emb = base_llm.get_input_embeddings()(ids)  # (1, P, D)
    return emb.expand(bsz, -1, -1).contiguous()


@torch.no_grad()
def greedy_decode(base_llm, tokenizer, adapter, char_embed, lm_head, TT, emg, prompt, max_len=128):
    """
    emg: (1, T, C)
    """
    emg_emb = adapter(emg)                             # (1, T', D)
    P = prompt_embeds(tokenizer, base_llm, prompt, 1)  # (1, P, D)

    cur_ids = [TT.BOS_IDX]
    for _ in range(max_len):
        char_in = torch.tensor(cur_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, L)
        char_in_emb = char_embed(char_in)                                              # (1, L, D)
        inp = torch.cat([P, emg_emb, char_in_emb], dim=1)                              # (1, P+T'+L, D)

        out = base_llm(inputs_embeds=inp)
        h_last = out.last_hidden_state[:, -1:, :]  # (1, 1, D)
        logits = lm_head(h_last).squeeze(1)        # (1, V)
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == TT.EOS_IDX:
            break
        cur_ids.append(nxt)

    return TT.ids_to_text(cur_ids[1:])


def read_reference(json_path: Path) -> str | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in ("text", "transcript", "sentence", "label"):
        if isinstance(data.get(k), str):
            return data[k]
    # fallback: first string value
    for v in data.values():
        if isinstance(v, str):
            return v
    return None


def compute_wer(ref: str, hyp: str) -> float:
    # jiwer works on word sequences; we feed cleaned text
    from jiwer import wer
    return float(wer(ref, hyp))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoint.pt")
    ap.add_argument("--normalizer", default="artifacts/emg_norm.pkl")
    ap.add_argument("--val_dir", default="data/val_emg")
    ap.add_argument("--output", default="predictions.xlsx")
    ap.add_argument("--prompt", default="Transcribe the silent speech from EMG: ")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--strip_suffix", default="_silent", help="strip this suffix from .npy stem to find matching .json")
    args = ap.parse_args()

    # load model bits
    ckpt = load_checkpoint(args.ckpt)
    base_llm, tokenizer, adapter, char_embed, lm_head, TT = build_modules(ckpt)

    # normalizer
    norm = FeatureNormalizer()
    norm.load(args.normalizer)

    val_dir = Path(args.val_dir)
    npy_files = sorted(val_dir.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"No .npy files found in {val_dir}")

    rows = []
    for npy in npy_files:
        # find matching JSON (same stem, or stem with/without strip_suffix)
        stem = npy.stem
        cand = [val_dir / f"{stem}.json"]
        if args.strip_suffix and stem.endswith(args.strip_suffix):
            cand.append(val_dir / f"{stem[:-len(args.strip_suffix)]}.json")
        else:
            cand.append(val_dir / f"{stem}{args.strip_suffix}.json")

        json_path = next((p for p in cand if p.exists()), None)

        # load + normalize emg
        x = np.load(npy).astype(np.float32)  # (T, C)
        x = norm.transform(x)
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, T, C)

        # decode
        pred = greedy_decode(
            base_llm, tokenizer, adapter, char_embed, lm_head, TT,
            x, args.prompt, max_len=args.max_len
        )

        # reference text (clean to match training/decoding)
        ref_raw = read_reference(json_path) if json_path else None
        ref_clean = TT.clean(ref_raw) if isinstance(ref_raw, str) else ""

        # per-sample WER (skip if no ref)
        wer_val = compute_wer(ref_clean, pred) if ref_clean else None

        rows.append({
            "file": npy.name,
            "json": json_path.name if json_path else "",
            "reference_raw": ref_raw if isinstance(ref_raw, str) else "",
            "reference": ref_clean,
            "prediction": pred,
            "wer": wer_val,
        })

    df = pd.DataFrame(rows)
    # overall WER on rows with references
    if (df["reference"] != "").any():
        overall = compute_wer(" ".join(df.loc[df.reference != "", "reference"]),
                              " ".join(df.loc[df.reference != "", "prediction"]))
        print(f"Files: {len(df)}  |  references: {(df.reference!='').sum()}  |  overall WER: {overall:.4f}")
    else:
        print(f"Files: {len(df)}  |  no references found; skipping WER.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"✓ Saved predictions → {out_path.resolve()}")

if __name__ == "__main__":
    main()
