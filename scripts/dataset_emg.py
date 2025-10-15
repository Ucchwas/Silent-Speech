import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scripts.data_utils import FeatureNormalizer, TextTransform

def _read_text(json_path: Path) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for k in ["text", "transcript", "sentence", "label"]:
        if k in data and isinstance(data[k], str):
            return data[k]
    for v in data.values():
        if isinstance(v, str):
            return v
    raise ValueError(f"No text field found in {json_path}")

class ClosedVocabEMGDataset(Dataset):
    """
    Directory with pairs: *.npy (EMG (T,C)) and *.json (text).
    If your npy files end with '_silent.npy', use strip_suffix='_silent'.
    """
    def __init__(self,
                 data_dir: str | Path,
                 normalizer_path: str | Path,
                 strip_suffix: str = "",
                 fit_normalizer_if_missing: bool = True):
        self.data_dir = Path(data_dir)
        self.strip_suffix = strip_suffix
        self.text = TextTransform()
        self.pairs: List[Tuple[Path, Path]] = []

        for npy in sorted(self.data_dir.glob("*.npy")):
            stem = npy.stem
            if strip_suffix and stem.endswith(strip_suffix):
                stem = stem[: -len(strip_suffix)]
            js = self.data_dir / f"{stem}.json"
            if js.exists():
                self.pairs.append((npy, js))

        if not self.pairs:
            raise RuntimeError(f"No (.npy, .json) pairs found in {self.data_dir}")

        # Normalizer
        self.normalizer = FeatureNormalizer()
        normalizer_path = Path(normalizer_path)
        if normalizer_path.exists():
            self.normalizer.load(normalizer_path)
        else:
            if not fit_normalizer_if_missing:
                raise FileNotFoundError(f"Missing normalizer at {normalizer_path}")
            feats = [np.load(p[0]) for p in self.pairs]
            self.normalizer.fit(feats)
            normalizer_path.parent.mkdir(parents=True, exist_ok=True)
            self.normalizer.save(normalizer_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        npy_path, js_path = self.pairs[idx]
        x = np.load(npy_path).astype(np.float32)  # (T, C)
        x = self.normalizer.transform(x)
        y_txt = _read_text(js_path)

        ids = self.text.text_to_ids(y_txt, add_bos_eos=True)
        in_ids = torch.tensor(ids[:-1], dtype=torch.long)   # starts with BOS
        out_ids = torch.tensor(ids[1:], dtype=torch.long)   # includes EOS

        x = torch.tensor(x, dtype=torch.float32)
        return {"emg": x, "in_ids": in_ids, "out_ids": out_ids,
                "txt": y_txt, "name": npy_path.stem}

def collate_fn(batch: List[Dict]):
    # EMG pad
    T = [b["emg"].shape[0] for b in batch]
    C = batch[0]["emg"].shape[1]
    maxT = max(T)
    emg = torch.zeros(len(batch), maxT, C, dtype=torch.float32)
    emg_mask = torch.zeros(len(batch), maxT, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["emg"].shape[0]
        emg[i, :t] = b["emg"]
        emg_mask[i, :t] = True

    # Char pad
    L = [len(b["in_ids"]) for b in batch]
    maxL = max(L)
    in_pad = TextTransform.PAD_IDX
    out_pad = -100  # CE ignore
    in_ids = torch.full((len(batch), maxL), in_pad, dtype=torch.long)
    out_ids = torch.full((len(batch), maxL), out_pad, dtype=torch.long)
    char_mask = torch.zeros(len(batch), maxL, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = len(b["in_ids"])
        in_ids[i, :l] = b["in_ids"]
        out_ids[i, :l] = b["out_ids"]
        char_mask[i, :l] = True

    names = [b["name"] for b in batch]
    txts = [b["txt"] for b in batch]
    return {"emg": emg, "emg_mask": emg_mask,
            "in_ids": in_ids, "out_ids": out_ids, "char_mask": char_mask,
            "names": names, "txts": txts}

def make_loader(data_dir, normalizer_path, batch_size=4, shuffle=True, num_workers=0, strip_suffix=""):
    ds = ClosedVocabEMGDataset(data_dir, normalizer_path=normalizer_path, strip_suffix=strip_suffix)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
