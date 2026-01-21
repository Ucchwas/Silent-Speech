import pickle
import string
import numpy as np
from dataclasses import dataclass
from typing import List
import jiwer
from unidecode import unidecode

def sliding_window(x: np.ndarray, win: int = 14, hop: int = 7) -> np.ndarray:
    segs = []
    for i in range(0, len(x) - win + 1, hop):
        segs.append(x[i: i + win])
    return np.stack(segs, axis=0)  # (N, win, C)

@dataclass
class NormState:
    mean: np.ndarray
    std: np.ndarray

class FeatureNormalizer:
    """Per-channel z-score: (x - mean) / (std + eps)"""
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.state: NormState | None = None

    def fit(self, arrays: List[np.ndarray]):
        cat = np.concatenate(arrays, axis=0)  # (sumT, C)
        self.state = NormState(mean=cat.mean(axis=0), std=cat.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.state is not None, "FeatureNormalizer not fitted/loaded"
        return (x - self.state.mean) / (self.state.std + self.eps)

    def save(self, path: str):
        assert self.state is not None
        with open(path, "wb") as f:
            pickle.dump(self.state, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.state = pickle.load(f)
        return self

class TextTransform:
    BASE_CHARS = string.ascii_lowercase + string.digits + " "
    VOCAB_OFFSET = 0
    BOS_IDX = len(BASE_CHARS)
    EOS_IDX = len(BASE_CHARS) + 1
    PAD_IDX = len(BASE_CHARS) + 2
    VOCAB_SIZE = len(BASE_CHARS) + 3

    def __init__(self):
        self._transform = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])

    def clean(self, s: str) -> str:
        return self._transform(unidecode(s))

    def text_to_ids(self, s: str, add_bos_eos: bool = True) -> list[int]:
        s = self.clean(s)
        ids = [self.BASE_CHARS.index(c) for c in s if c in self.BASE_CHARS]
        return ([self.BOS_IDX] + ids + [self.EOS_IDX]) if add_bos_eos else ids

    def ids_to_text(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.EOS_IDX:
                break
            if 0 <= i < len(self.BASE_CHARS):
                out.append(self.BASE_CHARS[i])
        return "".join(out).strip()
