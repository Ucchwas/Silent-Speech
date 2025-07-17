import string
import numpy as np
import jiwer
from unidecode import unidecode

# ─── Sliding window helper ──────────────────────────────────────────────

def sliding_window(x: np.ndarray, win: int = 14, hop: int = 7) -> np.ndarray:
    segs = []
    for i in range(0, len(x) - win + 1, hop):
        segs.append(x[i : i + win])
    return np.stack(segs, axis=0)  # (N, win, C)

# ─── Feature normalizer ────────────────────────────────────────────────

class FeatureNormalizer:
    def __init__(self, feature_samples, share_scale: bool = False):
        all_feats = np.concatenate(feature_samples, axis=0)
        self.feature_means   = all_feats.mean(axis=0, keepdims=True)
        if share_scale:
            self.feature_stddevs = all_feats.std()
        else:
            self.feature_stddevs = all_feats.std(axis=0, keepdims=True)

    def normalize(self, sample: np.ndarray) -> np.ndarray:
        return (sample - self.feature_means) / (self.feature_stddevs + 1e-8)

    def inverse(self, sample: np.ndarray) -> np.ndarray:
        return sample * self.feature_stddevs + self.feature_means

# ─── Text helper ───────────────────────────────────────────────────────

class TextTransform:
    def __init__(self):
        self.transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        self.chars = string.ascii_lowercase + string.digits + ' '

    def clean_text(self, text: str) -> str:
        return self.transformation(unidecode(text))

    def text_to_int(self, text: str) -> list[int]:
        text = self.clean_text(text)
        return [self.chars.index(c) for c in text if c in self.chars]

    def int_to_text(self, ints: list[int]) -> str:
        return ''.join(self.chars[i] for i in ints if 0 <= i < len(self.chars))
