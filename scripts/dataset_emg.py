import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scripts.data_utils import sliding_window, TextTransform, FeatureNormalizer


def extract_handcrafted_features(emg_signal, window_size=0.027, stride=0.010, sample_rate=800):
    """
    Extract handcrafted features from EMG signal as per the paper.
    
    The paper mentions extracting 112 time-varying features including:
    - Temporal features (from Jou et al., 2006)
    - Spectral features (from Gaddy and Klein, 2020)
    
    For simplicity, we'll implement common EMG features:
    - Root Mean Square (RMS)
    - Mean Absolute Value (MAV)
    - Zero Crossing Rate (ZCR)
    - Waveform Length (WL)
    - Variance
    - Standard Deviation
    - Min/Max values
    - Spectral features (FFT-based)
    """
    # Convert window size and stride from seconds to samples
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)
    
    features_list = []
    
    # Sliding window over the signal
    for i in range(0, len(emg_signal) - window_samples + 1, stride_samples):
        window = emg_signal[i:i + window_samples]
        
        # Features for each channel
        channel_features = []
        for ch in range(window.shape[1]):
            ch_data = window[:, ch]
            
            # Time domain features
            rms = np.sqrt(np.mean(ch_data ** 2))
            mav = np.mean(np.abs(ch_data))
            var = np.var(ch_data)
            std = np.std(ch_data)
            min_val = np.min(ch_data)
            max_val = np.max(ch_data)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(ch_data)) != 0)
            zcr = zero_crossings / len(ch_data)
            
            # Waveform length
            wl = np.sum(np.abs(np.diff(ch_data)))
            
            # Frequency domain features (simple FFT-based)
            fft = np.fft.rfft(ch_data)
            fft_mag = np.abs(fft)
            
            # Spectral features
            spectral_mean = np.mean(fft_mag)
            spectral_std = np.std(fft_mag)
            spectral_max = np.max(fft_mag)
            
            # Power in different frequency bands
            freqs = np.fft.rfftfreq(len(ch_data), 1/sample_rate)
            low_power = np.sum(fft_mag[freqs < 50])
            mid_power = np.sum(fft_mag[(freqs >= 50) & (freqs < 150)])
            high_power = np.sum(fft_mag[freqs >= 150])
            
            # Combine features for this channel
            features = [
                rms, mav, var, std, min_val, max_val, zcr, wl,
                spectral_mean, spectral_std, spectral_max,
                low_power, mid_power, high_power
            ]
            
            channel_features.extend(features)
        
        features_list.append(channel_features)
    
    return np.array(features_list)


class ClosedVocabEMGDataset(Dataset):
    """
    Dataset for closed vocabulary EMG-to-text conversion.
    Follows the paper's approach with proper normalization.
    """
    
    def __init__(
        self,
        data_dir,
        normalizer_path=None,
        window_size=14,
        hop_size=7,
        use_handcrafted=False,
        fit_normalizer=True,
        max_norm_samples=None
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.hop_size = hop_size
        self.use_handcrafted = use_handcrafted
        self.text_transform = TextTransform()
        
        # Collect all file pairs
        self.files = []
        for npy_file in sorted(self.data_dir.glob("*_silent.npy")):
            json_file = self.data_dir / f"{npy_file.stem.replace('_silent', '')}.json"
            if json_file.exists():
                self.files.append((npy_file, json_file))
        
        print(f"Found {len(self.files)} samples in {data_dir}")
        
        # Handle normalizer
        self.normalizer = None
        if normalizer_path and Path(normalizer_path).exists():
            with open(normalizer_path, 'rb') as f:
                self.normalizer = pickle.load(f)
            print(f"Loaded normalizer from {normalizer_path}")
        elif fit_normalizer and len(self.files) > 0:
            # Fit normalizer on subset of data
            print("Fitting normalizer...")
            feature_samples = []
            
            num_samples = min(len(self.files), max_norm_samples or len(self.files))
            for i in range(num_samples):
                npy_file, _ = self.files[i]
                emg = np.load(npy_file)
                
                if self.use_handcrafted:
                    features = extract_handcrafted_features(emg)
                else:
                    # For raw EMG, normalize the full signal
                    features = emg
                
                feature_samples.append(features)
            
            # Concatenate all features
            all_features = np.concatenate(feature_samples, axis=0)
            self.normalizer = FeatureNormalizer([all_features], share_scale=False)
            
            # Save normalizer if path provided
            if normalizer_path:
                Path(normalizer_path).parent.mkdir(parents=True, exist_ok=True)
                with open(normalizer_path, 'wb') as f:
                    pickle.dump(self.normalizer, f)
                print(f"Saved normalizer to {normalizer_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        npy_file, json_file = self.files[idx]
        
        # Load EMG signal
        emg = np.load(npy_file)  # (T, C)
        
        if self.use_handcrafted:
            # Extract handcrafted features
            features = extract_handcrafted_features(emg)
        else:
            # Use raw EMG
            features = emg
        
        # Normalize
        if self.normalizer:
            features = self.normalizer.normalize(features)
        
        # Load text
        with open(json_file, 'r') as f:
            meta = json.load(f)
        text = meta['text']
        
        # Convert text to character indices
        char_indices = self.text_transform.text_to_int(text)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        char_tensor = torch.LongTensor(char_indices)
        
        return features_tensor, char_tensor


def collate_fn(batch):
    """Collate function with proper padding"""
    features, chars = zip(*batch)
    
    # Pad feature sequences
    max_feat_len = max(f.shape[0] for f in features)
    feat_dim = features[0].shape[1]
    
    padded_features = []
    for feat in features:
        pad_len = max_feat_len - feat.shape[0]
        if pad_len > 0:
            feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_len))
        padded_features.append(feat)
    
    # Pad character sequences
    max_char_len = max(c.shape[0] for c in chars)
    padded_chars = []
    for char in chars:
        pad_len = max_char_len - char.shape[0]
        if pad_len > 0:
            char = torch.nn.functional.pad(char, (0, pad_len), value=-100)
        padded_chars.append(char)
    
    features_batch = torch.stack(padded_features)
    chars_batch = torch.stack(padded_chars)
    
    return features_batch, chars_batch


def get_dataloader(
    data_dir,
    normalizer_path=None,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    use_handcrafted=False,
    **kwargs
):
    """Create dataloader for EMG dataset"""
    dataset = ClosedVocabEMGDataset(
        data_dir,
        normalizer_path=normalizer_path,
        use_handcrafted=use_handcrafted,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )