#!/usr/bin/env python3
import shutil
from pathlib import Path
import random
import json

# Paths
data_dir = Path("data/extracted_emg_features")
train_dir = Path("data/train_emg")
val_dir = Path("data/val_emg")

# Create directories
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Get all EMG files
emg_files = list(data_dir.glob("*_silent.npy"))
print(f"Found {len(emg_files)} EMG files")

# Shuffle and split
random.seed(42)  # For reproducibility
random.shuffle(emg_files)

train_files = emg_files[:450]
val_files = emg_files[450:500]

print(f"Train set: {len(train_files)} files")
print(f"Validation set: {len(val_files)} files")

# Copy files
def copy_files(file_list, target_dir):
    for emg_file in file_list:
        # Copy EMG file
        shutil.copy2(emg_file, target_dir / emg_file.name)
        
        # Copy corresponding JSON
        json_file = emg_file.parent / f"{emg_file.stem.replace('_silent', '')}.json"
        if json_file.exists():
            shutil.copy2(json_file, target_dir / json_file.name)

print("Copying training files...")
copy_files(train_files, train_dir)

print("Copying validation files...")
copy_files(val_files, val_dir)

print("Done! Data split complete.")