#!/usr/bin/env python3
"""
Training script for EMG-to-text using pre-extracted features from Zenodo dataset.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from jiwer import wer
import string

# Import EMG adapter
from models.emg_adapter import EMGAdapter

# Set environment variable to avoid tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Constants
NUM_CHARS = 37  # a-z (26) + 0-9 (10) + space (1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextTransform:
    """Text to character index transformation"""
    def __init__(self):
        # Define character set: lowercase letters + digits + space
        self.chars = string.ascii_lowercase + string.digits + ' '
        assert len(self.chars) == NUM_CHARS
        
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
    
    def text_to_indices(self, text):
        """Convert text to character indices"""
        text = text.lower()
        # Only include characters in our vocabulary
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
        return indices
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])


class EMGDataset(Dataset):
    """Dataset for pre-extracted EMG features"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.text_transform = TextTransform()
        
        # Collect all valid sample pairs
        self.samples = []
        for npy_file in sorted(self.data_dir.glob("*_silent.npy")):
            json_file = self.data_dir / f"{npy_file.stem.replace('_silent', '')}.json"
            if json_file.exists():
                self.samples.append((npy_file, json_file))
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npy_file, json_file = self.samples[idx]
        
        # Load pre-extracted features (already normalized)
        features = np.load(npy_file).astype(np.float32)  # Shape: (T, 112)
        
        # Load text label
        with open(json_file, 'r') as f:
            data = json.load(f)
            text = data['text']
        
        # Convert text to indices
        char_indices = self.text_transform.text_to_indices(text)
        
        return features, char_indices, text


def collate_fn(batch):
    """Collate function with proper padding"""
    features_list, char_indices_list, texts = zip(*batch)
    
    # Find max lengths
    max_feat_len = max(f.shape[0] for f in features_list)
    max_char_len = max(len(c) for c in char_indices_list)
    
    # Pad features
    batch_features = []
    for feat in features_list:
        pad_len = max_feat_len - feat.shape[0]
        if pad_len > 0:
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        batch_features.append(feat)
    
    # Pad character indices with -100 (ignore index for cross entropy)
    batch_chars = []
    for chars in char_indices_list:
        chars = chars + [-100] * (max_char_len - len(chars))
        batch_chars.append(chars)
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(np.stack(batch_features))
    chars_tensor = torch.LongTensor(batch_chars)
    
    return features_tensor, chars_tensor, texts


class CharacterEmbedding(nn.Module):
    """Character embedding layer"""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, x):
        return self.embedding(x)


def generate_text(llm, adapter, char_embedding, lm_head, features, text_transform, max_length=50):
    """Generate text from EMG features using autoregressive decoding"""
    device = features.device
    
    with torch.no_grad():
        # Get EMG embeddings
        emg_embeds = adapter(features)  # (1, T, D)
        
        # Start with just EMG embeddings
        generated_indices = []
        current_embeds = emg_embeds
        
        for _ in range(max_length):
            # Forward through LLM
            outputs = llm(
                inputs_embeds=current_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get the last hidden state
            last_hidden = outputs.hidden_states[-1][:, -1:, :]  # (1, 1, D)
            
            # Project to vocabulary
            logits = lm_head(last_hidden)  # (1, 1, vocab_size)
            
            # Greedy decoding - take argmax
            next_token = torch.argmax(logits, dim=-1).item()
            generated_indices.append(next_token)
            
            # Check stopping condition (double space)
            if len(generated_indices) >= 2:
                if generated_indices[-1] == 36 and generated_indices[-2] == 36:  # 36 is space
                    generated_indices = generated_indices[:-1]  # Remove extra space
                    break
            
            # Get embedding for next token and append
            next_token_tensor = torch.tensor([[next_token]], device=device)
            next_embed = char_embedding(next_token_tensor)  # (1, 1, D)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        
        # Convert indices to text
        generated_text = text_transform.indices_to_text(generated_indices)
        return generated_text.strip()


def train_one_epoch(llm, adapter, char_embedding, lm_head, train_loader, optimizer, device):
    """Train for one epoch"""
    adapter.train()
    char_embedding.train()
    lm_head.train()
    
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for features, char_indices, texts in progress_bar:
        features = features.to(device)
        char_indices = char_indices.to(device)
        
        # Create mask for valid tokens (not padding)
        valid_mask = (char_indices >= 0)
        
        # Replace -100 with 0 for embedding lookup
        char_indices_clean = char_indices.clone()
        char_indices_clean[~valid_mask] = 0
        
        # Get embeddings
        emg_embeds = adapter(features)  # (B, T_emg, D)
        char_embeds = char_embedding(char_indices_clean)  # (B, T_char, D)
        
        # Teacher forcing: concatenate EMG embeddings with all but last character
        # EMG embeddings + character embeddings (except last one)
        inputs_embeds = torch.cat([
            emg_embeds,
            char_embeds[:, :-1, :]  # All characters except last
        ], dim=1)
        
        # Forward through LLM
        outputs = llm(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # (B, T_total, D)
        
        # Extract hidden states corresponding to character positions
        # Skip the EMG embedding positions
        char_hidden = hidden_states[:, emg_embeds.size(1):, :]  # (B, T_char-1, D)
        
        # Project to vocabulary
        logits = lm_head(char_hidden)  # (B, T_char-1, vocab_size)
        
        # Prepare targets (shift by 1)
        targets = char_indices[:, 1:]  # (B, T_char-1)
        target_mask = valid_mask[:, 1:]  # (B, T_char-1)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, NUM_CHARS),
            targets.reshape(-1),
            ignore_index=-100,
            reduction='none'
        )
        
        # Apply mask and compute mean
        loss = (loss * target_mask.reshape(-1)).sum() / target_mask.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + 
            list(char_embedding.parameters()) + 
            list(lm_head.parameters()), 
            max_norm=0.5
        )
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            correct = ((predictions == targets) * target_mask).sum().item()
            total_correct += correct
            total_tokens += target_mask.sum().item()
        
        # Update metrics
        total_loss += loss.item()
        current_acc = total_correct / total_tokens if total_tokens > 0 else 0
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.3f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, avg_acc


def evaluate(llm, adapter, char_embedding, lm_head, val_loader, text_transform, device):
    """Evaluate model on validation set"""
    adapter.eval()
    char_embedding.eval()
    lm_head.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, _, texts in tqdm(val_loader, desc="Evaluating"):
            features = features.to(device)
            
            # Generate predictions for each sample in batch
            for i in range(features.size(0)):
                pred_text = generate_text(
                    llm, adapter, char_embedding, lm_head,
                    features[i:i+1],
                    text_transform,
                    max_length=len(texts[i]) + 10
                )
                
                all_predictions.append(pred_text)
                all_targets.append(texts[i])
    
    # Calculate WER for each sample
    wer_scores = []
    for pred, target in zip(all_predictions, all_targets):
        score = wer(target, pred)
        wer_scores.append(score)
    
    avg_wer = np.mean(wer_scores)
    
    # Print some examples
    print("\nSample predictions:")
    indices = random.sample(range(len(all_predictions)), min(5, len(all_predictions)))
    for idx in indices:
        print(f"Target: '{all_targets[idx]}'")
        print(f"Pred:   '{all_predictions[idx]}'")
        print(f"WER:    {wer_scores[idx]:.3f}")
        print()
    
    return avg_wer, wer_scores


def main():
    # Configuration
    config = {
        'llm_path': 'models/llama3.2-3B',
        'train_dir': 'data/train_emg',
        'val_dir': 'data/val_emg',
        'batch_size': 8,
        'learning_rate': 1e-5,
        'num_epochs': 250,
        'eval_every': 10,
        'save_dir': 'models',
        'patience': 20
    }
    
    print(f"Device: {DEVICE}")
    print(f"Configuration: {config}")
    
    # Load LLaMA model
    print("\nLoading LLaMA model...")
    tokenizer = AutoTokenizer.from_pretrained(config['llm_path'], local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(
        config['llm_path'],
        local_files_only=True,
        torch_dtype=torch.float32,  # Use float32 for stability
        device_map=None  # We'll move it manually
    ).to(DEVICE)
    
    # Freeze LLM parameters
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False
    
    llm_embed_dim = llm.config.hidden_size
    print(f"LLM embedding dimension: {llm_embed_dim}")
    
    # Create model components
    print("\nCreating model components...")
    adapter = EMGAdapter(
        input_dim=112,
        hidden_dim=256,
        llm_embed_dim=llm_embed_dim,
        dropout=0.1
    ).to(DEVICE)
    
    char_embedding = CharacterEmbedding(NUM_CHARS, llm_embed_dim).to(DEVICE)
    
    # LM head - project from LLM hidden dim to character vocabulary
    lm_head = nn.Linear(llm_embed_dim, NUM_CHARS).to(DEVICE)
    nn.init.xavier_uniform_(lm_head.weight, gain=0.1)
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters()) + \
                   sum(p.numel() for p in char_embedding.parameters()) + \
                   sum(p.numel() for p in lm_head.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = EMGDataset(config['train_dir'])
    val_dataset = EMGDataset(config['val_dir'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW([
        {'params': adapter.parameters(), 'lr': config['learning_rate']},
        {'params': char_embedding.parameters(), 'lr': config['learning_rate']},  
        {'params': lm_head.parameters(), 'lr': config['learning_rate']} 
    ], weight_decay=0.01)

    # Add scheduler here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
        
    # Create text transform
    # In train.py, after creating TextTransform
    text_transform = TextTransform()
    print(f"Vocab size: {len(text_transform.chars)}")
    print(f"Characters: '{text_transform.chars}'")
    print(f"Space index: {text_transform.char_to_idx[' ']}")

    # Test encoding/decoding
    test_text = "january 16 1905"
    indices = text_transform.text_to_indices(test_text)
    decoded = text_transform.indices_to_text(indices)
    print(f"Original: '{test_text}'")
    print(f"Indices: {indices}")
    print(f"Decoded: '{decoded}'")
    
    # Training loop
    best_wer = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            llm, adapter, char_embedding, lm_head,
            train_loader, optimizer, DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        
        # Evaluate
        if epoch % config['eval_every'] == 0:
            val_wer, val_wer_scores = evaluate(
                llm, adapter, char_embedding, lm_head,
                val_loader, text_transform, DEVICE
            )
            print(f"Validation WER: {val_wer:.4f}")

            scheduler.step(val_wer)
            
            # Save best model
            if val_wer < best_wer:
                best_wer = val_wer
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'adapter_state_dict': adapter.state_dict(),
                    'char_embedding_state_dict': char_embedding.state_dict(),
                    'lm_head_state_dict': lm_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_wer': val_wer,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'config': {
                        'input_dim': 112,
                        'hidden_dim': 256,
                        'llm_embed_dim': llm_embed_dim,
                        'num_chars': NUM_CHARS
                    }
                }
                
                save_path = Path(config['save_dir']) / 'best_model.pth'
                torch.save(checkpoint, save_path)
                print(f"Saved best model with WER: {val_wer:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered. Best WER: {best_wer:.4f}")
                break
    
    print(f"\nTraining complete! Best WER: {best_wer:.4f}")


if __name__ == "__main__":
    main()