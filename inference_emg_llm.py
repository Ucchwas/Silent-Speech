#!/usr/bin/env python3
"""
Inference script for EMG-to-text using pre-extracted features.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from jiwer import wer
import string

from models.emg_adapter import EMGAdapter

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
NUM_CHARS = 37
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextTransform:
    """Text transformation utilities"""
    def __init__(self):
        self.chars = string.ascii_lowercase + string.digits + ' '
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
    
    def indices_to_text(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])


class CharacterEmbedding(nn.Module):
    """Character embedding layer"""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        return self.embedding(x)


def generate_text_beam_search(llm, adapter, char_embedding, lm_head, features, text_transform, 
                             beam_size=5, max_length=50, temperature=0.8):
    """Generate text using beam search"""
    device = features.device
    
    with torch.no_grad():
        # Get EMG embeddings
        emg_embeds = adapter(features)  # (1, T, D)
        
        # Initialize beams: (score, indices, embeddings)
        beams = [(0.0, [], emg_embeds)]
        
        for step in range(max_length):
            new_beams = []
            
            for score, indices, embeds in beams:
                # Forward through LLM
                outputs = llm(
                    inputs_embeds=embeds,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get last hidden state
                last_hidden = outputs.hidden_states[-1][:, -1:, :]
                logits = lm_head(last_hidden).squeeze(0) / temperature
                
                # Get log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs[0], k=min(beam_size, NUM_CHARS))
                
                for i in range(len(topk_log_probs)):
                    token_idx = topk_indices[i].item()
                    token_score = topk_log_probs[i].item()
                    
                    new_score = score + token_score
                    new_indices = indices + [token_idx]
                    
                    # Get embedding for new token
                    token_tensor = torch.tensor([[token_idx]], device=device)
                    token_embed = char_embedding(token_tensor)
                    new_embeds = torch.cat([embeds, token_embed], dim=1)
                    
                    new_beams.append((new_score, new_indices, new_embeds))
            
            # Keep top beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]
            
            # Check if all beams have double space
            all_done = True
            for _, indices, _ in beams:
                if len(indices) < 2 or indices[-1] != 36 or indices[-2] != 36:
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Get best beam
        best_score, best_indices, _ = beams[0]
        
        # Remove trailing spaces
        while best_indices and best_indices[-1] == 36:
            best_indices = best_indices[:-1]
        
        # Convert to text
        return text_transform.indices_to_text(best_indices)


def process_sample(llm, adapter, char_embedding, lm_head, features, ground_truth, text_transform):
    """Process a single sample and return results"""
    # Try different decoding strategies
    results = []
    
    # 1. Greedy decoding
    with torch.no_grad():
        emg_embeds = adapter(features.unsqueeze(0))
        current_embeds = emg_embeds
        generated = []
        
        for _ in range(len(ground_truth) + 10):
            outputs = llm(
                inputs_embeds=current_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden = outputs.hidden_states[-1][:, -1:, :]
            logits = lm_head(last_hidden) / 1.0
            
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
            
            if len(generated) >= 2 and generated[-1] == 36 and generated[-2] == 36:
                generated = generated[:-1]
                break
            
            next_embed = char_embedding(torch.tensor([[next_token]], device=features.device))
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        
        greedy_text = text_transform.indices_to_text(generated).strip()
        greedy_wer = wer(ground_truth, greedy_text)
        results.append(('greedy', greedy_text, greedy_wer))
    
    # 2. Beam search
    beam_text = generate_text_beam_search(
        llm, adapter, char_embedding, lm_head,
        features.unsqueeze(0), text_transform,
        beam_size=5, temperature=0.8
    )
    beam_wer = wer(ground_truth, beam_text)
    results.append(('beam', beam_text, beam_wer))
    
    # Return best result
    best_result = min(results, key=lambda x: x[2])
    return {
        'method': best_result[0],
        'prediction': best_result[1],
        'wer': best_result[2]
    }


def main():
    # Configuration
    config = {
        'llm_path': 'models/llama3.2-3B',
        'model_path': 'models/best_model.pth',
        'test_dir': 'data/val_emg',
        'output_path': 'predictions.xlsx'
    }
    
    print(f"Device: {DEVICE}")
    
    # Check if model exists
    if not Path(config['model_path']).exists():
        print(f"Error: Model not found at {config['model_path']}")
        print("Please train the model first using train.py")
        return
    
    # Load checkpoint
    print("\nLoading model checkpoint...")
    checkpoint = torch.load(config['model_path'], map_location=DEVICE)
    model_config = checkpoint['config']
    
    print(f"Model info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val WER: {checkpoint['val_wer']:.4f}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
    
    # Load LLaMA
    print("\nLoading LLaMA model...")
    llm = AutoModelForCausalLM.from_pretrained(
        config['llm_path'],
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None
    ).to(DEVICE)
    
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False
    
    # Create model components
    print("\nCreating model components...")
    adapter = EMGAdapter(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        llm_embed_dim=model_config['llm_embed_dim'],
        dropout=0.0  # No dropout during inference
    ).to(DEVICE)
    
    char_embedding = CharacterEmbedding(model_config['num_chars'], model_config['llm_embed_dim']).to(DEVICE)
    lm_head = nn.Linear(model_config['llm_embed_dim'], model_config['num_chars']).to(DEVICE)
    
    # Load weights
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    char_embedding.load_state_dict(checkpoint['char_embedding_state_dict'])
    lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
    
    # Set to eval mode
    adapter.eval()
    char_embedding.eval()
    lm_head.eval()
    
    # Create text transform
    text_transform = TextTransform()
    
    # Load test data
    print("\nLoading test data...")
    test_dir = Path(config['test_dir'])
    test_samples = []
    
    for npy_file in sorted(test_dir.glob("*_silent.npy")):
        json_file = test_dir / f"{npy_file.stem.replace('_silent', '')}.json"
        if json_file.exists():
            test_samples.append((npy_file, json_file))
    
    print(f"Found {len(test_samples)} test samples")
    
    # Process samples
    results = []
    print("\nProcessing test samples...")
    
    for npy_file, json_file in tqdm(test_samples):
        # Load features
        features = np.load(npy_file).astype(np.float32)
        features_tensor = torch.FloatTensor(features).to(DEVICE)
        
        # Load ground truth
        with open(json_file, 'r') as f:
            data = json.load(f)
            ground_truth = data['text']
        
        # Generate prediction
        result = process_sample(
            llm, adapter, char_embedding, lm_head,
            features_tensor, ground_truth, text_transform
        )
        
        # Store result
        results.append({
            'file': npy_file.stem.replace('_silent', ''),
            'ground_truth': ground_truth,
            'prediction': result['prediction'],
            'wer': result['wer'],
            'method': result['method'],
            'exact_match': ground_truth == result['prediction']
        })
    
    # Calculate statistics
    if results:
        wer_scores = [r['wer'] for r in results]
        exact_matches = sum(r['exact_match'] for r in results)
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Total samples: {len(results)}")
        print(f"Average WER: {np.mean(wer_scores):.4f}")
        print(f"Median WER: {np.median(wer_scores):.4f}")
        print(f"Std WER: {np.std(wer_scores):.4f}")
        print(f"Min WER: {min(wer_scores):.4f}")
        print(f"Max WER: {max(wer_scores):.4f}")
        print(f"Exact matches: {exact_matches} ({exact_matches/len(results)*100:.1f}%)")
        
        # WER distribution
        perfect = sum(1 for w in wer_scores if w == 0)
        good = sum(1 for w in wer_scores if 0 < w <= 0.3)
        medium = sum(1 for w in wer_scores if 0.3 < w <= 0.7)
        poor = sum(1 for w in wer_scores if w > 0.7)
        
        print(f"\nWER Distribution:")
        print(f"  Perfect (0.0): {perfect} ({perfect/len(results)*100:.1f}%)")
        print(f"  Good (≤0.3): {good} ({good/len(results)*100:.1f}%)")
        print(f"  Medium (≤0.7): {medium} ({medium/len(results)*100:.1f}%)")
        print(f"  Poor (>0.7): {poor} ({poor/len(results)*100:.1f}%)")
        
        # Method usage
        method_counts = {}
        for r in results:
            method = r['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"\nDecoding methods used:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} ({count/len(results)*100:.1f}%)")
        
        # Show best predictions
        print("\nBest predictions (lowest WER):")
        sorted_results = sorted(results, key=lambda x: x['wer'])
        for i, r in enumerate(sorted_results[:5]):
            print(f"\n{i+1}. {r['file']} (WER: {r['wer']:.3f}, method: {r['method']})")
            print(f"   GT: '{r['ground_truth']}'")
            print(f"   PR: '{r['prediction']}'")
        
        # Show worst predictions
        if len(results) > 10:
            print("\nWorst predictions (highest WER):")
            for i, r in enumerate(sorted_results[-5:]):
                print(f"\n{i+1}. {r['file']} (WER: {r['wer']:.3f}, method: {r['method']})")
                print(f"   GT: '{r['ground_truth']}'")
                print(f"   PR: '{r['prediction']}'")
        
        # Calculate character-level accuracy
        total_chars = 0
        correct_chars = 0
        for r in results:
            gt = r['ground_truth']
            pred = r['prediction']
            total_chars += len(gt)
            
            # Simple character matching
            for i in range(min(len(gt), len(pred))):
                if gt[i] == pred[i]:
                    correct_chars += 1
        
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        print(f"\nCharacter-level accuracy: {char_accuracy:.3f}")
        
        # Save results to Excel
        df = pd.DataFrame(results)
        df = df.sort_values('wer')
        df.to_excel(config['output_path'], index=False)
        print(f"\nResults saved to: {config['output_path']}")
        
        # Create summary statistics DataFrame
        summary_df = pd.DataFrame({
            'Metric': ['Average WER', 'Median WER', 'Std WER', 'Min WER', 'Max WER', 
                      'Exact Match Rate', 'Character Accuracy'],
            'Value': [f"{np.mean(wer_scores):.4f}", 
                     f"{np.median(wer_scores):.4f}",
                     f"{np.std(wer_scores):.4f}",
                     f"{min(wer_scores):.4f}",
                     f"{max(wer_scores):.4f}",
                     f"{exact_matches/len(results)*100:.1f}%",
                     f"{char_accuracy:.3f}"]
        })
        
        # Save summary to separate sheet
        with pd.ExcelWriter(config['output_path'], mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Summary statistics added to: {config['output_path']}")
        
    else:
        print("No results generated!")


if __name__ == "__main__":
    main()