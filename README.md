# Silent-Speech: Facial EMG → Text with LLM + NeuroSymbolic Decoding

This repository implements **silent speech recognition from facial EMG** by conditioning a **frozen decoder-only LLM** on EMG embeddings. Only lightweight modules are trained: an **EMG adapter**, a **soft prompt**, and an **autoregressive (AR) character head** (optional **CTC head**).  
At inference, **NeuroSymbolic (NS) decoding** improves reliability using **lexicon/trie constraints** and **character 5-gram fusion**, with tunable boundary control (**β, κ, γ**) to reduce spelling/spacing errors and insertions.

Optionally, the pipeline can generate **audio** from predicted text using a CPU **Text-to-Speech (TTS)** script.

---

## Features
- **EMG → Text** with a frozen decoder-only LLM conditioned on EMG embeddings
- Trainable lightweight modules:
  - EMG adapter
  - soft prompt
  - AR head (optional CTC head)
- **NeuroSymbolic decoding** (inference-time control):
  - lexicon/trie constraints
  - character 5-gram fusion (**β**)
  - word bonus (**κ**) + OOV penalty (**γ**)
  - optional CTC candidate proposals + NS reranking
- Optional **TTS** to generate `.wav` files from predictions

---

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```


## EMG → Text
Prepare splits
```bash
python split_data.py
```

## Train
```bash
python train_emg_llm.py
```

## Inference
```bash
python inference_emg_llm.py \
  --ckpt artifacts/checkpoint.pt \
  --normalizer artifacts/emg_norm.pkl \
  --val_dir data/val_emg \
  --output predictions.xlsx
```

## Outputs
- predictions.xlsx — text predictions
- artifacts/ — checkpoints, normalizer, logs, and other run artifacts


## NeuroSymbolic Decoding (NS)

NS decoding is inference-time control that combines neural scores with structured constraints:
Trie/lexicon constraint: blocks invalid spellings/words for closed/template vocabularies
Character 5-gram fusion (β): stabilizes spelling and spaces
Word bonus (κ) + OOV penalty (γ): controls word boundaries and reduces insertions
Optional: CTC candidate proposals + NS reranking for extra robustness

## Audio (Optional)

Generate .wav files (one per prediction row) using the Coqui TTS helper script:
```bash
python scripts/tts_from_predictions_coqui.py \
  --xlsx predictions.xlsx \
  --out_dir artifacts/audio_preds \
  --col prediction
```
