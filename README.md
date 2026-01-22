## Silent-Speech (EMG → Text → Audio)

End-to-end pipeline for silent speech recognition from multi-channel EMG.
We keep the LLM frozen and train a small EMG adapter + LM head (with soft-prompt and optional CTC aux loss).
Optionally synthesize audio from the predicted text.

🚀 Quick start
1) Split
python scripts/split_data.py

2) Train (EMG → LLM)
python train_emg_llm.py

3) Inference (write predictions.xlsx)
python inference_emg_llm.py \
  --ckpt artifacts/checkpoint.pt \
  --normalizer artifacts/emg_norm.pkl \
  --val_dir data/val_emg \
  --output predictions.xlsx

🔊 (Optional) Text-to-Speech on CPU (separate venv)

Create a lightweight venv for TTS (keeps your training env clean):

python3.10 -m venv $HOME/venvs/ttscpu310
source $HOME/venvs/ttscpu310/bin/activate
pip install -U pip
pip install --no-cache-dir "numpy==1.26.4" "pandas==2.2.2" "openpyxl==3.1.2" "soundfile==0.12.1"
pip install --no-cache-dir torch==2.5.1+cpu torchaudio==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir TTS==0.22.0


Install espeak-ng locally (phonemizer backend for Coqui TTS):

cd $HOME
wget https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.tar.gz
tar xzf 1.51.tar.gz
cd espeak-ng-1.51
./autogen.sh
./configure --prefix=$HOME/.local
make -j4 && make install

# make it discoverable
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export PHONEMIZER_ESPEAK_PATH=$HOME/.local/bin/espeak-ng
espeak-ng --version


Synthesize WAVs from predictions:

# inside the TTS venv
python scripts/tts_from_predictions_coqui.py \
  --xlsx predictions.xlsx \
  --out_dir artifacts/audio_preds \
  --col prediction \
  --model tts_models/en/ljspeech/vits


Switching envs

# leave TTS venv
deactivate
# back to training env (example)
conda activate ss2
# enter TTS venv again
source $HOME/venvs/ttscpu310/bin/activate
