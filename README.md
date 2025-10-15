# Silent-Speech

1. python split_data.py
2. python train_emg_llm.py
3. python inference_emg_llm.py
4. source $HOME/venvs/ttscpu310/bin/activate


cd $HOME
wget https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.tar.gz
tar xzf 1.51.tar.gz
cd espeak-ng-1.51
./autogen.sh
./configure --prefix=$HOME/.local
make -j4
make install

export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export PHONEMIZER_ESPEAK_PATH=$HOME/.local/bin/espeak-ng
espeak-ng --version


5. python scripts/tts_from_predictions_coqui.py



python3.10 -m venv $HOME/venvs/ttscpu310

source $HOME/venvs/ttscpu310/bin/activate

pip install -U pip

pip install --no-cache-dir "numpy==1.26.4" "pandas==2.2.2" "openpyxl==3.1.2" "soundfile==0.12.1"

pip install --no-cache-dir torch==2.5.1+cpu torchaudio==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu

pip install --no-cache-dir TTS==0.22.0

python scripts/tts_from_predictions_coqui.py