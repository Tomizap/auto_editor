# --- mise à jour et outils systèmes ---
sudo apt update && sudo apt install -y ffmpeg git build-essential libsndfile1 libsndfile1-dev

# --- Python venv ---
python3 -m venv env
source env/bin/activate
pip install --upgrade pip wheel setuptools

# --- Installer PyTorch (CUDA 12.1 example) - ADAPTE si version différente ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Dépendances Python principales ---
pip install faster-whisper webrtcvad pydub numpy tqdm pysubs2 opencv-python-headless mediapipe
pip install git+https://github.com/m-bain/whisperx.git

# --- (optionnel mais utile) font, ffmpeg extras ---
sudo apt install -y fonts-dejavu-core

# Vérification rapide
python - <<'PY'
import torch, sys
print("torch cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
PY

# .hf_cache

# télécharger les emojis twemoji
git clone https://github.com/twitter/twemoji.git
mv twemoji/assets/72x72/* assets/twemoji
rm -rf twemoji

# forge police
sudo apt install fontforge python3-fontforge
fontforge --script tools/build_emoji_font.ff