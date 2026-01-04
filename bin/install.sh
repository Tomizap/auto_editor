#!/bin/bash
set -e

echo "=== SYSTEM SETUP ==="

apt update
apt install -y \
  software-properties-common \
  ffmpeg \
  git \
  build-essential \
  libsndfile1 \
  libsndfile1-dev \
  fonts-dejavu-core

echo "=== PYTHON 3.10 (deadsnakes) ==="

add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y \
  python3.10 \
  python3.10-venv \
  python3.10-dev

command -v python3.10 >/dev/null || {
  echo "❌ Python 3.10 introuvable"
  exit 1
}

echo "=== PYTHON ENV ==="

rm -rf env
python3.10 -m venv env
source env/bin/activate

pip install --upgrade pip wheel
pip install "setuptools<81"

echo "=== PYTORCH CUDA ==="

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== CORE PYTHON DEPS ==="

pip install numpy
pip install tqdm
pip install pydub
pip install pysubs2
pip install webrtcvad
pip install opencv-python-headless
pip install mediapipe
pip install faster-whisper

echo "=== WHISPERX ==="

pip install git+https://github.com/m-bain/whisperx.git

echo "=== VERIFY ==="

python - <<'PY'
import torch
import mediapipe as mp

print("Python OK")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MediaPipe version:", mp.__version__)
print("MediaPipe solutions OK:", hasattr(mp, "solutions"))
PY

echo "=== MODELS ==="

mkdir -p models
wget -O models/face_landmarker.task \
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task


echo "=== RCLONE ==="
curl https://rclone.org/install.sh | sudo bash

echo "=== INSTALL COMPLETE ==="
