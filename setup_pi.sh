#!/usr/bin/env bash
set -euo pipefail

# Run from repo root on the Pi:
#   chmod +x setup_pi.sh
#   ./setup_pi.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Checking architecture..."
uname -m

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is intended for ARM64 (aarch64)."
fi

echo "[2/6] Installing system packages..."
sudo apt update
sudo apt install -y \
  build-essential wget curl git ca-certificates \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev liblzma-dev libncursesw5-dev tk-dev uuid-dev xz-utils \
  libatlas-base-dev libopenblas-dev libjpeg-dev

if command -v python3.11 >/dev/null 2>&1; then
  echo "[3/6] Python 3.11 already installed: $(python3.11 --version)"
else
  echo "[3/6] Building Python 3.11.11 (this can take a while)..."
  cd /tmp
  wget -q https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz
  tar -xzf Python-3.11.11.tgz
  cd Python-3.11.11
  ./configure --enable-optimizations
  make -j"$(nproc)"
  sudo make altinstall
  echo "Installed: $(python3.11 --version)"
  cd "$PROJECT_DIR"
fi

echo "[4/6] Creating Python 3.11 virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate
python --version

echo "[5/6] Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt

echo "[6/6] Done."
echo "Activate and run:"
echo "  cd \"$PROJECT_DIR\""
echo "  source .venv/bin/activate"
echo "  python inference_full_pi.py"
