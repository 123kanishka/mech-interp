#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo " VAST MECH-INTERP BOOTSTRAP"
echo "=========================================="

echo
echo "[1/5] Checking NVIDIA GPU..."
nvidia-smi

echo
echo "[2/5] Creating Python environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv --system-site-packages .venv
fi

source .venv/bin/activate

echo
echo "[3/5] Installing dependencies..."

python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt

echo
echo "[4/5] Preparing caches..."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

mkdir -p "$HF_HOME"

echo
echo "[5/5] Running checks..."

python scripts/check_gpu.py
python scripts/check_environment.py --gpu

echo
echo "=========================================="
echo " VAST ENVIRONMENT READY"
echo "=========================================="
