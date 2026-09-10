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

if [[ -x /venv/main/bin/python ]]; then
    # Vast's PyTorch image keeps its tested CUDA build here. A nested venv does
    # not inherit packages from this environment and may make pip download a
    # replacement CUDA stack.
    if [[ -e .venv && ! -L .venv ]]; then
        mv .venv ".venv.incompatible.$(date +%Y%m%d_%H%M%S)"
    fi
    ln -sfn /venv/main .venv
elif [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo
echo "[3/5] Installing dependencies..."

python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt

echo
echo "[4/5] Preparing caches..."

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

mkdir -p "$HF_HOME"

echo "Hugging Face cache: $HF_HOME"

echo
echo "[5/5] Running checks..."

python scripts/check_gpu.py
python scripts/check_environment.py --gpu

echo
echo "=========================================="
echo " VAST ENVIRONMENT READY"
echo "=========================================="
