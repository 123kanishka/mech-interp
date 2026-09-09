#!/usr/bin/env bash
set -e

echo "======================================"
echo " Bootstrapping Vast mech-interp setup"
echo "======================================"

cd "$(dirname "$0")/.."

# Reuse the PyTorch/CUDA packages from Vast's base environment.
if [ ! -d ".venv" ]; then
    python3 -m venv --system-site-packages .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip

echo
echo "Installing research dependencies..."
pip install -r requirements.txt

# Keep Hugging Face downloads in one predictable location.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo
echo "Checking GPU..."
python scripts/check_gpu.py

echo
echo "Checking mech-interp packages..."
python scripts/check_environment.py

echo
echo "======================================"
echo " Vast environment is ready."
echo "======================================"
