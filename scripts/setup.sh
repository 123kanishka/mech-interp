cat > scripts/setup.sh <<'EOF'
#!/usr/bin/env bash

set -e

echo "=== Mech-Interp Environment Setup ==="

# Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip

echo
echo "Installing research dependencies..."
pip install -r requirements.txt

echo
echo "Checking PyTorch..."
python - <<'PY'
try:
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

except ImportError:
    print("PyTorch is not installed.")
    print("On Vast, use the PyTorch template.")
PY

echo
echo "Checking TransformerLens..."

python - <<'PY'
import transformer_lens
print("TransformerLens imported successfully.")
PY

echo
echo "=== Setup complete ==="
EOF