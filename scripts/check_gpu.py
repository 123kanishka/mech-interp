import torch

print("=" * 50)
print("MECH-INTERP GPU CHECK")
print("=" * 50)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)

    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")

    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x

    print(f"GPU tensor test: {y.device}")
    print("GPU computation successful ✓")

else:
    print("No CUDA GPU detected.")
