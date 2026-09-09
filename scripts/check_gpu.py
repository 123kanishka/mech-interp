import torch

print("=" * 60)
print("MECH-INTERP GPU CHECK")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU not detected.")

print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

props = torch.cuda.get_device_properties(0)

print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")

x = torch.randn((2048, 2048), device="cuda")
y = x @ x

print("GPU tensor:", y.device)
print("GPU computation: SUCCESS")
