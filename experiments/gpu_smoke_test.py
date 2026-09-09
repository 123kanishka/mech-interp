import time
import torch

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

device = "cuda"

a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

torch.cuda.synchronize()

start = time.time()

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()

print(f"Runtime: {time.time() - start:.2f}s")
print("Smoke test complete.")
