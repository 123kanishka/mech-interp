import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true")
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

packages = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "scipy",
]

if args.gpu:
    packages.extend(["datasets", "sae_lens", "jlens"])

print("=" * 60)
print("MECH-INTERP ENVIRONMENT CHECK")
print("=" * 60)

failed = []

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {package:<22} {version}")
    except Exception as exc:
        failed.append(package)
        print(f"✗ {package:<22} {exc}")

if failed:
    raise SystemExit("\nEnvironment check failed: " + ", ".join(failed))

print("\nEnvironment: SUCCESS")
