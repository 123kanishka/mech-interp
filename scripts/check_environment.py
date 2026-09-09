packages = [
    "torch",
    "transformers",
    "transformer_lens",
    "datasets",
    "accelerate",
    "einops",
    "jaxtyping",
    "numpy",
    "pandas",
    "scipy",
]

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
    raise SystemExit(
        "\nEnvironment check failed: " + ", ".join(failed)
    )

print("\nEnvironment: SUCCESS")
