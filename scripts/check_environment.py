packages = [
    "torch",
    "transformers",
    "transformer_lens",
    "datasets",
    "einops",
    "jaxtyping",
    "numpy",
    "pandas",
    "scipy",
]

print("=" * 50)
print("MECH-INTERP ENVIRONMENT CHECK")
print("=" * 50)

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {package:<20} {version}")
    except Exception as e:
        print(f"✗ {package:<20} {e}")
