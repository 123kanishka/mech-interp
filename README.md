# SAE--Jacobian-Lens Comparison

Reproducible MATS mini-project comparing information exposed by sparse
autoencoders (SAEs), the Jacobian lens (J-Lens), and a language model's actual
next-token predictions.

## Research question

At matched residual-stream layers and token positions, how closely does the
vocabulary distribution recovered from an SAE reconstruction match the
J-Lens distribution from the original activation? Does either distribution
become more predictive of the actual next token at later layers?

The primary model is `Qwen/Qwen3.5-4B`. Public residual-stream SAEs exist only
at layers 3, 15, and 27, so SAE comparisons are restricted to those matched
layers. J-Lens and direct-logit-lens baselines can be evaluated at every layer.

## Experimental stages

1. `synthetic`: CPU-only pipeline validation. It checks metrics, controls,
   serialization, plots, deterministic seeding, and status markers. It is not
   scientific evidence.
2. `preflight`: a small real-model GPU run that validates artifact versions,
   layer conventions, tensor shapes, memory use, and output integrity.
3. `full`: the preregistered larger dataset run. It writes one JSONL record per
   evaluated position so partial results survive interruption.

Run the local validation without downloading models:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_sae_jlens.py \
  --config configs/sae_jlens.yaml --stage synthetic
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The real GPU stages intentionally fail unless their required artifacts and
dependencies are available. See `docs/EXPERIMENT_DESIGN.md` before running.

## Vast workflow

```bash
./scripts/sync_to_vast.sh
ssh vast-mech
cd /workspace/mech-interp
./scripts/bootstrap_vast.sh
./scripts/start_experiment.sh experiments/run_sae_jlens.py \
  --config configs/sae_jlens.yaml --stage preflight
./scripts/experiment_status.sh
./scripts/sync_from_vast.sh
```

Do not launch `full` until the preflight output has been inspected and its
runtime, peak VRAM, disk use, and projected cost have been recorded.
