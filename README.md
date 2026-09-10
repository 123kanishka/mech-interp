# SAE--Jacobian-Lens Comparison

Reproducible MATS mini-project comparing information exposed by sparse
autoencoders (SAEs), the Jacobian lens (J-Lens), and a language model's actual
next-token predictions.

## Research question

At matched residual-stream layers and token positions, how closely do the
token labels of the strongest fired SAE features match J-Lens's top poised-to-
verbalize tokens? Which ranked token set better recovers the following observed
tokens, and how does that relationship change by layer?

The methods do not feed into one another. J-Lens transports the original
activation with its fitted Jacobian. SAE independently selects its ten strongest
positive features, directly unembeds each normalized decoder direction to one
token label, and weights those labels by feature activation.

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
# From the laptop, using the host and SSH port shown by Vast:
./scripts/provision_vast_from_github.sh \
  VAST_HOST VAST_PORT "$(git rev-parse HEAD)"

# Then add/update the vast-mech SSH alias and connect:
ssh vast-mech
cd /workspace/mech-interp
./scripts/start_experiment.sh experiments/run_sae_jlens.py \
  --config configs/sae_jlens.yaml --stage preflight
./scripts/experiment_status.sh
./scripts/sync_from_vast.sh
```

`provision_vast_from_github.sh` clones the public GitHub repository and checks
out the exact tested commit. It installs dependencies but deliberately does
not download model, SAE, J-Lens, or dataset artifacts; those downloads begin
only when the preflight is launched. Hugging Face artifacts are cached under
`/workspace/.hf_home`, matching the Vast base image, so duplicate model caches
are not created.

Do not launch `full` until the preflight output has been inspected and its
runtime, peak VRAM, disk use, and projected cost have been recorded.
