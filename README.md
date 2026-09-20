# J-Lens Safety Steering

This repository contains two mechanistic-interpretability studies on
`Qwen/Qwen3.5-4B`:

- signed J-Lens token-direction steering for harmful and benign prompts;
- a matched-layer comparison of SAE feature labels and J-Lens token readouts.

The current steering protocol is implemented in `src/jlens_safety` and run by
`experiments/run_jlens_safety_learned.py` with
`configs/jlens_safety_learned.yaml`. It learns 32 signed coefficients and
compares last-position with all-position steering. See
[`docs/JLENS_SIGNED_TOKENS.md`](docs/JLENS_SIGNED_TOKENS.md) for the protocol.

The earlier scalar-steering protocol remains reproducible through
`experiments/run_jlens_safety.py`, `configs/jlens_safety.yaml`, and
[`docs/JLENS_SAFETY.md`](docs/JLENS_SAFETY.md).

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

## SAE/J-Lens experiment

The SAE/J-Lens comparison uses `experiments/run_sae_jlens.py` and
`configs/sae_jlens.yaml`. Its stages are:

1. `synthetic`: CPU validation of metrics, controls, serialization, and plots.
2. `preflight`: a small GPU run that checks artifacts, hook points, shapes,
   memory, and output integrity.
3. `full`: the registered dataset run with resumable JSONL output.

Run the local validation without downloading models:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_sae_jlens.py \
  --config configs/sae_jlens.yaml --stage synthetic
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

GPU stages require their pinned model, lens, SAE, and dataset artifacts. The
full design is in [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

## Vast workflow

```bash
# Use the host and SSH port shown by Vast.
./scripts/provision_vast_from_github.sh \
  VAST_HOST VAST_PORT "$(git rev-parse HEAD)"

ssh vast-mech
cd /workspace/mech-interp
./scripts/start_experiment.sh experiments/run_sae_jlens.py \
  --config configs/sae_jlens.yaml --stage preflight
./scripts/experiment_status.sh
./scripts/sync_from_vast.sh
```

`provision_vast_from_github.sh` installs an exact pushed commit. Pass an SSH key
as its fourth argument when the default `~/.ssh/id_ed25519` is not appropriate.
Model artifacts are downloaded by the experiment stages and cached under
`/workspace/.hf_home`.
