# Experiment design and validity contract

## Claim

For residual-stream activations at matched layers, compare the original
J-Lens readout with the readout after compression through a pretrained SAE.
The experiment estimates how much J-Lens-visible next-token information is
preserved by the SAE and how this varies across depth.

This design does **not** claim that SAE features literally are tokens. SAE
features are marked "token-associated" only when an individual decoder
direction produces a sufficiently concentrated vocabulary readout under a
fixed, preregistered rule. The percentage failing that rule is reported.

## Units and distributions

- Unit of analysis: a non-special source token position with its observed next
  token as target.
- Original distribution: `softmax(unembed(J_l h_l))`.
- SAE distribution: `softmax(unembed(J_l SAE_l(h_l)))`.
- Direct-logit baseline: `softmax(unembed(h_l))`.
- Model distribution: the model's ordinary logits at that source position.
- Frequency-adjusted variants subtract `alpha * log(unigram_probability)`
  before softmax. Raw and adjusted results are both retained.

## Primary outcomes

1. Jensen-Shannon similarity between original and SAE J-Lens distributions.
2. Change by layer in target-token negative log likelihood and reciprocal rank.
3. Rank-biased overlap and top-k overlap.

Secondary outcomes include reconstruction cosine similarity, normalized MSE,
explained variance, active feature count, token-associated feature fraction,
and agreement with the model distribution.

## Controls

- Feature-activation permutation within each evaluated batch.
- Random isotropic residual matched to the SAE reconstruction norm.
- Direct logit lens.
- Full model logits.
- Raw versus unigram-adjusted vocabulary distributions.
- Stratification by target-token frequency and source position.

Random controls use the manifest seed and are stored beside real observations.

## Leakage and circularity precautions

The J-Lens was fitted on WikiText-103, so headline generalization estimates
must also be reported on a distinct held-out corpus before making broad claims.
The WikiText run is a matched-domain replication/characterization, not an
independent generalization result. The full GPU configuration will add a
second corpus after the preflight confirms its license and text schema.

Feature token association is descriptive, not the primary metric, because it
uses the same J-Lens readout as the comparison. The primary SAE result is
information preservation under reconstruction, with random and direct-logit
controls.

## Execution gates

1. Synthetic run and unit tests pass locally.
2. GPU preflight confirms exact model, SAE hook points, J-Lens dimensions,
   finite outputs, and correct source/target offset.
   It refuses artifact downloads unless at least 20 GiB is free on the target
   filesystem.
3. Inspect raw prompts and at least 30 scored examples.
4. Record peak VRAM, examples/second, projected disk, runtime, and rental cost.
5. Only then approve and launch the full run.
6. Independently recompute headline metrics from JSONL before writing claims.
