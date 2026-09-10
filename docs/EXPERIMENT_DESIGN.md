# Experiment design and validity contract

## Claim

For residual-stream activations at matched layers, independently construct a
J-Lens vocabulary distribution and an SAE vocabulary distribution, then
compare them with each other and with the observed next token. Neither method
is used to construct the other's distribution.

This design does **not** claim that SAE features literally are tokens. SAE
features are marked "token-associated" only when an individual decoder
direction produces a sufficiently concentrated vocabulary readout under a
fixed direct-unembedding rule. The percentage failing that rule is reported.

## Units and distributions

- Unit of analysis: a non-special source token position with its observed next
  token as target.
- J-Lens distribution: `softmax(unembed(J_l h_l))`.
- SAE distribution: `softmax(unembed(SAE_l(h_l)))`.
- Direct-logit baseline: `softmax(unembed(h_l))`.
- Model distribution: the model's ordinary logits at that source position.
- Frequency-adjusted variants subtract `alpha * log(unigram_probability)`
  before softmax. Raw and adjusted results are both retained.

## Primary outcomes

1. Jensen-Shannon similarity between the independent J-Lens and SAE distributions.
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

Feature token association is descriptive, not the primary metric. It is
computed by passing normalized SAE decoder directions through the model's
ordinary final norm and unembedding; J-Lens is never involved. The primary
SAE result uses the complete SAE reconstruction, with random, permuted, and
direct-logit controls.

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
