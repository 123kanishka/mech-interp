# Experiment design and validity contract

## Claim

For residual-stream activations at matched layers, independently construct a
J-Lens vocabulary distribution and an SAE vocabulary distribution, then
compare them with each other and with the observed future-token window. Neither method
is used to construct the other's distribution.

This design does **not** claim that SAE features literally are tokens. SAE
features are marked "token-associated" only when an individual decoder
direction produces a sufficiently concentrated vocabulary readout under a
fixed direct-unembedding rule. The percentage failing that rule is reported.

## Units and distributions

- Unit of analysis: a source-token position and its next 10 observed corpus tokens.
- SAE feature-token distribution: select the 10 strongest positive SAE features;
  independently label each feature with the argmax token obtained by directly
  unembedding its normalized decoder direction; sum duplicate labels and weight
  labels by normalized feature activation.
- J-Lens distribution: compute `softmax(unembed(J_l h_l))`, retain its 10 highest
  probability tokens, and renormalize their probability mass.
- The primary comparison is therefore 10 fired SAE features/labels versus 10
  J-Lens poised-to-verbalize tokens at the same activation.
- SAE-reconstruction and final-model distributions are retained as secondary controls.

## Primary outcomes

1. Weighted Jaccard and Jensen-Shannon similarity between sparse SAE-feature and
   top-10 J-Lens probability mass.
2. Rank-biased overlap between the two ranked token lists.
3. Future-token NDCG@10, recall@10, precision@10, hit@10, reciprocal rank, and
   total probability mass for both methods against the next 10 observed tokens.
4. Every outcome is reported at layers 3, 15, and 27 and pooled across all three.

Secondary outcomes include reconstruction cosine similarity, normalized MSE,
explained variance, active feature count, token-associated feature fraction,
and agreement with the model distribution.

## Controls

- Permutation of token labels across the fired features while retaining weights.
- Random vocabulary labels with the same feature weights.
- Full SAE-reconstruction distribution as a secondary construct-validity control.
- Full model logits as an upper/behavioral reference.

Random controls use the manifest seed and are stored beside real observations.

## Leakage and circularity precautions

The J-Lens was fitted on WikiText-103, so headline generalization estimates
must also be reported on a distinct held-out corpus before making broad claims.
The WikiText run is a matched-domain replication/characterization, not an
independent generalization result. The full GPU configuration will add a
second corpus after the preflight confirms its license and text schema.

Feature token association is the primary SAE construct. It is computed by
passing each normalized SAE decoder direction through the model's ordinary
final norm and unembedding; J-Lens is never involved. J-Lens independently
operates on the original activation. The published compatible Qwen3.5-4B SAE
release contains only layers 3, 15, and 27, so these are all available matched
layers rather than all transformer blocks.

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
