# Fresh signed-token steering experiment (schema 3)

This document specifies the schema-3 signed-coefficient protocol. Scientific
claims must come from completed GPU runs, not from CPU tests or pilot data.

## What stays fixed

- Model: pinned Qwen3.5-4B, BF16, non-thinking, greedy generation.
- Shortlisting: at each observed layer, collect the final prompt-position
  activation for training prompts; use the pinned J-lens and the model's final
  normalization/unembedding; average logits separately for harmful and benign
  training prompts; select the top **32 absolute differences**, excluding
  special IDs. No validation or test outcomes choose these tokens.
- The J dictionary is the same linear `W_U[selected_ids] @ J` construction,
  not an exact gradient through RMSNorm or the paper's nonnegative decomposition.
- Median residual norm is the training prompt-end norm at that layer, shared
  between scopes. All-position norms may differ: report relative norms rather
  than asserting every position receives the same percentage perturbation.

## Changed intervention and fitting

For unit dictionary rows `u_i`, apply `h' = h + median_norm * sum(beta_i * u_i)`.
Every beta is signed. All 32 are jointly learned; no individual sign is fixed.
The actual update is constrained to norm <= 0.10; coefficient L2 penalizes
large canceling weights. Correlated token directions mean individual beta
values are not uniquely interpretable causal contributions.

The target model stays frozen and in eval mode. Teacher-forced completion NLL
fits refusal targets on harmful requests and helpful targets on benign ones.
A benign KL penalty compares to the frozen model on the same reference prefix.
Reference completions are capped at 256 tokens for fitting only; counts of capped
references are saved. Evaluation does not use reference completions.

Adam only updates beta. Its moments, beta, step, deterministic row cursor,
reference truncation counts, and elapsed time are saved in pickle-free NPZ files.
Failed partial accumulation is replayed from the last completed saved step.
There is one registered epoch, not an outcome-driven stopping rule.

- `last`: prompt end during prefill; each newly processed token during decoding.
- `all`: every unpadded prompt position during prefill; each newly processed token
  during decoding. No repeated editing of cached past positions.
- Teacher forcing edits positions >= prompt_length-1 for `last`, and all input
  positions for `all`, matching those inference semantics.
- Exploratory `total` norm matching divides the prefill update by sqrt(number
  of edited positions); decode edits are unchanged. This matches per-call
  Frobenius perturbation, not total trajectory energy when response lengths differ.

## Selection and controls

1. Run a development-only engineering and baseline-failure pilot. Both position
   hooks must preserve zero-strength outputs, and backward must update beta.
2. Fit the original prompt difference and shortlist on the new training data.
3. Scalar screening on validation chooses one common layer for every arm. If
   zero steering wins, use the predeclared middle layer. This deliberately tests
   every method at the same layer; it is not a search for each method's optimal layer.
4. At that layer, fit J-token, unembedding-token and random-32-direction controls
   for both scopes and all seeds [11,22,33], with the same fitting budget.
5. Evaluate scalar candidates and every learned fit on the full validation set.
   Zero is eligible; a nonzero deployment selection must reduce validation
   macro harm and meet benign/math constraints. An alphabetic tie is not evidence
   for a negative sign. All learned seeds are retained for scientific evaluation,
   including infeasible or negative results.
6. Freeze directions, coefficients, complete validation outputs and exact test
   jobs. Test generation is deterministic and can be sharded across machines.

Primary test arms: unsteered baseline, scalar last, scalar all, learned J last
for each seed, learned J all for each seed (9 arms with the default seeds).
Controls, residual interventions and total-norm-matched all-position diagnostics
run on the same fixed stratified diagnostic subset, not the full test.
Random/ordinary-token controls therefore support only subset-scoped comparisons.

The default is a large research protocol, not the old six-hour run: 18 coefficient
fits x 8,000 examples, about 54,500 validation generations, and about 47,967 final
generations (using the published dataset sizes below). The measured preflight
may reject the 14-hour ceiling. A larger compute budget or a separately registered
smaller protocol is required in that case; this implementation makes no claim
that these counts fit within 14 hours on two RTX 3090s.

## Data

- Published HarmBench jailbreak strings with escaped UTF-16 surrogate pairs are
  decoded to the intended Unicode characters before tokenization. The manifest
  records the normalization; an unpaired surrogate fails preparation.
- Malformed training-source rows lacking a prompt or reference completion are
  excluded before sampling, with their original IDs and reasons recorded in
  `invalid_training_rows.json` and the data manifest. Configured split sizes stay
  unchanged. Malformed test rows still fail preparation; they are never dropped.
- WildJailbreak train: 2,000 each direct harmful, direct benign, adversarial
  harmful, adversarial benign (8,000 total).
- Validation: 400 each of those types (1,600), plus 300 GSM8K training questions.
- Test: full official WildJailbreak eval; full XSTest; all standard text behaviors
  in the **official HarmBench test split**, both direct and supplied published
  adversarial test cases; full GSM8K test split.
- Normalized exact prompt/base matches are unioned into request groups across
  corpora. Groups that intersect test are excluded from training/validation.
  Remaining groups are deterministically assigned to train or validation before
  quota sampling. XSTest focus groups remain intact. No outcome-based sampling.
- Missing base-request IDs in a source are disclosed as prompt-only grouping.
  Exact grouping is not a guarantee against semantic or template overlap.
- Training-pool prompts above the registered context limit are excluded with
  counts, before sampling. Test prompts are never silently truncated or dropped.
- XSTest/HarmBench/GSM8K were examined previously. Regenerating answers does not
  turn those public benchmarks into previously unseen holdouts.

WildJailbreak revision and filenames are verified against its public HF metadata.
Access to the actual gated files requires the account to accept the dataset's
terms. Do not automatically accept them or substitute another dataset.

The default adversarial suite uses the pinned HarmBench HumanJailbreaks
`random_subset_5` procedure, seed 1: shuffle the published literal prefix list
and prepend five prefixes to each standard test behavior, before consulting
any model outputs. At the pinned revision there are 159 standard test behaviors,
so this adds 795 cases. Downloaded Python is parsed as literal data, never run.
This is a fixed published transfer suite, not adaptive attack optimization.
It does not reproduce the entire HarmBench benchmark.

For a different published suite, set `harmbench_attack_mode: file`, supply a JSON
`{BehaviorID: [test_case_string, ...]}`, and record source attribution. Every
standard test behavior must be covered. Source/test-case hashes are recorded.

## Measures and claim limits

WildGuard harmful-response rate, benign refusal, GSM8K exact final-answer
accuracy, truncation, empty outputs, repetition, and perturbation diagnostics
are saved. HarmBench WildGuard rates are explicitly **not official behavior ASR**.
The pinned 13B HarmBench behavior classifier runs separately after test generation,
using the official template extracted as a literal from the pinned repository
(downloaded code is never executed). Unquantized FP16 with CPU offload may be
needed on a 24GB card; preflight includes its resource cost. Both per-prompt and
per-behavior-any-success results are saved separately from WildGuard scores.
The report also exports completion files compatible with the official
`evaluate_completions.py` for independent reproduction. Malformed labels or
overlength inputs fail rather than silently counting as safe.

The old 192-token evaluation cap is increased to 768. Target prompts may use up
to 8,192 tokens to retain long published test cases; the training-pool cap stays
2,048. WildGuard inputs are capped by the smaller of 16,384 and the loaded model's
declared context limit. Long-input judge quality still needs the manual audit;
a supported context window does not prove robust judging at that length.
The pilot must still show
<=10% truncation and >=5 harmful baseline responses on the harmful pilot pool;
otherwise stop and redesign before fitting/final testing. Passing those gates
does not guarantee adequate power. A validation paired-group power estimate is
saved; zero discordance is marked uninformative. No test-driven sample resizing.

Primary factorial comparisons average fitting seeds **within each prompt**,
then bootstrap whole request groups. Five contrasts per corpus/metric use
Bonferroni-adjusted bootstrap intervals. There is no claim of adjustment across
all corpora/metrics. Per-condition descriptive summaries have ordinary 95% CIs.
Fitting seeds are not extra independent benchmark prompts.

Same-prefix prefill traces support location/persistence diagnostics. Decode
traces after outputs diverge do not establish causal mediation. A fixed random
sample is exported for blinded human review; labels and condition keys are in
a separate file. Human review is not fabricated or marked complete automatically.
`SUCCESS` means computational completeness; `claim_readiness.json` separately
records limits and pending human review. Missing required official labels prevent
computational completion. The `official-judge` stage can resume that phase alone.

## Execution

Run CPU checks first:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

After dataset/model access is ready (the published attack suite is configured):

```bash
python experiments/run_jlens_safety_learned.py --run-dir results/safety_runs_v3/fresh_run --stage prepare
python experiments/run_jlens_safety_learned.py --run-dir results/safety_runs_v3/fresh_run --stage preflight --shard-count 2
python experiments/run_jlens_safety_learned.py --run-dir results/safety_runs_v3/fresh_run --stage validate
```

Preflight measures GPU backward, both inference scopes, WildGuard, disk/VRAM,
and conservative total runtime. The much larger protocol may exceed the 14-hour
ceiling; the program stops rather than shrinking data or claiming an unsupported
completion time. Validation/fitting remain sequential on the coordinator. Only
frozen final-test generation/judging is distributed.

Copy the complete coordinator run directory (including manifest, frozen data,
directions, coefficients and validation records, **before any test outputs**) to
each worker with the same code and packages. Then launch disjoint shards:

```bash
# On worker 0:
./scripts/start_experiment.sh experiments/run_jlens_safety_learned.py --run-dir results/safety_runs_v3/fresh_run --stage test-shard --shard-index 0 --shard-count 2
# On worker 1:
./scripts/start_experiment.sh experiments/run_jlens_safety_learned.py --run-dir results/safety_runs_v3/fresh_run --stage test-shard --shard-index 1 --shard-count 2
```

Collect worker copies, then use `scripts/merge_safety_shards.py --coordinator ...
--worker ...` and run the new runner with `--stage report`. Generation and judge
records must exactly match frozen expected jobs. Keep remote instances until
result copies and their checksums have been verified.
