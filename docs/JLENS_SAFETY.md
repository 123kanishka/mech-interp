# J-lens safety steering: bounded Qwen3.5-4B study

Status: implementation and CPU checks only until the GPU preflight succeeds.
This is separate from the completed SAE/token comparison. Nothing in its results
establishes causal harmful-reasoning effects.

## Research contract

Question: can a candidate intervention drawn from selected J-lens token
directions reduce held-out harmful compliance at acceptable benign utility cost?
Compare early/middle/late layers, static versus conditional application,
ordinary residual difference steering, direct-unembedding directions, three
norm-matched random directions, and a norm-matched combined intervention.

Model: `Qwen/Qwen3.5-4B`, dense, BF16, pinned revision, text-only, **non-thinking**
chat template. The 4B checkpoint has a compatible J-lens and fits on a single
24 GB GPU with more headroom than the 9B and 27B variants. This does not measure
safety of hidden thinking-mode chains.

All layers are zero-based **block-output** residual hooks: 3, 15, 27 for
interventions, plus 30 for final available J-lens observation. No SAE
download/training is needed.

## Directions and gating

1. Extract only final-prompt-position activations on training prompts.
2. At each observed layer, select 32 non-special token labels by absolute
   harmful-minus-benign mean J-lens logit contrast on training data only.
3. Construct linear token covectors `W_U[token] @ J_layer`. Project the
   harmful-minus-benign residual mean difference onto their selected span using
   an SVD rank cutoff. Normalize the result.
4. This is a **signed selected linear span**, not the paper's sparse nonnegative
   decomposition and not the entire J-space. These covectors omit the derivative
   through final RMSNorm; they must not be described as exact local gradients.
5. Residual baseline: normalized full mean difference. Logit control: projection
   onto the same token labels' raw unembedding directions. This controls the
   coordinate transform, not the label-selection procedure independently.
6. A separate standardized-residual logistic probe is trained on prompt labels.
   Both its scores and the J-derived scalar are evaluated against prompt labels
   AND actual judged baseline response harmfulness on held-out data.
7. Steering is `h += alpha * median_training_residual_norm * unit_direction`.
   Test both signs; harmful-prompt directions can encode refusal as well as risk.
8. Conditional gate: threshold at the 90th percentile of training-benign
   direction scores. Evaluate it on validation; never calibrate with test data.
   Gate decision uses unmodified prefill state and is frozen throughout that
   response. It is **not** a validated dynamically updated reasoning-risk score.

Modes: last prompt position plus each cached decode position (`all`); only the
last prompt position (`prefill`); prompt-gated full trajectory (`gated`). Earlier
prompt positions are not changed. All calls use one unpadded sequence, greedy
generation, KV/recurrent caching, a 512-token input ceiling and 192-token output
cap. Overlong benchmark prompts fail rather than silently truncate.

## Six hypothesis checks

| Test | Implementation/evidence | What would weaken the claim |
|---|---|---|
| Signal location | Baseline J-score and independent probe AUROC/AP by layer | Sensitive-topic detection without prediction of unsafe responses |
| Causal selectivity | Matched held-out prompts across steering/control methods | Random/direct/residual controls perform equally well |
| Conditional application | Gated versus always-on safety and utility | Gate misses risk or benign refusals stay high |
| Persistence | Prefill-only versus repeated steering; downstream traces | Effect disappears downstream or needs repeated intervention |
| Readout-only suppression challenge | Joint inspection of score changes and judged compliance | Lower readout scores without safer responses |
| No early/J advantage challenge | Early/middle/late and non-J controls at matched norms | Late or conventional steering is as good/better |

Downstream prefill comparisons share identical inputs and are causal hook
diagnostics. Later generated-token traces may have different prefixes and are
**descriptive**, not matched-prefix causal mediation. Neither proves absence of
harmful internal computation. If baseline has no unsafe responses, behavioural
AUROC is undefined and safety improvement may have a floor effect.

## Benchmarks and selection

- XSTest: train 50 safe / 50 unsafe, validation 10 / 10, test 75 / 50.
  Integer allocation preserves whole focus groups while meeting exact counts
  and minimizing category imbalance. No focus crosses partitions.
- HarmBench: 100 held-out standard direct text behaviours; contextual and
  copyright cases excluded because they need different evaluation handling.
- GSM8K: 10 validation examples from original train, 75 held-out examples from
  original test. Correctness uses explicit `####` numeric answers. This is a
  narrow utility proxy; a short non-thinking cap can depress math performance.
- Sources, revisions, raw CSV hashes, selected prompt IDs and split hashes are
  frozen in the run directory. Small subsets are **not** full benchmark scores.

Total: 100 training / 30 validation / 300 test, 430 unique prompts. Training
fits directions/probes, not Qwen weights. HarmBench is final-test only.

Screen 3 methods × 3 layers × 4 signed strengths plus baseline on 32 predefined
validation prompts (4 safe XSTest, 4 unsafe, 4 GSM8K). Expand the winner for
each method/layer (9 plus baseline) to all 30 validation prompts, reusing the
screening outputs. Evaluate gated and prefill-only versions of each layer's
selected J strength on all 30. Compare only fully evaluated candidates. The
screening heatmap uses the same 12 prompts for every candidate.
Select minimum harmful compliance subject to at most +5 percentage points
benign over-refusal, at most 5 percentage points math-accuracy loss, and no
increase in empty outputs relative to baseline. Tie-break on over-refusal,
math accuracy, and smaller strength. These noisy small-validation constraints
are operational filters, not statistical safety guarantees.

If no candidate passes, keep the best explicitly **infeasible diagnostic**;
never call it a successful safety setting. Freeze all 12 final conditions before
any test generation. Four primary conditions (baseline, validation-selected J,
residual, direct-logit) run on all 300 test prompts. Eight additional conditions
run on a predefined 60-prompt diagnostic subset: 15 from each dataset/role.
Compare all 12 on that SAME
subset in `exploratory/summary.json`; main `summary.json` contains only the four
complete-test conditions. Diagnostics are exploratory, not equally powered
headline evidence. The combined condition is a norm-matched mixture of signed
J and ordinary residual directions selected at the same layer. It is not
AlphaSteer and does not establish improvement over published state of the art.
An exact AlphaSteer reproduction is a separate extension, not part of this
time-bounded suite or a silently claimed deliverable.

## Evaluator and metrics

WildGuard 7B BF16 runs sequentially after unloading Qwen. Uses its official prompt
template and separately parses response harmfulness and refusal. No paid API,
no keyword-refusal substitute, no automatic acceptance of gated model terms.
Access to `allenai/wildguard` must be granted to the Hugging Face account used
on the GPU. Credentials belong in the normal Hugging Face login, not the
repository. Preflight checks weight access before target loading.
Malformed/missing judge output stops selection/completion, never counts as safe.

Headline: harmful compliance, benign over-refusal, math accuracy. Also output
truncation, empty responses, repeated-trigram fraction (not a semantic fluency
measure), intervention frequency/magnitude, latency, and preflight peak VRAM.
Prompt-weighted 95% bootstrap intervals resample entire focus/behaviour groups;
paired changes use the same prompts and groups. Report individual corpora and
pooled values, acknowledging small counts and multiple comparisons. HarmBench
direct prompts scored with WildGuard are **not official HarmBench ASR**.

Raw generations, generated token IDs, judge text/labels, traces, selected token
labels, parameter-selection scorecards, and checksums are saved. Review the
stratified `manual_review_sample.json` before writing scientific conclusions;
automated SUCCESS does not mean a human has validated the judge.

## Compute contract: one RTX 3090, target 10–14 hours or less

Default: 100 training, 30 validation, 300 test prompts.
Validation: 37×12 + 10×18 + 6×30 = 804 generations.
Test: 4×300 + 8×60 = 1,680 generations.
Total: **2,484 generations**, capped at 192 new tokens each, plus judging,
training extraction and preflight. The cap is **476,928 target output tokens**,
not a guaranteed runtime. The enlarged suite may exceed 14 hours; the ceiling
remains unchanged pending throughput measurement and approval of any extension.
Runtime must be measured on the target GPU during preflight.

Preflight measures target generation, activation extraction, and WildGuard
sequentially. Projects full token-cap runtime using the slowest measured target
seconds/token and judge seconds/example, adds a 1.5× safety margin and 30 minutes.
It rejects estimates that exceed the remaining 14-hour active budget. Host load,
downloads, parsing failures and debugging can still invalidate estimates.
If rejected, obtain approval for a longer budget or a reduced protocol in a
**new config/run before fitting/testing**;
do not trim test outcomes after observing them. Aim for a forecast below 12 hours
to leave recovery time. Smaller samples make uncertainty wider.

The cumulative active-process timer checks between examples. An individual
generation/download can overrun the check boundary; this is not a provider
billing cap or a strict wall-clock kill. Idle rented GPU time is still billable.
Budget stops preserve outputs and never mark SUCCESS. Dataset/direction caches
and per-example resume avoid duplicate work. Target and judge never occupy GPU
memory simultaneously. Full-vocabulary readouts occur during fitting only.
Preflight warms persistent FLA/Triton kernels before measuring throughput and
records that one-time cost separately as `kernel_warmup_seconds`; multiplying
compilation time by every planned example would make the forecast meaningless.

## Running on Vast (after receiving the new connection details)

Use existing CUDA PyTorch; do not reinstall drivers. Install
`requirements-safety.txt` in the project environment; it omits the old SAE
dependency and pins Flash Linear Attention and Causal Conv1d so Qwen does not
silently use impractically slow reference operators. For RTX 3090 workers,
building the Causal Conv1d wheel only for CUDA compute capability 8.6 is safe and
substantially faster than compiling unused architectures. Preflight requires 40
GiB of **free** disk. Pinned Qwen, WildGuard
and J-lens weights total about 24.22 decimal GB. Plan 50–70 GB including the
environment, bounded outputs and temporary headroom. A fresh 100 GB container
is suitable with one shared cache and no duplicated model revisions.
Do not delete old caches/results automatically. Check free space before renting.

From `/workspace/mech-interp` on the GPU:

```bash
python -m pip install -r requirements-safety.txt
PYTHONPATH=src python -m unittest discover -s tests -p test_safety.py -v
python experiments/run_jlens_safety.py --run-dir results/safety_runs/safety_01 --stage prepare
python experiments/run_jlens_safety.py --run-dir results/safety_runs/safety_01 --stage preflight
```

Inspect `preflight.json` (budget, memory, hook checks) and
`preflight_examples.json` (chat formatting, generation and evaluator outputs).
Only proceed when these are acceptable:

```bash
./scripts/start_experiment.sh experiments/run_jlens_safety.py \
  --run-dir results/safety_runs/safety_01 --stage full
```

An explicit operator may bypass only the runtime-estimate gate with
`--stage validate --allow-unestimated-run`. This writes
`preflight_override.json`; it does not pretend that preflight passed, and it
does not bypass dataset, manifest, checksum, frozen-selection, shard-completeness,
or hard-budget checks.

The existing detached-session launcher allows laptop disconnection. Rerun the
same full command to resume. Never launch two workers in the same directory;
the OS file lock prevents concurrent appends. Code, configuration and package
versions must match the original manifest. A new experiment requires a new run
directory. Dataset preparation should happen in the same GPU environment;
do not reuse a CPU-prepared manifest with different PyTorch dependencies.

`--stage report` recomputes metrics/plots from saved test data without loading
the models. It can run on the CPU laptop after the GPU budget is exhausted;
the configuration and code must match, while differing analysis-environment
package versions are recorded separately in `report_environment.json`.
Before destroying an instance, sync this entire safety run folder
and logs locally and compare checksums. No script here destroys/stops instances.

## References

- J-lens: https://transformer-circuits.pub/2026/workspace/
- Official J-lens implementation: https://github.com/anthropics/jacobian-lens
- Qwen checkpoint: https://huggingface.co/Qwen/Qwen3.5-4B
- Refusal-direction baseline inspiration: https://github.com/andyrdt/refusal_direction
- HarmBench: https://github.com/centerforaisafety/HarmBench
- XSTest: https://github.com/paul-rottger/xstest
- WildGuard template/model: https://huggingface.co/allenai/wildguard
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
