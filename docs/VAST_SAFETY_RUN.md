# Safety deployment and resource plan — 2026-09-11

Status: larger-data configuration and deterministic two-GPU test sharding are
implemented. Instances 50598128 and 50598882 (same host 155385, RTX 3090 each)
were supplied, but both direct ports closed during SSH key exchange and both
proxy ports refused connections when checked. No full scientific run has
started. Re-read live connection details before deployment; do not use the old
instance's SSH endpoint.

## Verified access and artifacts

An authenticated metadata request for the pinned WildGuard weight file now
succeeds with the laptop's existing Hugging Face login (previously HTTP 403).
This checks permission without downloading the full model. GPU authentication,
actual loading, VRAM and evaluator output must still pass preflight. No token
was printed, put in the repo, or transferred to a new host.

Pinned artifact sizes retrieved from Hugging Face:

| Artifact | Decimal GB |
|---|---:|
| Qwen3.5-4B: two safetensors shards | 9.32 |
| WildGuard 7B: two safetensors shards | 14.50 |
| Chosen Qwen J-lens file | 0.406 |
| Total | 24.22 |

Allow 10–20 GB for environment/build caches, 1–3 GB for bounded outputs and
datasets, and 15–20 GB temporary/free headroom: approximately 50–70 GB planned
footprint. A fresh 100 GB container is sufficient under this policy. Use one HF
cache; don't duplicate snapshots or retain unrelated old models. Preflight has
a 40 GiB free-space guard. Never delete user caches/results automatically.
No raw full-vocabulary distributions or all-token activations are retained.
Unlimited retries/cache duplication have no finite worst-case storage bound.

## Screenshot comparison (not a live availability guarantee)

All five fully visible offers are one RTX 3090 with 24 GB VRAM and similar
advertised GPU bandwidth/compute. No per-host measured inference timings exist.

| Location / machine | Screenshot $/hour | Allocated RAM | Assessment |
|---|---:|---:|---|
| Yunnan / 78080 | 0.175 | 32 GB | Viable; less loading headroom than Fujian |
| Fujian / 78398 | 0.175 | 48 GB | Preferred: tied lowest price, most RAM |
| CN / 144003 | 0.202 | 32 GB | Viable, slower advertised download link |
| Poland / 31375 | 0.202 | 16 GB | Avoid: higher model-loading/swap/OOM risk |
| Taiwan / 35807 | 0.215 | 28 GB | Possible; less RAM headroom |

The sixth row is clipped: do not infer missing specifications. The filter bar
does not mean each offer supplies two GPUs; the offer cards explicitly say 1×.

## Runtime scenarios, NOT benchmarks or guarantees

2,484 target responses at up to 192 tokens = 476,928 target output tokens.
The slower coordinator performs 804 validation and 840 test jobs. Preflight
measures real target/judge times, applies a 1.5× margin and adds 30 minutes.

| Assumed r | Assumed j | Scenario duration |
|---:|---:|---:|
| 40 tokens/s | 2 s | about 5.2 h with margin |
| 20 tokens/s | 4 s | about 9.9 h with margin |
| 10 tokens/s | 6 s | about 19.4 h with margin |

These apply equally to the GPU-compute portion of all five offers, NOT as
different measured per-host predictions. Low RAM, reference-kernel fallbacks,
host contention, prompt costs, download stalls, retries or outages can exceed
even the slow scenario. There is no finite guaranteed worst-case finish time.
Short outputs and cached duplicate judgements can reduce actual work.
Prices also incur storage/bandwidth charges; consult Vast's billing breakdown.

The active-process hard ceiling is 14 hours and the distributed target is six
hours. Preflight accepts --shard-count 2 and estimates the coordinator's
sequential validation plus one half-test shard. If that conservative projection
exceeds six hours, it blocks fitting: revise counts in a new run rather than
starting a run that cannot meet the requested target.

## New-host execution plan

1. Receive the rented instance's new SSH endpoint. Verify its host identity and
   GPU, allocated RAM, actual free disk and CUDA environment.
2. Ship the complete code/config once into /workspace/mech-interp. Use a NEW run
   directory and GPU-created manifest; older pilot/CPU manifests are not
   compatible with this changed code/config/environment.
3. Install requirements using existing CUDA PyTorch; authenticate model access
   without exposing secrets. Check compatible optimized Qwen kernels rather
   than assuming reference fallbacks meet the deadline.
4. Run tests, prepare pinned splits, and run GPU preflight. Inspect actual model
   and judge outputs, hook equivalence, memory use and conservative forecast.
5. After acceptance of preflight and the run budget, run --stage validate on
   GPU 1. Copy the frozen, checksum-identical run to GPU 2. Launch --stage
   test-shard --shard-index 0 --shard-count 2 on GPU 1 and index 1 on GPU 2 in
   detached tmux sessions. Each worker generates and then judges its own exact
   840-job shard. This avoids GPU model contention and produces deterministic,
   disjoint coverage. Merge with scripts/merge_safety_shards.py, run --stage
   report on the coordinator and verify all 1,680 test jobs before SUCCESS.
   The coordinator sequence before sharding is training extraction → screening
   generation/judging → expanded validation → gated/prefill validation → freeze.
   Qwen and WildGuard load sequentially, never together on the GPU. No manual
   approval per condition is needed. Never launch parallel workers in one run.
6. Completed examples are saved incrementally and skipped on resume. Failures,
   malformed judge labels and budget exhaustion stop the pipeline; partial work
   is preserved, not labeled successful. No automatic credit top-up or instance
   destruction is authorized.
7. Sync the entire run directory and logs with scripts/sync_safety_run.py using
   NEW connection details. Verify final SHA-256 manifest after SUCCESS.
   Laptop-independent GPU execution is not laptop-independent backup: local
   copies require the laptop online. A separate remote backup destination needs
   to be configured if backup while the laptop is off is required.

Do not destroy the instance until an external copy is checksum-verified.
Container storage alone is not a durable off-instance backup.

## Preserved prior preparation

Old run (data/tests only, not scientific results):
results/safety_runs/safety_20260911_vast50572343 and corresponding local
results/logs/safety_20260911_vast50572343. The older pilot contained 64 training,
32 validation and 96 test examples; do not reuse it for the enlarged study.

New local dataset-only verification is under
results/safety_runs/larger_data_validation_v2; it is not a GPU preflight or
a completed experiment and its CPU manifest must not be used to resume on Vast.

References: https://huggingface.co/allenai/wildguard ;
https://docs.vast.ai/guides/reference/billing
