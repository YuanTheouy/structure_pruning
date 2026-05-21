# FastForward Experiment Review

Date: 2026-05-19

## 2026-05-21 Stress-Recovery Update

The active experiment priority has changed. The paper should not claim that PAS
is a universally better single-budget `30%` pruning method. The current
evidence gate is:

```text
Stability matters only if local stress S35 = L35 - L30 predicts
held-out cross-budget regret and same-protocol recovery quality.
```

Current seed `3025` handoff is recorded in:

- `docs/PAS_SEED3025_RESULTS_SUMMARY_2026_05_21.md`
- `docs/CLAIM_EVIDENCE.md`
- `docs/PROJECT_PLAN.md`

Server-reset seed `3025` artifacts already show that the selected `PAS-Slope`
priority vector trades a small probe-side endpoint cost for better `0.40`
stress behavior:

| Rule | Candidate | `PoBR_sigma` | `StressGain_h` | `Regret_h` |
| --- | --- | --- | --- | --- |
| `FF-Endpoint` | `p0_candidates_seed3025_gpu6_opt-2.7b_seed3031_step000037_ep000037` | `0.0` | `0.0` | `0.6400670596401818` |
| `PAS-Slope` | `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048` | `0.04040185393219975` | `0.6400670596401818` | `0.0` |

This is not enough by itself for a recovery claim. The required next steps are:

1. P0: export every candidate's `L30`, `L35`, `L40`, `S35`, `Delta40`, and
   `Regret40`; then test whether `S35` predicts stress regret after controlling
   for `L30`.
2. P1: run one identical recovery protocol on a fixed candidate subset and test
   whether `S35` predicts recovered PPL or `RecoveryGain`.
3. P2 downstream retention only if P0/P1 are positive.

Do not start periodic PAS / RPVS until P0/P1 pass. If P0 or P1 fails, the paper
should keep the claim at "budget-transferable priority-vector diagnostics" and
must not claim PAS improves recovery.

P0 stress table was completed on 2026-05-21 for `opt-2.7b`, seed `3025`, 20
candidates:

| Metric | Value |
| --- | --- |
| `Pearson(S35,Regret40)` | `0.1625803636488299` |
| `Spearman(S35,Regret40)` | `0.09473684210526315` |
| `Pearson(S35,Delta40)` | `0.8337100870651009` |
| `Spearman(S35,Delta40)` | `0.8285714285714285` |
| `partial_corr(S35,Regret40|L30)` | `0.8107895146421564` |
| `partial_corr(S35,L40|L30)` | `0.8107895146421564` |
| `L40~L30+S35` | `beta_S35=1.2588788204496062`, `R2=0.8817489807506785` |
| `Regret40~L30+S35` | `beta_S35=1.2588788204496064`, `R2=0.8817489807506785` |

Review interpretation: P0 passes the intended controlled-stability test. Raw
`S35 -> Regret40` is weak, but `S35 -> Delta40` and controlled
`S35 -> Regret40 | L30` are strong. Proceed to P1 same-protocol recovery before
making any recovery claim.

## Reviewed Artifacts

- Runbook and smoke-test summary: `docs/PROJECT_PLAN.md`
- Claim-evidence status: `docs/CLAIM_EVIDENCE.md`
- Minimal runner: `ew_p0_minimal.py`
- Multi-GPU probe runner: `scripts/run_ew_p0_multigpu.sh`
- High-sparsity evaluator: `evaluate_high_sparsity_curve.py`
- Correlation analyzer: `analyze_curvature_correlation.py`

The raw server artifacts are recorded under `/workspace/ckpts/opt-2.7b/sparsity_0.30/...`, but they are not present on this local filesystem. The review below therefore treats the documented smoke-test summary as a run report, not as independently recomputed evidence.

## What The Smoke Test Establishes

The smoke test is useful. It establishes that the engineering path is real:

1. The server can load a pure causal-LM OPT-2.7B checkpoint.
2. Multi-GPU independent candidate search can generate and merge a candidate pool.
3. The early-warning probe can evaluate neighboring sparsity projections.
4. The reranker can emit `rerank_results.csv`, `best_candidate.json`, and `selected_candidates.json`.

This means the project has moved from "paper idea plus scripts" to "pipeline smoke-tested on a real LLM."

## What It Does Not Establish Yet

Do not use the smoke test as manuscript evidence for scientific claims. The current run used:

- `N_SAMPLES=8`
- `TOP_K=8`
- only `10` PPO episodes per worker
- a small smoke candidate pool

This is enough to debug the pipeline, but not enough to support claims about correlation, high-sparsity robustness, or superiority over endpoint-only selection.

## Important Methodology Issue

The current probe field `future_degradation` is computed as:

```text
logppl_plus - logppl_zero
```

where `logppl_plus` is already one of the three points used to compute curvature at `sigma + delta`. This is acceptable as a local slope/probe diagnostic, but it is not a separated future-degradation target for the paper's predictive claim.

For the paper-level curvature-correlation claim, use separated budgets:

- warning/probe path: `0.25 / 0.30 / 0.35`
- future target: `0.40`

The correlation table should use degradation from `0.30` to `0.40`, not `0.30` to `0.35`, otherwise reviewers can object that the future label overlaps the probe.

## Smoke-Test Signal Read

The documented best smoke candidate has:

- `probe logppl_zero = 5.9382`
- `probe logppl_plus = 6.4687`
- `slope = 10.6116`
- `curvature = -90.6248`
- `future_degradation = 0.5306` as currently defined

Because the curvature is negative, the curvature penalty is zero for this candidate. In this smoke run, curvature selection may collapse to endpoint selection for good candidates. This is not a failure, but it means the next run must explicitly compare:

- endpoint best
- slope best
- curvature best

and then evaluate those selected candidates at a separated future sparsity, preferably `0.40`.

## Next Required Run

Run a larger P0 with separated future evaluation:

```bash
cd /workspace/structure_pruning

export MODEL=/workspace/Models/opt-2.7b
export MODEL_NAME=opt-2.7b
export WIKITEXT2_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
export CKPT_ROOT=/workspace/ckpts

export GPU_IDS="0 1 2 3 4 5 6 7"
export TARGET_SPARSITY=0.30
export DELTA=0.05
export TRAIN_EPISODES=5000
unset EPISODES_PER_WORKER
export N_SAMPLES=32
export TOP_K=20
export NUM_COLLECT=15
export LEARNING_EPOCH=10
export SEED=2025

bash scripts/run_p0_candidate_search_multigpu.sh
bash scripts/run_ew_p0_multigpu.sh
python evaluate_high_sparsity_curve.py \
  --selected_candidates_json /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/selected_candidates.json \
  --model /workspace/Models/opt-2.7b \
  --model_name opt-2.7b \
  --sparsities 0.30 0.35 0.40 \
  --num_samples 64 \
  --output_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_high_sparsity
```

After that run, copy or commit the following raw artifacts so the coordinator can inspect them directly:

- `p0_candidates/candidates/candidates.jsonl`
- `p0_ew/probe_results.csv`
- `p0_ew/rerank_results.csv`
- `p0_ew/selected_candidates.json`
- `p0_high_sparsity/high_sparsity_results.csv`
- shard logs and candidate-search logs

## Decision Gate

Historical note: this decision gate was written before the seed3025 results made
PAS-Slope the primary score and moved curvature to ablation. The current active
method is defined in `docs/ACTIVE_CORE_METHOD.md`.

Proceed to manuscript table filling only if the larger P0 shows at least one of:

- PAS-Slope/local budget sensitivity and future `0.40` degradation have positive Pearson/Spearman correlation;
- PAS-Slope-selected candidate has better `0.40` PPL than endpoint-selected candidate at similar endpoint quality;
- path-divergence figure shows endpoint-similar candidates separating at `0.40`.

If none hold, pivot the paper toward "compression-path diagnostics" rather than
"PAS-Slope improves selection."
