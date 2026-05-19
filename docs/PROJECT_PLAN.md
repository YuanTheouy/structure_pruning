# FastForward Early-Warning Project Plan

## Target

- Scientific target: show that early-warning curvature is a useful selection signal for static structured LLM pruning.
- First evidence target: one reproducible P0 run on a small candidate pool.
- Submission target: unset until the curvature and high-sparsity tables are filled.

## Current Status

- Method status: clearly specified as neighboring-budget projection of the same FastForward priority vector, followed by endpoint/slope/curvature reranking.
- Code status: probe, rerank, correlation, plotting, ablation, and high-sparsity evaluation scripts exist.
- Paper status: abstract/method/experiment scaffold is coherent; several tables still contain `[fill]`.
- Biggest risk: the curvature signal may not correlate strongly enough with future degradation on the tested model/candidates.
- Novelty status: 2026-05-19 critic review says the safest framing is not a new pruning algorithm, but an early-warning guided policy-selection framework on top of a candidate generator.

## P0 Experiment

1. Choose one model/candidate source, preferably OPT-2.7B if candidate files/checkpoints are available locally.
2. Probe top-k candidates at sparsities `sigma-delta`, `sigma`, and `sigma+delta`, with default `sigma=0.30`, `delta=0.05`.
3. Run endpoint, slope, and curvature reranking.
4. Evaluate selected candidates at 30%, 35%, and 40% sparsity.
5. Generate path-divergence, curvature-scatter, high-sparsity-curve, and overhead summaries.

## Current Server P0 Runbook

This is an execution note only. Do not treat a successful run as a paper conclusion until the CSV/JSON artifacts are inspected and recorded in `docs/CLAIM_EVIDENCE.md`.

### Resource Layout

- Model root: `/workspace/Models`
- Dataset root: `/workspace/datasets`
- Checkpoint/artifact root: `/workspace/ckpts`
- Current smoke-test model: `/workspace/Models/opt-2.7b`
- Current model name: `opt-2.7b`
- Current dataset path: `/workspace/datasets/wikitext/wikitext-2-raw-v1`
- Current target sparsity: `0.30`
- Current early-warning delta: `0.05`

### Model Download Route

`facebook/opt-2.7b` was downloaded successfully through Hugging Face mirror with the server proxy. If the model must be re-downloaded, use:

```bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:3128
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:3128
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60

hf download facebook/opt-2.7b \
  --local-dir /workspace/Models/opt-2.7b \
  --max-workers 1 \
  --include "*.json" "*.txt" "*.model" "*.tiktoken" "*.safetensors" "*.bin" "merges.txt" "vocab.json"
```

Avoid `BLIP2-OPT-2.7B` for this P0 because it is a visual-multimodal model, not the pure causal LM path expected by the pruning code. ModelScope API/git repeatedly returned 504 for the tested OPT/Qwen repos on this server, so HF mirror plus proxy is the preferred model download path for now.

### Script Roles

- `scripts/run_p0_candidate_search.sh`: single-worker PPO pruning search. It calls `amc_searchPPO.py --job=train`, generates candidate policies, and writes `candidates.jsonl`, `scores/*.pt`, and `policies/*.json`.
- `scripts/run_p0_candidate_search_multigpu.sh`: launches multiple independent PPO workers on `GPU_IDS`, each with a different seed, then merges their candidate pools.
- `scripts/run_ew_p0_minimal.sh`: single-worker resource check plus early-warning probe/rerank over an existing candidate pool.
- `scripts/run_ew_p0_multigpu.sh`: shards early-warning probe over `GPU_IDS`, merges `probe_results.csv`, then runs `ew_rerank.py`.
- `scripts/merge_p0_candidate_pools.py`: merges per-worker candidate pools and keeps top-k endpoint candidates.
- `scripts/merge_ew_probe_results.py`: merges sharded probe CSV/JSONL files.

### Why Multi-GPU Is Needed

A single PPO candidate search is mostly sequential and will not naturally saturate all eight A100s. The faster P0 path is to run several independent PPO workers in parallel, one per GPU, then merge their top candidates. This is suitable for P0/smoke testing and candidate-pool generation. It is not identical to one long PPO trajectory, so final experiment reporting must record `GPU_IDS`, worker count, seeds, and per-worker episode budget.

### Smoke-Test Commands

Use these commands to verify the full chain quickly. They intentionally use a small episode/sample budget.

```bash
cd /workspace/structure_pruning
git fetch origin
git merge --ff-only origin/main

export MODEL=/workspace/Models/opt-2.7b
export MODEL_NAME=opt-2.7b
export WIKITEXT2_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
export CKPT_ROOT=/workspace/ckpts

export GPU_IDS="0 1 2 3 4 5 6 7"
export TARGET_SPARSITY=0.30
export DELTA=0.05
export TRAIN_EPISODES=80
export EPISODES_PER_WORKER=10
export N_SAMPLES=8
export TOP_K=8
export NUM_COLLECT=5
export LEARNING_EPOCH=3
export SEED=2025

nohup bash scripts/run_p0_candidate_search_multigpu.sh \
  > /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidate_multigpu.log 2>&1 &
```

Monitor:

```bash
tail -f /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidate_multigpu.log
tail -f /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_parallel_gpu0/search.log
watch -n 5 nvidia-smi
watch -n 10 'for d in /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_parallel_gpu*/candidates; do echo -n "$d "; [ -f "$d/candidates.jsonl" ] && wc -l "$d/candidates.jsonl" || echo 0; done'
```

After the candidate search log reports a merged pool, run early-warning rerank:

```bash
nohup bash scripts/run_ew_p0_multigpu.sh \
  > /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew_multigpu.log 2>&1 &

tail -f /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew_multigpu.log
```

Expected candidate pool:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates
```

Expected early-warning output:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew
```

### Larger P0 Settings

After the smoke test completes, increase budget gradually:

```bash
export TRAIN_EPISODES=5000
unset EPISODES_PER_WORKER
export N_SAMPLES=32
export TOP_K=20
export NUM_COLLECT=15
export LEARNING_EPOCH=10

bash scripts/run_p0_candidate_search_multigpu.sh
bash scripts/run_ew_p0_multigpu.sh
```

If GPU0 is already occupied, exclude it:

```bash
export GPU_IDS="1 2 3 4 5 6 7"
```

### Stop And Cleanup

Do not press `Ctrl-C` on the foreground multi-GPU launcher unless the run should stop. If stopped accidentally, logs may show `KeyboardInterrupt` inside `import torch`, which means the worker was interrupted rather than the code failing.

Check and stop residual workers:

```bash
ps -ef | grep -E "amc_searchPPO.py|ew_probe_candidates.py" | grep -v grep
pkill -f amc_searchPPO.py
pkill -f ew_probe_candidates.py
```

### Artifact Checklist

Record these after each run:

- Candidate pool: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates/candidates.jsonl`
- All candidate log: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates/all_candidates.jsonl`
- Probe results: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/probe_results.csv`
- Probe JSONL: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/probe_results.jsonl`
- Rerank results: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/rerank_results.csv`
- Selected candidate JSON: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/selected_candidates.json`
- Best candidate JSON: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/best_candidate.json`

### Observed P0 Smoke-Test Artifacts

Run observed on 2026-05-19 with:

- Model: `opt-2.7b`
- Dataset: `wikitext2`
- Target sparsity: `0.30`
- Delta: `0.05`
- GPU IDs: `0 1 2 3 4 5 6 7`
- PPO budget: `TRAIN_EPISODES=80`, `EPISODES_PER_WORKER=10`
- Candidate top-k: `8`
- Probe samples: `N_SAMPLES=8`
- PPO inner loop: `NUM_COLLECT=5`, `LEARNING_EPOCH=3`
- Seeds: `2025` through `2032`

Observed candidate-generation summary:

```text
Merged 58 candidates from 8 pools
Wrote 8 selected candidates to /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates/candidates.jsonl
```

Observed file counts:

```text
8 /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates/candidates.jsonl
9 /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/probe_results.csv
```

Observed early-warning output summary:

```text
Merged 8 probe rows into /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/probe_results.csv
Wrote /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/rerank_results.csv
Wrote /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/best_candidate.json
Wrote /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/selected_candidates.json
```

Observed top candidate from `best_candidate.json`:

```text
candidate_id: p0_candidates_parallel_gpu2_opt-2.7b_seed2027_step000000_ep000000
selected_mode: curvature
selection_score: 5.938169465710291
endpoint_logppl from candidate record: 5.713615864981652
endpoint_ppl from candidate record: 302.9645690917969
probe logppl_minus: 5.181026925382635
probe logppl_zero: 5.938169465710291
probe logppl_plus: 6.468749987255017
probe ppl_minus: 177.86537170410156
probe ppl_zero: 379.2400817871094
probe ppl_plus: 644.6773681640625
slope: 10.611610430894522
curvature: -90.62480751317209
future_degradation: 0.5305805215447261
actual_sparsity_minus: 0.2501126329304517
actual_sparsity_zero: 0.3005624973626173
actual_sparsity_plus: 0.350333810748619
probe eval seconds total: 144.03351759910583
```

Top-8 rerank order reported by the smoke test:

```text
rank=1 id=p0_candidates_parallel_gpu2_opt-2.7b_seed2027_step000000_ep000000 score=5.938169
rank=2 id=p0_candidates_parallel_gpu0_opt-2.7b_seed2025_step000009_ep000009 score=6.422321
rank=3 id=p0_candidates_parallel_gpu0_opt-2.7b_seed2025_step000001_ep000001 score=6.508482
rank=4 id=p0_candidates_parallel_gpu2_opt-2.7b_seed2027_step000009_ep000009 score=6.856250
rank=5 id=p0_candidates_parallel_gpu5_opt-2.7b_seed2030_step000001_ep000001 score=64.010385
rank=6 id=p0_candidates_parallel_gpu0_opt-2.7b_seed2025_step000004_ep000004 score=137.143111
rank=7 id=p0_candidates_parallel_gpu7_opt-2.7b_seed2032_step000009_ep000009 score=341.656157
rank=8 id=p0_candidates_parallel_gpu4_opt-2.7b_seed2029_step000004_ep000004 score=434.800688
```

Interpretation guardrail: this is only a smoke-test artifact proving that the pipeline can produce candidates, probe rows, and rerank outputs. It is too small for a manuscript claim because it uses only 8 probe samples and 10 PPO episodes per worker.

## Minimum Manuscript Tables To Fill

| Table/Figure | Required File Artifact |
| --- | --- |
| Endpoint-similar path divergence figure | `path_divergence.pdf` plus source CSV |
| Curvature correlation table | correlation JSON/CSV from `analyze_curvature_correlation.py` |
| Main high-sparsity table | `high_sparsity_results.csv` |
| Early-warning ablation table | output from `make_ablation_table.py` or `compare_endpoint_slope_curvature.py` |
| Overhead table | output from `summarize_overhead.py` |

## Review Questions

- Is curvature better than slope, or is slope sufficient?
- Does curvature help only at 40% sparsity, or also at 30/35%?
- Are candidates truly endpoint-similar before future divergence is claimed?
- Does calibration change the ranking, or only improve endpoint PPL?
- Are all PPL values computed on the same calibration/evaluation split and sequence length?

## Positioning Guardrails

- Lead with compression-resilience diagnosis and candidate selection.
- Treat FastForward as the candidate generator, not as the paper's full novelty.
- Treat Ridge calibration as optional post-pruning correction, not a contribution.
- Do not claim discovery of LLM compression phase transitions; cite phase-transition work as motivation.
- Do not claim universal collapse prediction; report the tested candidate pool, models, budgets, and correlation.
- Explicitly separate probe budgets from future-evaluation budgets to avoid leakage objections.
