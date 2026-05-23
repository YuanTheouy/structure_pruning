# PAS Model Replication Runbook

Date: 2026-05-23

Purpose: replicate the strict nested local-delta result on another model before
making any broader PAS claim. Start with OPT-1.3B because it is much cheaper
than another OPT-2.7B-size run and has seed2025 P0 artifacts from the earlier
pilot.

## Decision Target

For each replicated model, answer two separate questions:

- Local capability question: do strict nested small-radius slopes
  `S3025`, `S3050`, or `S31` explain downstream@30 after controlling endpoint
  `L30`?
- Stress robustness question: does `S35` remain the strong predictor of
  `L40` / `Regret40` after controlling endpoint `L30`?

Keep these claims separated. `S35` can support the stress-test story even when
the small local slopes do not support a SAM/local-flatness story.

## P0: Reuse Or Rebuild OPT-1.3B P0 Artifacts

First check whether the existing OPT-1.3B seed2025 artifacts are present:

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main

CAND=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates
PAS=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025

wc -l "$CAND/candidates.jsonl"
ls -lh "$PAS/probe_results.csv" "$PAS/heldout_results.csv"
```

If those files exist, do not rerun search/PAS. If they are missing, rebuild
the OPT-1.3B pilot:

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main

export MODEL=/workspace/Models/opt-1.3b
export MODEL_NAME=opt-1.3b
export WIKITEXT2_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
export CKPT_ROOT=/workspace/ckpts
export GPU_IDS="0 1 2 3 4 5 6 7"
export TARGET_SPARSITY=0.30
export DELTA=0.05
export TRAIN_EPISODES=400
export N_SAMPLES=16
export TOP_K=20
export SHORTLIST_SIZE=2
export RANDOM_REPEATS=500
export NUM_COLLECT=5
export LEARNING_EPOCH=3
export HELDOUT_N_SAMPLES=32
export SEED=2025
export RUN_ID_PREFIX=p0_candidates_opt13b_seed2025
export MERGED_CANDIDATE_DIR=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates
export CANDIDATE_DIR=$MERGED_CANDIDATE_DIR
export EW_OUTPUT_DIR=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_ew_seed2025
export PAS_OUTPUT_DIR=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025
export RUN_CANDIDATE_SEARCH=true
export RUN_PROBE=true
export RUN_PAS=true

bash scripts/run_pas_p0_server.sh
```

## P1: Export OPT-1.3B Stress Table

```bash
cd /workspace/structure_pruning

CAND=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates
PAS=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025
STRESS=/workspace/ckpts/pas_stress_recovery_opt13b
mkdir -p "$STRESS"

python scripts/pas_export_candidate_stress_table.py \
  --model /workspace/Models/opt-1.3b \
  --model-name opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --candidate-pool "$CAND" \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --probe-results "$PAS/probe_results.csv" \
  --heldout-results "$PAS/heldout_results.csv" \
  --output-dir "$STRESS"
```

Expected output:

```text
/workspace/ckpts/pas_stress_recovery_opt13b/candidate_stress_table_opt13b_seed2025.csv
```

## P2: Strict Nested Local PPL Probes

Run the two cheap strict nested probes. The first gives `S3025` and `S3050`;
the second gives an independently centered `S3050` and `S31`.

```bash
cd /workspace/structure_pruning

CAND=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates

bash scripts/pas_run_local_probe_multigpu.sh \
  --model /workspace/Models/opt-1.3b \
  --model-name opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --candidate-pool "$CAND" \
  --probe-sparsity 0.3025 \
  --delta 0.0025 \
  --base-sparsity 0.3000 \
  --projection-mode nested_from_base \
  --output-dir /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0025_nested \
  --gpu-ids "0 1 2 3 4 5 6 7" \
  --num-samples 64 \
  --batch-size 8 \
  --candidate-top-k 20

bash scripts/pas_run_local_probe_multigpu.sh \
  --model /workspace/Models/opt-1.3b \
  --model-name opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --candidate-pool "$CAND" \
  --probe-sparsity 0.3050 \
  --delta 0.0050 \
  --base-sparsity 0.3000 \
  --projection-mode nested_from_base \
  --output-dir /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0050_nested \
  --gpu-ids "0 1 2 3 4 5 6 7" \
  --num-samples 64 \
  --batch-size 8 \
  --candidate-top-k 20

wc -l /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0025_nested/probe_results.csv
wc -l /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0050_nested/probe_results.csv
grep -R "Traceback\|RuntimeError\|failed" \
  /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0025_nested/logs \
  /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0050_nested/logs || true
```

Expected line count is `21` for each CSV: one header plus 20 candidates.

Materialize the local-delta tables:

```bash
LOCAL=/workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_nested

python scripts/pas_materialize_local_delta_tables.py \
  --model opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --delta0025-probe-results /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0025_nested/probe_results.csv \
  --delta0050-probe-results /workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_delta0050_nested/probe_results.csv \
  --output-dir "$LOCAL"
```

Expected outputs:

```text
$LOCAL/local_delta_scores.csv
$LOCAL/local_delta_scores_s31.csv
$LOCAL/local_delta_scores_with_s31.csv
```

## P3: Downstream@30 For OPT-1.3B

Use the same six-task, limit-100 setting as the OPT-2.7B run. This is raw
`no_recovery` downstream@30; do not add recovery here.

```bash
cd /workspace/structure_pruning

unset HF_DATASETS_OFFLINE
unset HF_ENDPOINT
export HF_HOME=/workspace/datasets/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/datasets/.cache/huggingface/datasets

CAND=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates
STRESS=/workspace/ckpts/pas_stress_recovery_opt13b
DOWN=/workspace/ckpts/pas_stress_recovery_opt13b/downstream_seed2025_raw

RUN_DOWNSTREAM_NOW=true bash scripts/pas_run_downstream_batch.sh \
  --model /workspace/Models/opt-1.3b \
  --model-name opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --candidate-pool "$CAND" \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --candidate-table "$STRESS/candidate_stress_table_opt13b_seed2025.csv" \
  --output-dir "$DOWN" \
  --gpu-ids "0 1 2 3 4 5 6 7" \
  --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq \
  --limit 100 \
  --batch-size 4 \
  --eval-num-samples 64 \
  --recovery-method no_recovery
```

Collect the downstream summary:

```bash
python scripts/pas_collect_downstream_results.py \
  --model /workspace/Models/opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --candidate-pool "$CAND" \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --candidate-table "$STRESS/candidate_stress_table_opt13b_seed2025.csv" \
  --downstream-dir "$DOWN" \
  --output-dir "$STRESS"
```

Expected outputs:

```text
$STRESS/downstream_candidate_summary_opt13b_seed2025.csv
$STRESS/downstream_analysis_opt13b_seed2025.csv
$STRESS/downstream_retention_opt13b.csv
$STRESS/downstream_manifest_opt13b.json
```

## P4: Join And Read The Replication

```bash
cd /workspace/structure_pruning

STRESS=/workspace/ckpts/pas_stress_recovery_opt13b
LOCAL=/workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_nested
ANALYSIS=/workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_nested_analysis

python scripts/pas_analyze_local_delta_probe.py \
  --model opt-1.3b \
  --dataset wikitext2 \
  --seed 2025 \
  --local-delta-csv "$LOCAL/local_delta_scores_with_s31.csv" \
  --stress-table "$STRESS/candidate_stress_table_opt13b_seed2025.csv" \
  --downstream-summary "$STRESS/downstream_candidate_summary_opt13b_seed2025.csv" \
  --output-dir "$LOCAL"

python scripts/pas_analyze_local_selection_downstream.py \
  --local-delta-table "$LOCAL/local_delta_scores_with_s31.csv" \
  --downstream-summary "$STRESS/downstream_candidate_summary_opt13b_seed2025.csv" \
  --stress-table "$STRESS/candidate_stress_table_opt13b_seed2025.csv" \
  --output-dir "$ANALYSIS"

cat "$LOCAL/local_delta_analysis.md"
cat "$ANALYSIS/local_signal_correlation.md"
cat "$ANALYSIS/local_selection_downstream_table.md"
```

## Interpretation Gate

Positive local-flatness replication:

- `S3050 -> avg_pruned_score | L30` is negative on all candidates and is not
  obviously weaker than `S3025` or `S31`.
- The selected candidate from `PAS-S3050` is downstream-competitive with the
  oracle/downstream winner inside the relevant endpoint-close scope.

Positive stress replication:

- `S35 -> L40 | L30` and `S35 -> Regret40 | L30` stay strongly positive.
- `S35` remains clearly stronger than `S3025`, `S3050`, and `S31` for
  predicting `L40` / `Regret40`.

If OPT-1.3B is inconclusive because the endpoint winner is already oracle or
downstream variance is too small, keep it as a cheap sanity check and run the
same sequence on the next available larger/different-family model. Use the
same script stack and only change `MODEL`, `MODEL_NAME`, `SEED`, `CAND`, `PAS`,
`STRESS`, `LOCAL`, and `DOWN`.
