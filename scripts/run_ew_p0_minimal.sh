#!/bin/bash
set -euo pipefail

TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
DELTA=${DELTA:-"0.05"}
TOP_K=${TOP_K:-"20"}
DATASET=${DATASET:-"wikitext2"}
N_SAMPLES=${N_SAMPLES:-"32"}
BATCH_SIZE=${BATCH_SIZE:-"50"}
SEED=${SEED:-"2025"}
GPU_ID=${GPU_ID:-"0"}
NUM_SHARDS=${NUM_SHARDS:-"1"}
SHARD_ID=${SHARD_ID:-"0"}
LAMBDA_EW=${LAMBDA_EW:-"1.0"}
TAU=${TAU:-"0.0"}

ARGS=(
  --sigma "${TARGET_SPARSITY}"
  --delta "${DELTA}"
  --top_k "${TOP_K}"
  --dataset "${DATASET}"
  --num_samples "${N_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --gpu_id "${GPU_ID}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --lambda_ew "${LAMBDA_EW}"
  --tau "${TAU}"
)

if [ -n "${MODEL:-}" ]; then
  ARGS+=(--model "${MODEL}")
fi
if [ -n "${MODEL_NAME:-}" ]; then
  ARGS+=(--model_name "${MODEL_NAME}")
fi
if [ -n "${CANDIDATE_DIR:-}" ]; then
  ARGS+=(--candidate_dir "${CANDIDATE_DIR}")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${WIKITEXT2_PATH:-}" ]; then
  ARGS+=(--wikitext2_path "${WIKITEXT2_PATH}")
fi
if [ -n "${RUN_ID:-}" ]; then
  ARGS+=(--run_id "${RUN_ID}")
fi

python -u ew_p0_minimal.py "${ARGS[@]}" "$@"
