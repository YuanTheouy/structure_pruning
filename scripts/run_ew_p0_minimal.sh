#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-"python3"}
MODEL_NAME=${MODEL_NAME:-"TinyLlama-1.1B-Chat-v1.0"}
MODEL=${MODEL:-"/workspace/Models/${MODEL_NAME}"}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-"/workspace/datasets/wikitext/wikitext-2-raw-v1"}
WIKITEXT2_CONFIG=${WIKITEXT2_CONFIG:-"wikitext-2-raw-v1"}
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
CKPT_ROOT=${CKPT_ROOT:-"/workspace/ckpts"}
CANDIDATE_DIR=${CANDIDATE_DIR:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/p0_candidates/candidates"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/p0_ew"}

ARGS=(
  --model "${MODEL}"
  --model_name "${MODEL_NAME}"
  --sigma "${TARGET_SPARSITY}"
  --delta "${DELTA}"
  --top_k "${TOP_K}"
  --dataset "${DATASET}"
  --wikitext2_path "${WIKITEXT2_PATH}"
  --num_samples "${N_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --gpu_id "${GPU_ID}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --lambda_ew "${LAMBDA_EW}"
  --tau "${TAU}"
  --ckpt_root "${CKPT_ROOT}"
  --candidate_dir "${CANDIDATE_DIR}"
  --output_dir "${OUTPUT_DIR}"
)

if [ -n "${RUN_ID:-}" ]; then
  ARGS+=(--run_id "${RUN_ID}")
fi

"${PYTHON_BIN}" -u ew_p0_minimal.py "${ARGS[@]}" "$@"
