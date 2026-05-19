#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-"python3"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
MODEL=${MODEL:-"/workspace/Models/${MODEL_NAME}"}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-"/workspace/datasets/wikitext/wikitext-2-raw-v1"}
TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
DELTA=${DELTA:-"0.05"}
TOP_K=${TOP_K:-"20"}
DATASET=${DATASET:-"wikitext2"}
N_SAMPLES=${N_SAMPLES:-"32"}
BATCH_SIZE=${BATCH_SIZE:-"50"}
SEED=${SEED:-"2025"}
CKPT_ROOT=${CKPT_ROOT:-"/workspace/ckpts"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
LAMBDA_EW=${LAMBDA_EW:-"1.0"}
TAU=${TAU:-"0.0"}
RERANK_MODE=${RERANK_MODE:-"curvature"}
CANDIDATE_DIR=${CANDIDATE_DIR:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/p0_candidates/candidates"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/p0_ew"}

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_SHARDS=${#GPU_ARRAY[@]}
if [ "${NUM_SHARDS}" -lt 1 ]; then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

export WIKITEXT2_PATH
mkdir -p "${OUTPUT_DIR}/shards"

pids=()
probe_csvs=()
for shard_id in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard_id]}"
  shard_dir="${OUTPUT_DIR}/shards/shard_${shard_id}"
  log_path="${shard_dir}/probe.log"
  mkdir -p "${shard_dir}"
  probe_csvs+=("${shard_dir}/probe_results.csv")
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" -u ew_probe_candidates.py \
      --candidate_dir "${CANDIDATE_DIR}" \
      --model "${MODEL}" \
      --model_name "${MODEL_NAME}" \
      --target_sparsity "${TARGET_SPARSITY}" \
      --delta "${DELTA}" \
      --top_k "${TOP_K}" \
      --dataset "${DATASET}" \
      --num_samples "${N_SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu_id "${gpu}" \
      --output_dir "${shard_dir}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_id "${shard_id}" \
      --seed "$((SEED + shard_id))"
  ) > "${log_path}" 2>&1 &
  pids+=("$!")
  echo "=> Probe shard ${shard_id}/${NUM_SHARDS}: GPU ${gpu}, log ${log_path}"
done

failed=0
for shard_id in "${!pids[@]}"; do
  if ! wait "${pids[$shard_id]}"; then
    echo "=> Probe shard ${shard_id} failed; see its probe.log" >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi

"${PYTHON_BIN}" scripts/merge_ew_probe_results.py \
  --output_dir "${OUTPUT_DIR}" \
  "${probe_csvs[@]}"

"${PYTHON_BIN}" ew_rerank.py \
  --probe_results "${OUTPUT_DIR}/probe_results.csv" \
  --mode "${RERANK_MODE}" \
  --lambda_ew "${LAMBDA_EW}" \
  --tau "${TAU}" \
  --output_dir "${OUTPUT_DIR}" \
  --candidates_jsonl "${CANDIDATE_DIR}/candidates.jsonl"

echo "EW output: ${OUTPUT_DIR}"
