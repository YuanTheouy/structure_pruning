#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-"python3"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
MODEL=${MODEL:-"/workspace/Models/${MODEL_NAME}"}
TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
TOP_K=${TOP_K:-"20"}
CKPT_ROOT=${CKPT_ROOT:-"/workspace/ckpts"}
SEED=${SEED:-"2025"}
TRAIN_EPISODES=${TRAIN_EPISODES:-"5000"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
RUN_ID_PREFIX=${RUN_ID_PREFIX:-"p0_candidates_parallel"}
MERGED_CANDIDATE_DIR=${MERGED_CANDIDATE_DIR:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/p0_candidates/candidates"}

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_WORKERS=${#GPU_ARRAY[@]}
if [ "${NUM_WORKERS}" -lt 1 ]; then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

EPISODES_PER_WORKER=${EPISODES_PER_WORKER:-"$(( (TRAIN_EPISODES + NUM_WORKERS - 1) / NUM_WORKERS ))"}
TOP_K_PER_WORKER=${TOP_K_PER_WORKER:-"${TOP_K}"}

candidate_dirs=()
pids=()

echo "=> Launching ${NUM_WORKERS} PPO candidate workers"
echo "=> Total episode budget: ${TRAIN_EPISODES}; per worker: ${EPISODES_PER_WORKER}"

for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  run_id="${RUN_ID_PREFIX}_gpu${gpu}"
  run_root="${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/${run_id}"
  candidate_dir="${run_root}/candidates"
  log_path="${run_root}/search.log"
  mkdir -p "${run_root}" "${candidate_dir}"
  candidate_dirs+=("${candidate_dir}")

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MODEL="${MODEL}"
    export MODEL_NAME="${MODEL_NAME}"
    export TARGET_SPARSITY="${TARGET_SPARSITY}"
    export TOP_K="${TOP_K_PER_WORKER}"
    export SEED="$((SEED + idx))"
    export TRAIN_EPISODES="${EPISODES_PER_WORKER}"
    export RUN_ID="${run_id}"
    export RUN_ROOT="${run_root}"
    export CANDIDATE_DIR="${candidate_dir}"
    bash scripts/run_p0_candidate_search.sh
  ) > "${log_path}" 2>&1 &
  pids+=("$!")
  echo "=> Worker ${idx}: GPU ${gpu}, seed $((SEED + idx)), log ${log_path}"
done

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "=> Worker ${idx} failed; see its search.log" >&2
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  exit 1
fi

"${PYTHON_BIN}" scripts/merge_p0_candidate_pools.py \
  --output_dir "${MERGED_CANDIDATE_DIR}" \
  --top_k "${TOP_K}" \
  "${candidate_dirs[@]}"

echo "Merged candidate pool: ${MERGED_CANDIDATE_DIR}"
