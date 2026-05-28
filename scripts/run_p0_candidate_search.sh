#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-"python3"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
MODEL=${MODEL:-"/workspace/Models/${MODEL_NAME}"}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-"/workspace/datasets/wikitext/wikitext-2-raw-v1"}
WIKITEXT2_CONFIG=${WIKITEXT2_CONFIG:-"wikitext-2-raw-v1"}
CKPT_ROOT=${CKPT_ROOT:-"/workspace/ckpts"}

TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
TOP_K=${TOP_K:-"20"}
DATASET=${DATASET:-"wikitext2"}
N_SAMPLES=${N_SAMPLES:-"32"}
BATCH_SIZE=${BATCH_SIZE:-"50"}
SEED=${SEED:-"2025"}
GPU_ID=${GPU_ID:-"0"}
TRAIN_EPISODES=${TRAIN_EPISODES:-"5000"}
NUM_COLLECT=${NUM_COLLECT:-"15"}
LEARNING_EPOCH=${LEARNING_EPOCH:-"10"}
RUN_ID=${RUN_ID:-"p0_candidates"}
USE_GRADUAL_PRUNING=${USE_GRADUAL_PRUNING:-"false"}
GRADUAL_INITIAL_SPARSITY=${GRADUAL_INITIAL_SPARSITY:-"0.05"}
GRADUAL_PRUNING_END_EPISODE=${GRADUAL_PRUNING_END_EPISODE:-"1000"}
CANDIDATE_SAVE_MODE=${CANDIDATE_SAVE_MODE:-"topk_and_periodic"}
SAVE_EVERY=${SAVE_EVERY:-"25"}
USE_DATASET_GROWTH=${USE_DATASET_GROWTH:-"false"}
DATASET_INITIAL_RATIO=${DATASET_INITIAL_RATIO:-"0.05"}
DATASET_FINAL_RATIO=${DATASET_FINAL_RATIO:-"${DATASET_INITIAL_RATIO}"}
DATASET_GROWTH_START_EPISODE=${DATASET_GROWTH_START_EPISODE:-"0"}
DATASET_GROWTH_END_EPISODE=${DATASET_GROWTH_END_EPISODE:-"${TRAIN_EPISODES}"}

RUN_ROOT=${RUN_ROOT:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/${RUN_ID}"}
CANDIDATE_DIR=${CANDIDATE_DIR:-"${RUN_ROOT}/candidates"}

mkdir -p "${RUN_ROOT}" "${CANDIDATE_DIR}"

export WIKITEXT2_PATH
export WIKITEXT2_CONFIG

GRADUAL_FLAGS=()
if [[ "${USE_GRADUAL_PRUNING}" == "true" ]]; then
  GRADUAL_FLAGS+=(
    --use_gradual_pruning
    --gradual_final_sparsity="${TARGET_SPARSITY}"
    --gradual_initial_sparsity="${GRADUAL_INITIAL_SPARSITY}"
    --gradual_pruning_end_episode="${GRADUAL_PRUNING_END_EPISODE}"
  )
fi

SAVE_FLAGS=(
  --candidate_save_mode="${CANDIDATE_SAVE_MODE}"
)
if [[ -n "${SAVE_EVERY}" ]]; then
  SAVE_FLAGS+=(--save_every="${SAVE_EVERY}")
fi

DATASET_FLAGS=()
if [[ "${USE_DATASET_GROWTH}" == "true" ]]; then
  DATASET_FLAGS+=(
    --use_dataset_growth
    --dataset_initial_ratio="${DATASET_INITIAL_RATIO}"
    --dataset_final_ratio="${DATASET_FINAL_RATIO}"
    --dataset_growth_start_episode="${DATASET_GROWTH_START_EPISODE}"
    --dataset_growth_end_episode="${DATASET_GROWTH_END_EPISODE}"
  )
else
  DATASET_FLAGS+=(
    --dataset_initial_ratio="${DATASET_INITIAL_RATIO}"
    --dataset_final_ratio="${DATASET_FINAL_RATIO}"
    --dataset_growth_start_episode="${DATASET_GROWTH_START_EPISODE}"
    --dataset_growth_end_episode="${DATASET_GROWTH_END_EPISODE}"
  )
fi

"${PYTHON_BIN}" -u amc_searchPPO.py \
  --job=train \
  --model="${MODEL}" \
  --model_name="${MODEL_NAME}" \
  --dataset_name="${DATASET}" \
  --preserve_ratio="$(awk "BEGIN {printf \"%.6f\", 1 - ${TARGET_SPARSITY}}")" \
  --structure \
  --prune=para \
  --lbound=0.1 \
  --rbound=1.0 \
  --n_samples="${N_SAMPLES}" \
  --data_bsize="${BATCH_SIZE}" \
  --num_collect="${NUM_COLLECT}" \
  --learning_epoch="${LEARNING_EPOCH}" \
  --reward=reward_ppl \
  --train_episode="${TRAIN_EPISODES}" \
  --seed="${SEED}" \
  --gpu_id="${GPU_ID}" \
  --output="${RUN_ROOT}/logs" \
  --export_path="${RUN_ROOT}/endpoint_best.pth.tar" \
  --enable_downstream=false \
  --state_mode=0 \
  --save_candidates \
  "${SAVE_FLAGS[@]}" \
  --candidate_top_k="${TOP_K}" \
  --candidate_dir="${CANDIDATE_DIR}" \
  --run_id="${RUN_ID}" \
  "${DATASET_FLAGS[@]}" \
  "${GRADUAL_FLAGS[@]}"

echo "Candidate pool: ${CANDIDATE_DIR}"
