#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-"python3"}
MODEL=${MODEL:-"/workspace/Models/opt-2.7b"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-"/workspace/datasets/wikitext/wikitext-2-raw-v1"}
WIKITEXT2_CONFIG=${WIKITEXT2_CONFIG:-"wikitext-2-raw-v1"}
CKPT_ROOT=${CKPT_ROOT:-"/workspace/ckpts"}

TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
TOP_K=${TOP_K:-"20"}
DATASET=${DATASET:-"wikitext2"}
N_SAMPLES=${N_SAMPLES:-"32"}
BATCH_SIZE=${BATCH_SIZE:-"50"}
SEED=${SEED:-"2025"}
TRAIN_EPISODES=${TRAIN_EPISODES:-"5000"}
RUN_ID=${RUN_ID:-"p0_candidates"}

RUN_ROOT=${RUN_ROOT:-"${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/${RUN_ID}"}
CANDIDATE_DIR=${CANDIDATE_DIR:-"${RUN_ROOT}/candidates"}

mkdir -p "${RUN_ROOT}" "${CANDIDATE_DIR}"

export WIKITEXT2_PATH
export WIKITEXT2_CONFIG

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
  --num_collect=15 \
  --learning_epoch=10 \
  --reward=reward_ppl \
  --train_episode="${TRAIN_EPISODES}" \
  --seed="${SEED}" \
  --output="${RUN_ROOT}/logs" \
  --export_path="${RUN_ROOT}/endpoint_best.pth.tar" \
  --enable_downstream=false \
  --state_mode=0 \
  --save_candidates \
  --candidate_save_mode=topk \
  --candidate_top_k="${TOP_K}" \
  --candidate_dir="${CANDIDATE_DIR}" \
  --run_id="${RUN_ID}"

echo "Candidate pool: ${CANDIDATE_DIR}"
