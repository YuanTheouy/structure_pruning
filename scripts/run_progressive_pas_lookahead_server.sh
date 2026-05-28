#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot server runner for Progressive PAS Lookahead.
# Defaults intentionally use GPUs 4,5,6,7 and avoid the old 400-episode debug pool.

cd /workspace/structure_pruning

PYTHON_BIN=${PYTHON_BIN:-"python3"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
MODEL=${MODEL:-"/workspace/Models/${MODEL_NAME}"}
SEED=${SEED:-"9025"}
TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
TRAIN_EPISODES=${TRAIN_EPISODES:-"5000"}
SEARCH_GPU=${SEARCH_GPU:-"4"}
GPU_IDS=${GPU_IDS:-"4,5,6,7"}
TOP_K=${TOP_K:-"50"}
REPLAY_TOP_K=${REPLAY_TOP_K:-"20"}
N_SAMPLES=${N_SAMPLES:-"64"}
BATCH_SIZE=${BATCH_SIZE:-"50"}
PREFIX_STEPS=${PREFIX_STEPS:-"300,500,700,1000,1500,2000,5000"}
STAGES=${STAGES:-"0.05,0.10,0.15,0.20,0.25,0.30"}
STAGE_WINDOW=${STAGE_WINDOW:-"0.015"}
EPSILON=${EPSILON:-"0.05"}
MARGIN=${MARGIN:-"0.02"}
SAVE_EVERY=${SAVE_EVERY:-"25"}
GRADUAL_INITIAL_SPARSITY=${GRADUAL_INITIAL_SPARSITY:-"0.05"}
GRADUAL_PRUNING_END_EPISODE=${GRADUAL_PRUNING_END_EPISODE:-"1000"}

RUN_ID_DEFAULT="progressive_pas_lookahead_seed${SEED}_ep${TRAIN_EPISODES}_gpu${SEARCH_GPU}"
RUN_ID=${RUN_ID_OVERRIDE:-"${RUN_ID_DEFAULT}"}
RUN_ROOT=${RUN_ROOT_OVERRIDE:-"/workspace/ckpts/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/${RUN_ID}"}
CANDIDATE_DIR=${CANDIDATE_DIR_OVERRIDE:-"${RUN_ROOT}/candidates"}
OUT_ROOT=${OUT_ROOT_OVERRIDE:-"/workspace/ckpts/pas_progressive_lookahead/${MODEL_NAME}_seed${SEED}_ep${TRAIN_EPISODES}"}
REPLAY_DIR=${REPLAY_DIR_OVERRIDE:-"${OUT_ROOT}/replay"}
LOG_DIR=${LOG_DIR_OVERRIDE:-"${OUT_ROOT}/logs"}
mkdir -p "${RUN_ROOT}" "${CANDIDATE_DIR}" "${REPLAY_DIR}" "${LOG_DIR}"

unset HF_DATASETS_OFFLINE
unset HF_ENDPOINT
export HF_HOME="${HF_HOME:-/workspace/datasets/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/datasets/.cache/huggingface/datasets}"
export WIKITEXT2_PATH="${WIKITEXT2_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
export WIKITEXT2_CONFIG="${WIKITEXT2_CONFIG:-wikitext-2-raw-v1}"

cat <<EOF
===== Progressive PAS Lookahead Server Run =====
MODEL_NAME=${MODEL_NAME}
MODEL=${MODEL}
SEED=${SEED}
TRAIN_EPISODES=${TRAIN_EPISODES}
SEARCH_GPU=${SEARCH_GPU}
GPU_IDS=${GPU_IDS}
RUN_ROOT=${RUN_ROOT}
CANDIDATE_DIR=${CANDIDATE_DIR}
REPLAY_DIR=${REPLAY_DIR}
EOF

echo "===== PHASE 1: gradual FastForward candidate search ====="
if [[ -s "${CANDIDATE_DIR}/all_candidates.jsonl" ]]; then
  echo "SKIP search: existing ${CANDIDATE_DIR}/all_candidates.jsonl"
else
  CUDA_VISIBLE_DEVICES="${SEARCH_GPU}" \
  PYTHON="${PYTHON_BIN}" \
  MODEL_NAME="${MODEL_NAME}" \
  MODEL="${MODEL}" \
  SEED="${SEED}" \
  GPU_ID="0" \
  TARGET_SPARSITY="${TARGET_SPARSITY}" \
  TOP_K="${TOP_K}" \
  N_SAMPLES="${N_SAMPLES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TRAIN_EPISODES="${TRAIN_EPISODES}" \
  RUN_ID="${RUN_ID}" \
  RUN_ROOT="${RUN_ROOT}" \
  CANDIDATE_DIR="${CANDIDATE_DIR}" \
  USE_GRADUAL_PRUNING="true" \
  GRADUAL_INITIAL_SPARSITY="${GRADUAL_INITIAL_SPARSITY}" \
  GRADUAL_PRUNING_END_EPISODE="${GRADUAL_PRUNING_END_EPISODE}" \
  CANDIDATE_SAVE_MODE="topk_and_periodic" \
  SAVE_EVERY="${SAVE_EVERY}" \
    bash scripts/run_p0_candidate_search.sh 2>&1 | tee "${LOG_DIR}/search.log"
fi

echo "===== PHASE 2: offline Progressive PAS lookahead replay ====="
"${PYTHON_BIN}" scripts/pas_progressive_lookahead_replay.py \
  --model "${MODEL}" \
  --model-name "${MODEL_NAME}" \
  --dataset wikitext2 \
  --candidate-dir "${CANDIDATE_DIR}" \
  --target-sparsity "${TARGET_SPARSITY}" \
  --heldout-sparsity 0.40 \
  --stages "${STAGES}" \
  --prefix-steps "${PREFIX_STEPS}" \
  --stage-window "${STAGE_WINDOW}" \
  --top-k "${REPLAY_TOP_K}" \
  --epsilon "${EPSILON}" \
  --margin "${MARGIN}" \
  --projection-mode nested_from_base \
  --gpu-ids "${GPU_IDS}" \
  --n-samples "${N_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --output-dir "${REPLAY_DIR}" \
  2>&1 | tee "${LOG_DIR}/replay.log"

echo "===== PHASE 3: efficiency summary ====="
"${PYTHON_BIN}" scripts/pas_progressive_efficiency_table.py \
  "${REPLAY_DIR}/progressive_pas_selection.csv" \
  --output-dir "${REPLAY_DIR}" \
  2>&1 | tee "${LOG_DIR}/efficiency.log"

echo "===== DONE ====="
echo "summary=${REPLAY_DIR}/progressive_pas_summary.md"
echo "efficiency=${REPLAY_DIR}/progressive_pas_efficiency.md"
cat "${REPLAY_DIR}/progressive_pas_efficiency.md"
