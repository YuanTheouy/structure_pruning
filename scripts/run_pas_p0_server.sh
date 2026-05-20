#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-"python3"}
MODEL=${MODEL:-"/workspace/Models/opt-2.7b"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-"/workspace/datasets/wikitext/wikitext-2-raw-v1"}
CKPT_ROOT=${CKPT_ROOT:-"/workspace/ckpts"}

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
DELTA=${DELTA:-"0.05"}
TRAIN_EPISODES=${TRAIN_EPISODES:-"5000"}
N_SAMPLES=${N_SAMPLES:-"32"}
TOP_K=${TOP_K:-"20"}
SHORTLIST_SIZE=${SHORTLIST_SIZE:-"5"}
RANDOM_REPEATS=${RANDOM_REPEATS:-"1000"}
NUM_COLLECT=${NUM_COLLECT:-"15"}
LEARNING_EPOCH=${LEARNING_EPOCH:-"10"}
SEED=${SEED:-"2025"}
HELDOUT_SPARSITY=${HELDOUT_SPARSITY:-"0.40"}
HELDOUT_N_SAMPLES=${HELDOUT_N_SAMPLES:-"64"}
BATCH_SIZE=${BATCH_SIZE:-"50"}

RUN_CANDIDATE_SEARCH=${RUN_CANDIDATE_SEARCH:-"true"}
RUN_PROBE=${RUN_PROBE:-"true"}
RUN_PAS=${RUN_PAS:-"true"}

RUN_ROOT="${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}"
CANDIDATE_DIR=${CANDIDATE_DIR:-"${RUN_ROOT}/p0_candidates/candidates"}
EW_OUTPUT_DIR=${EW_OUTPUT_DIR:-"${RUN_ROOT}/p0_ew"}
PAS_OUTPUT_DIR=${PAS_OUTPUT_DIR:-"${RUN_ROOT}/p0_pas"}

export MODEL MODEL_NAME WIKITEXT2_PATH CKPT_ROOT
export GPU_IDS TARGET_SPARSITY DELTA TRAIN_EPISODES N_SAMPLES TOP_K NUM_COLLECT LEARNING_EPOCH SEED BATCH_SIZE

echo "=> PAS P0 run root: ${RUN_ROOT}"
echo "=> Candidate pool: ${CANDIDATE_DIR}"
echo "=> Probe budgets: $(awk "BEGIN {printf \"%.2f\", ${TARGET_SPARSITY} - ${DELTA}}") ${TARGET_SPARSITY} $(awk "BEGIN {printf \"%.2f\", ${TARGET_SPARSITY} + ${DELTA}}")"
echo "=> Held-out future budget: ${HELDOUT_SPARSITY} (analysis only)"

if [ "${RUN_CANDIDATE_SEARCH}" = "true" ]; then
  unset EPISODES_PER_WORKER
  bash scripts/run_p0_candidate_search_multigpu.sh
fi

if [ "${RUN_PROBE}" = "true" ]; then
  CANDIDATE_DIR="${CANDIDATE_DIR}" OUTPUT_DIR="${EW_OUTPUT_DIR}" bash scripts/run_ew_p0_multigpu.sh
fi

if [ "${RUN_PAS}" = "true" ]; then
  "${PYTHON_BIN}" pas_evidence_pipeline.py \
    --candidate_dir "${CANDIDATE_DIR}" \
    --probe_results "${EW_OUTPUT_DIR}/probe_results.csv" \
    --model "${MODEL}" \
    --model_name "${MODEL_NAME}" \
    --target_sparsity "${TARGET_SPARSITY}" \
    --delta "${DELTA}" \
    --future_sparsity "${HELDOUT_SPARSITY}" \
    --top_k "${TOP_K}" \
    --shortlist_size "${SHORTLIST_SIZE}" \
    --seed "${SEED}" \
    --random_repeats "${RANDOM_REPEATS}" \
    --probe_num_samples "${N_SAMPLES}" \
    --heldout_num_samples "${HELDOUT_N_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${PAS_OUTPUT_DIR}"
fi

echo "=> Expected PAS artifacts:"
echo "   ${PAS_OUTPUT_DIR}/probe_results.csv"
echo "   ${PAS_OUTPUT_DIR}/heldout_results.csv"
echo "   ${PAS_OUTPUT_DIR}/warning_correlation.csv"
echo "   ${PAS_OUTPUT_DIR}/selection_regret.csv"
echo "   ${PAS_OUTPUT_DIR}/selected_candidates.json"
echo "   ${PAS_OUTPUT_DIR}/path_divergence.pdf"
echo "   ${PAS_OUTPUT_DIR}/endpoint_ambiguity_scatter.pdf"
echo "   ${PAS_OUTPUT_DIR}/warning_correlation.pdf"
echo "   ${PAS_OUTPUT_DIR}/artifact_manifest.json"
