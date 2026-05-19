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
NUM_COLLECT=${NUM_COLLECT:-"15"}
LEARNING_EPOCH=${LEARNING_EPOCH:-"10"}
SEED=${SEED:-"2025"}
FUTURE_SPARSITY=${FUTURE_SPARSITY:-"0.40"}
FUTURE_N_SAMPLES=${FUTURE_N_SAMPLES:-"64"}

RUN_CANDIDATE_SEARCH=${RUN_CANDIDATE_SEARCH:-"true"}
RUN_EW=${RUN_EW:-"true"}
RUN_HIGH_SPARSITY=${RUN_HIGH_SPARSITY:-"true"}
RUN_CORRELATION=${RUN_CORRELATION:-"true"}

RUN_ROOT="${CKPT_ROOT}/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}"
CANDIDATE_DIR=${CANDIDATE_DIR:-"${RUN_ROOT}/p0_candidates/candidates"}
EW_OUTPUT_DIR=${EW_OUTPUT_DIR:-"${RUN_ROOT}/p0_ew"}
HIGH_SPARSITY_DIR=${HIGH_SPARSITY_DIR:-"${RUN_ROOT}/p0_high_sparsity"}
CORRELATION_DIR=${CORRELATION_DIR:-"${RUN_ROOT}/p0_future_correlation"}
PROBE_MINUS_SPARSITY=${PROBE_MINUS_SPARSITY:-"$(awk "BEGIN {printf \"%.2f\", ${TARGET_SPARSITY} - ${DELTA}}")"}
PROBE_PLUS_SPARSITY=${PROBE_PLUS_SPARSITY:-"$(awk "BEGIN {printf \"%.2f\", ${TARGET_SPARSITY} + ${DELTA}}")"}
HIGH_SPARSITIES=${HIGH_SPARSITIES:-"${TARGET_SPARSITY} ${PROBE_PLUS_SPARSITY} ${FUTURE_SPARSITY}"}
read -r -a HIGH_SPARSITY_ARRAY <<< "${HIGH_SPARSITIES}"

export MODEL MODEL_NAME WIKITEXT2_PATH CKPT_ROOT
export GPU_IDS TARGET_SPARSITY DELTA TRAIN_EPISODES N_SAMPLES TOP_K NUM_COLLECT LEARNING_EPOCH SEED

echo "=> Larger P0 run root: ${RUN_ROOT}"
echo "=> Probe sparsities: ${PROBE_MINUS_SPARSITY}, ${TARGET_SPARSITY}, ${PROBE_PLUS_SPARSITY}"
echo "=> Separated future target: logPPL(${FUTURE_SPARSITY}) - logPPL(${TARGET_SPARSITY})"

if [ "${RUN_CANDIDATE_SEARCH}" = "true" ]; then
  unset EPISODES_PER_WORKER
  bash scripts/run_p0_candidate_search_multigpu.sh
fi

if [ "${RUN_EW}" = "true" ]; then
  CANDIDATE_DIR="${CANDIDATE_DIR}" OUTPUT_DIR="${EW_OUTPUT_DIR}" bash scripts/run_ew_p0_multigpu.sh
fi

if [ "${RUN_HIGH_SPARSITY}" = "true" ]; then
  "${PYTHON_BIN}" evaluate_high_sparsity_curve.py \
    --selected_candidates_json "${EW_OUTPUT_DIR}/selected_candidates.json" \
    --model "${MODEL}" \
    --model_name "${MODEL_NAME}" \
    --sparsities "${HIGH_SPARSITY_ARRAY[@]}" \
    --num_samples "${FUTURE_N_SAMPLES}" \
    --output_dir "${HIGH_SPARSITY_DIR}"
fi

if [ "${RUN_CORRELATION}" = "true" ]; then
  "${PYTHON_BIN}" analyze_curvature_correlation.py \
    --probe_results "${EW_OUTPUT_DIR}/probe_results.csv" \
    --future_results_csv "${HIGH_SPARSITY_DIR}/high_sparsity_results.csv" \
    --target_sparsity "${TARGET_SPARSITY}" \
    --future_sparsity "${FUTURE_SPARSITY}" \
    --output_dir "${CORRELATION_DIR}"
fi

echo "=> Expected artifacts:"
echo "   ${CANDIDATE_DIR}/candidates.jsonl"
echo "   ${EW_OUTPUT_DIR}/probe_results.csv"
echo "   ${EW_OUTPUT_DIR}/rerank_results.csv"
echo "   ${EW_OUTPUT_DIR}/selected_candidates.json"
echo "   ${HIGH_SPARSITY_DIR}/high_sparsity_results.csv"
echo "   ${CORRELATION_DIR}/correlation_table.csv"
