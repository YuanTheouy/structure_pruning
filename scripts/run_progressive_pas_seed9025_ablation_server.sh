#!/usr/bin/env bash
set -Eeuo pipefail

# Replay-only same-pool ablation for OPT-2.7B seed9025 Progressive PAS.
# This script does not launch PPO search. It reuses the candidate pool from the
# existing seed9025 run and writes one replay directory per top_k/epsilon pair.

cd /workspace/structure_pruning

PYTHON_BIN=${PYTHON_BIN:-"python3"}
MODEL_NAME=${MODEL_NAME:-"opt-2.7b"}
MODEL=${MODEL:-"/workspace/Models/${MODEL_NAME}"}
SEED=${SEED:-"9025"}
TRAIN_EPISODES=${TRAIN_EPISODES:-"2000"}
TARGET_SPARSITY=${TARGET_SPARSITY:-"0.30"}
GPU_IDS=${GPU_IDS:-"4,5,6,7"}
N_SAMPLES=${N_SAMPLES:-"64"}
BATCH_SIZE=${BATCH_SIZE:-"50"}
PREFIX_STEPS=${PREFIX_STEPS:-"300,500,700,1000,1500,2000"}
STAGES=${STAGES:-"0.05,0.10,0.15,0.20,0.25,0.30"}
STAGE_WINDOW=${STAGE_WINDOW:-"0.015"}
MIN_PREFIX_STEP=${MIN_PREFIX_STEP:-"300"}
CARRY_FORWARD_MODE=${CARRY_FORWARD_MODE:-"none"}
DATASET_INITIAL_RATIO=${DATASET_INITIAL_RATIO:-"0.05"}
GRADUAL_PRUNING_SCHEDULE=${GRADUAL_PRUNING_SCHEDULE:-"staircase"}
RUN_ID=${RUN_ID_OVERRIDE:-"progressive_pas_lookahead_seed${SEED}_ep${TRAIN_EPISODES}_gpu4_fixed${DATASET_INITIAL_RATIO}_${GRADUAL_PRUNING_SCHEDULE}"}
RUN_ROOT=${RUN_ROOT_OVERRIDE:-"/workspace/ckpts/${MODEL_NAME}/sparsity_${TARGET_SPARSITY}/${RUN_ID}"}
CANDIDATE_DIR=${CANDIDATE_DIR_OVERRIDE:-"${RUN_ROOT}/candidates"}
OUT_ROOT=${OUT_ROOT_OVERRIDE:-"/workspace/ckpts/pas_progressive_lookahead/${MODEL_NAME}_seed${SEED}_ep${TRAIN_EPISODES}_fixed${DATASET_INITIAL_RATIO}_${GRADUAL_PRUNING_SCHEDULE}/ablation_official_20260615"}
LOG_DIR=${LOG_DIR_OVERRIDE:-"${OUT_ROOT}/logs"}
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

unset HF_DATASETS_OFFLINE
export HF_HOME="${HF_HOME:-/workspace/datasets/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/datasets/.cache/huggingface/datasets}"
export WIKITEXT2_PATH="${WIKITEXT2_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
export WIKITEXT2_CONFIG="${WIKITEXT2_CONFIG:-wikitext-2-raw-v1}"

if [[ ! -s "${CANDIDATE_DIR}/all_candidates.jsonl" && ! -s "${CANDIDATE_DIR}/candidates.jsonl" ]]; then
  echo "Missing candidate pool under ${CANDIDATE_DIR}" >&2
  echo "Set CANDIDATE_DIR_OVERRIDE to the existing seed9025 candidate directory." >&2
  exit 2
fi

if [[ ! -f "${WIKITEXT2_PATH}/dataset_dict.json" && ! -f "${WIKITEXT2_PATH}/state.json" ]]; then
  echo "Missing WikiText-2 dataset under ${WIKITEXT2_PATH}" >&2
  echo "Run scripts/download_p0_resources.py --skip_model before launching ablation." >&2
  exit 2
fi

run_replay() {
  local top_k="$1"
  local epsilon="$2"
  local margin="$3"
  local eps_label="${epsilon/./p}"
  local margin_label="${margin/./p}"
  local replay_dir="${OUT_ROOT}/topk${top_k}_eps${eps_label}_margin${margin_label}"
  local shared_probe_root="${OUT_ROOT}/topk${top_k}_eps0p05_margin0p00/probes"
  local shared_final_probe_root="${OUT_ROOT}/topk${top_k}_eps0p05_margin0p00/final_probes"
  mkdir -p "${replay_dir}"

  echo "===== replay top_k=${top_k} epsilon=${epsilon} margin=${margin} ====="
  echo "shared_probe_root=${shared_probe_root}"
  echo "shared_final_probe_root=${shared_final_probe_root}"
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
    --top-k "${top_k}" \
    --min-prefix-step "${MIN_PREFIX_STEP}" \
    --promotion-mode official \
    --promotion-min-candidates "${top_k}" \
    --carry-forward-mode "${CARRY_FORWARD_MODE}" \
    --epsilon "${epsilon}" \
    --margin "${margin}" \
    --projection-mode nested_from_base \
    --gpu-ids "${GPU_IDS}" \
    --n-samples "${N_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-dir "${replay_dir}" \
    --probe-root-dir "${shared_probe_root}" \
    --final-probe-root-dir "${shared_final_probe_root}" \
    2>&1 | tee "${LOG_DIR}/replay_topk${top_k}_eps${eps_label}_margin${margin_label}.log"

  "${PYTHON_BIN}" scripts/pas_progressive_efficiency_table.py \
    "${replay_dir}/progressive_pas_selection.csv" \
    --output-dir "${replay_dir}" \
    2>&1 | tee "${LOG_DIR}/efficiency_topk${top_k}_eps${eps_label}_margin${margin_label}.log"

  "${PYTHON_BIN}" scripts/pas_progressive_partial_report.py \
    "${replay_dir}" \
    --epsilon "${epsilon}" \
    2>&1 | tee "${LOG_DIR}/partial_topk${top_k}_eps${eps_label}_margin${margin_label}.log"
}

run_replay 20 0.05 0.00
run_replay 20 0.10 0.00
run_replay 50 0.05 0.00
run_replay 50 0.10 0.00

"${PYTHON_BIN}" scripts/summarize_progressive_pas_ablation.py \
  "${OUT_ROOT}"/topk*_eps*_margin* \
  --output-dir "${OUT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/ablation_summary.log"

echo "===== DONE ====="
echo "candidate_dir=${CANDIDATE_DIR}"
echo "ablation_dir=${OUT_ROOT}"
echo "summary=${OUT_ROOT}/progressive_pas_ablation_summary.md"
