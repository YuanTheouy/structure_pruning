#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/structure_pruning

PYTHON_BIN=${PYTHON_BIN:-"/root/venvs/ff/bin/python"}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

OUT_ROOT=${OUT_ROOT:-"/workspace/ckpts/fastforward_journal_p0_20260618"}
CKPT_ROOT=${CKPT_ROOT:-"${OUT_ROOT}/ckpts"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
SEED=${SEED:-"2025"}
OPT_EPISODES=${OPT_EPISODES:-"5000"}
LLAMA_EPISODES=${LLAMA_EPISODES:-"8000"}
N_SAMPLES=${N_SAMPLES:-"64"}
BATCH_SIZE=${BATCH_SIZE:-"8"}
RECON_SAMPLE=${RECON_SAMPLE:-"32"}
TOP_K=${TOP_K:-"50"}
SAVE_EVERY=${SAVE_EVERY:-"25"}
DATASET_INITIAL_RATIO=${DATASET_INITIAL_RATIO:-"0.05"}
DATASET_FINAL_RATIO=${DATASET_FINAL_RATIO:-"1.0"}
DATASET_GROWTH_END_EPISODE=${DATASET_GROWTH_END_EPISODE:-"1000"}
GRADUAL_INITIAL_SPARSITY=${GRADUAL_INITIAL_SPARSITY:-"0.05"}
GRADUAL_PRUNING_END_EPISODE=${GRADUAL_PRUNING_END_EPISODE:-"1000"}
WIKITEXT2_PATH=${WIKITEXT2_PATH:-"/workspace/datasets/wikitext/wikitext-2-raw-v1"}
WIKITEXT2_CONFIG=${WIKITEXT2_CONFIG:-"wikitext-2-raw-v1"}
OPT13_MODEL=${OPT13_MODEL:-"/workspace/Models/opt-1.3b"}
OPT27_MODEL=${OPT27_MODEL:-"/workspace/Models/opt-2.7b"}
LLAMA7B_MODEL=${LLAMA7B_MODEL:-"/workspace/Models/llama-2-7b-hf"}
ABLATION_MODEL_NAME=${ABLATION_MODEL_NAME:-"llama-2-7b-hf"}
ABLATION_MODEL=${ABLATION_MODEL:-"${LLAMA7B_MODEL}"}
RUN_CLEAN=${RUN_CLEAN:-"true"}
RUN_ABLATION=${RUN_ABLATION:-"true"}
RUN_DENSE_UNIFORM=${RUN_DENSE_UNIFORM:-"true"}
STOP_PAS=${STOP_PAS:-"true"}
FORCE_RERUN=${FORCE_RERUN:-"false"}

IFS=', ' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "GPU_IDS is empty" >&2
  exit 2
fi
MAX_PARALLEL=${MAX_PARALLEL:-"${#GPU_ARRAY[@]}"}
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo missing)
mkdir -p "${OUT_ROOT}/runs" "${OUT_ROOT}/logs" "${CKPT_ROOT}"

if [[ "${STOP_PAS}" == "true" ]]; then
  pkill -u "$USER" -f 'pas_progressive_lookahead_replay.py' 2>/dev/null || true
  pkill -u "$USER" -f 'amc_searchPPO.py --job=probe' 2>/dev/null || true
fi

if [[ ! -f "${WIKITEXT2_PATH}/dataset_dict.json" && ! -f "${WIKITEXT2_PATH}/state.json" ]]; then
  echo "Missing WikiText-2 under ${WIKITEXT2_PATH}" >&2
  echo "Run scripts/download_p0_resources.py --skip_model first." >&2
  exit 2
fi

model_specs=(
  "opt-1.3b|${OPT13_MODEL}|${OPT_EPISODES}"
  "opt-2.7b|${OPT27_MODEL}|${OPT_EPISODES}"
  "llama-2-7b-hf|${LLAMA7B_MODEL}|${LLAMA_EPISODES}"
)

for spec in "${model_specs[@]}"; do
  IFS='|' read -r model_name model_path _episodes <<< "${spec}"
  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model ${model_name}: ${model_path}" >&2
    echo "Set OPT13_MODEL, OPT27_MODEL, or LLAMA7B_MODEL to the correct server path." >&2
    exit 2
  fi
done

if [[ ! -d "${ABLATION_MODEL}" ]]; then
  echo "Ablation model ${ABLATION_MODEL} missing; falling back to OPT-2.7B." >&2
  ABLATION_MODEL_NAME="opt-2.7b"
  ABLATION_MODEL="${OPT27_MODEL}"
fi

safe_label() {
  echo "$1" | tr '/:.' '___' | tr -cd 'A-Za-z0-9_-'
}

sparsity_label() {
  echo "$1" | sed 's/0\.//; s/\./p/g'
}

stage_sparsities() {
  case "$1" in
    0.2|0.20) echo "0.05,0.10,0.15,0.20" ;;
    0.3|0.30) echo "0.05,0.10,0.15,0.20,0.25,0.30" ;;
    *) echo "${GRADUAL_INITIAL_SPARSITY},$1" ;;
  esac
}

episodes_for_model() {
  case "$1" in
    *llama*|*Llama*) echo "${LLAMA_EPISODES}" ;;
    *) echo "${OPT_EPISODES}" ;;
  esac
}

write_meta() {
  local path="$1"
  shift
  mkdir -p "$(dirname "${path}")"
  "$PYTHON_BIN" - "$path" "$@" <<'PY'
import json
import os
import sys

path = sys.argv[1]
pairs = sys.argv[2:]
payload = {}
for item in pairs:
    key, value = item.split("=", 1)
    payload[key] = value
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

compile_candidate() {
  local gpu="$1"
  local model_name="$2"
  local model_path="$3"
  local target_sparsity="$4"
  local seed="$5"
  local best_json="$6"
  local output_dir="$7"
  local recon="$8"
  local export_path="${output_dir}/final_static_checkpoint.pth.tar"
  local recon_flags=()
  if [[ "${recon}" == "true" ]]; then
    recon_flags+=(--recon --recon_sample="${RECON_SAMPLE}")
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u amc_searchPPO.py \
    --job=compile \
    --model="${model_path}" \
    --model_name="${model_name}" \
    --dataset_name=wikitext2 \
    --preserve_ratio="$(awk "BEGIN {printf \"%.6f\", 1 - ${target_sparsity}}")" \
    --final_sparsity="${target_sparsity}" \
    --best_candidate_path="${best_json}" \
    --export_path="${export_path}" \
    --final_policy_path="${output_dir}/final_policy.json" \
    --structure \
    --prune=para \
    --lbound=0.1 \
    --rbound=1.0 \
    --n_samples="${N_SAMPLES}" \
    --data_bsize="${BATCH_SIZE}" \
    --seed="${seed}" \
    --enable_downstream=false \
    "${recon_flags[@]}"
}

run_search_job() {
  local gpu="$1"
  local workstream="$2"
  local variant="$3"
  local model_name="$4"
  local model_path="$5"
  local target_sparsity="$6"
  local seed="$7"
  local train_episodes="$8"
  local use_gradual="$9"
  local use_growth="${10}"
  local action_wall_mode="${11}"
  local agent_path="${12}"

  local model_label
  model_label=$(safe_label "${model_name}")
  local sp_label
  sp_label=$(sparsity_label "${target_sparsity}")
  local run_id="${workstream}_${variant}_${model_label}_sp${sp_label}_seed${seed}"
  local run_dir="${OUT_ROOT}/runs/${run_id}"
  local run_root="${run_dir}/search"
  local candidate_dir="${run_root}/candidates"
  local log_path="${run_dir}/server.log"
  local meta_path="${run_dir}/run_metadata.json"

  if [[ "${FORCE_RERUN}" != "true" && -f "${meta_path}" ]] && grep -q '"status": "complete"' "${meta_path}"; then
    echo "=> SKIP complete ${run_id}"
    return 0
  fi

  mkdir -p "${run_dir}" "${candidate_dir}"
  local start_epoch
  start_epoch=$(date +%s)
  write_meta "${meta_path}" \
    "run_id=${run_id}" "status=running" "workstream=${workstream}" "variant=${variant}" \
    "model_name=${model_name}" "model_path=${model_path}" "target_sparsity=${target_sparsity}" \
    "seed=${seed}" "gpu_ids=${gpu}" "gpu_count=1" "train_episodes=${train_episodes}" \
    "n_samples=${N_SAMPLES}" "batch_size=${BATCH_SIZE}" "candidate_dir=${candidate_dir}" \
    "wikitext2_path=${WIKITEXT2_PATH}" "wikitext2_config=${WIKITEXT2_CONFIG}" \
    "commit_hash=${COMMIT_HASH}" "log_path=${log_path}" "start_epoch=${start_epoch}" \
    "calibration_recipe=raw compile plus ridge reconstruction compile with recon_sample=${RECON_SAMPLE}" \
    "notes=FastForward journal P0 one-shot run"

  local gradual_stages
  gradual_stages=$(stage_sparsities "${target_sparsity}")
  local search_start
  search_start=$(date +%s)
  {
    echo "===== ${run_id} ====="
    echo "gpu=${gpu} model=${model_name} sparsity=${target_sparsity} variant=${variant}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHON="${PYTHON_BIN}" \
    MODEL_NAME="${model_name}" \
    MODEL="${model_path}" \
    CKPT_ROOT="${CKPT_ROOT}" \
    TARGET_SPARSITY="${target_sparsity}" \
    TOP_K="${TOP_K}" \
    DATASET="wikitext2" \
    N_SAMPLES="${N_SAMPLES}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    SEED="${seed}" \
    GPU_ID="0" \
    TRAIN_EPISODES="${train_episodes}" \
    RUN_ID="${run_id}" \
    RUN_ROOT="${run_root}" \
    CANDIDATE_DIR="${candidate_dir}" \
    USE_GRADUAL_PRUNING="${use_gradual}" \
    GRADUAL_INITIAL_SPARSITY="${GRADUAL_INITIAL_SPARSITY}" \
    GRADUAL_PRUNING_END_EPISODE="${GRADUAL_PRUNING_END_EPISODE}" \
    GRADUAL_PRUNING_SCHEDULE="staircase" \
    GRADUAL_STAGE_SPARSITIES="${gradual_stages}" \
    USE_DATASET_GROWTH="${use_growth}" \
    DATASET_INITIAL_RATIO="${DATASET_INITIAL_RATIO}" \
    DATASET_FINAL_RATIO="${DATASET_FINAL_RATIO}" \
    DATASET_GROWTH_START_EPISODE="0" \
    DATASET_GROWTH_END_EPISODE="${DATASET_GROWTH_END_EPISODE}" \
    CANDIDATE_SAVE_MODE="topk_and_periodic" \
    SAVE_EVERY="${SAVE_EVERY}" \
    ACTION_WALL_MODE="${action_wall_mode}" \
    AGENT_PATH="${agent_path}" \
    WIKITEXT2_PATH="${WIKITEXT2_PATH}" \
    WIKITEXT2_CONFIG="${WIKITEXT2_CONFIG}" \
    bash scripts/run_p0_candidate_search.sh
  } 2>&1 | tee "${log_path}"
  local search_end
  search_end=$(date +%s)

  local best_json="${candidate_dir}/best_candidate.json"
  "${PYTHON_BIN}" scripts/make_best_candidate_from_pool.py \
    --candidate-dir "${candidate_dir}" \
    --output "${best_json}"

  local compile_start
  compile_start=$(date +%s)
  compile_candidate "${gpu}" "${model_name}" "${model_path}" "${target_sparsity}" "${seed}" "${best_json}" "${run_dir}/compile_raw" "false" \
    2>&1 | tee "${run_dir}/compile_raw.log"
  compile_candidate "${gpu}" "${model_name}" "${model_path}" "${target_sparsity}" "${seed}" "${best_json}" "${run_dir}/compile_calib" "true" \
    2>&1 | tee "${run_dir}/compile_calib.log"
  local compile_end
  compile_end=$(date +%s)
  local end_epoch
  end_epoch=$(date +%s)

  write_meta "${meta_path}" \
    "run_id=${run_id}" "status=complete" "workstream=${workstream}" "variant=${variant}" \
    "model_name=${model_name}" "model_path=${model_path}" "target_sparsity=${target_sparsity}" \
    "seed=${seed}" "gpu_ids=${gpu}" "gpu_count=1" "train_episodes=${train_episodes}" \
    "n_samples=${N_SAMPLES}" "batch_size=${BATCH_SIZE}" "candidate_dir=${candidate_dir}" \
    "best_candidate_path=${best_json}" "raw_metadata_path=${run_dir}/compile_raw/final_static_checkpoint.pth.json" \
    "calib_metadata_path=${run_dir}/compile_calib/final_static_checkpoint.pth.json" \
    "wikitext2_path=${WIKITEXT2_PATH}" "wikitext2_config=${WIKITEXT2_CONFIG}" \
    "commit_hash=${COMMIT_HASH}" "log_path=${log_path}" "start_epoch=${start_epoch}" \
    "search_start_epoch=${search_start}" "search_end_epoch=${search_end}" \
    "compile_start_epoch=${compile_start}" "compile_end_epoch=${compile_end}" "end_epoch=${end_epoch}" \
    "calibration_recipe=raw compile plus ridge reconstruction compile with recon_sample=${RECON_SAMPLE}" \
    "notes=FastForward journal P0 one-shot run"
}

run_export_job() {
  local gpu="$1"
  local workstream="$2"
  local variant="$3"
  local model_name="$4"
  local model_path="$5"
  local target_sparsity="$6"
  local preserve_ratio="$7"
  local recon="$8"
  local seed="$9"

  local model_label
  model_label=$(safe_label "${model_name}")
  local sp_label
  sp_label=$(sparsity_label "${target_sparsity}")
  local run_id="${workstream}_${variant}_${model_label}_sp${sp_label}_seed${seed}"
  local run_dir="${OUT_ROOT}/runs/${run_id}"
  local log_path="${run_dir}/server.log"
  local meta_path="${run_dir}/run_metadata.json"
  local export_path="${run_dir}/export/checkpoint.pth.tar"
  mkdir -p "${run_dir}/export"

  local start_epoch
  start_epoch=$(date +%s)
  local recon_flags=()
  local calib_recipe="none"
  if [[ "${recon}" == "true" ]]; then
    recon_flags+=(--recon --recon_sample="${RECON_SAMPLE}")
    calib_recipe="ridge reconstruction with recon_sample=${RECON_SAMPLE}"
  fi
  write_meta "${meta_path}" \
    "run_id=${run_id}" "status=running" "workstream=${workstream}" "variant=${variant}" \
    "model_name=${model_name}" "model_path=${model_path}" "target_sparsity=${target_sparsity}" \
    "seed=${seed}" "gpu_ids=${gpu}" "gpu_count=1" "train_episodes=0" \
    "n_samples=${N_SAMPLES}" "batch_size=${BATCH_SIZE}" "candidate_dir=" \
    "wikitext2_path=${WIKITEXT2_PATH}" "wikitext2_config=${WIKITEXT2_CONFIG}" \
    "commit_hash=${COMMIT_HASH}" "log_path=${log_path}" "start_epoch=${start_epoch}" \
    "calibration_recipe=${calib_recipe}" "notes=export baseline"

  {
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u amc_searchPPO.py \
      --job=export \
      --model="${model_path}" \
      --model_name="${model_name}" \
      --dataset_name=wikitext2 \
      --preserve_ratio="${preserve_ratio}" \
      --structure \
      --prune=para \
      --lbound=0.1 \
      --rbound=1.0 \
      --n_samples="${N_SAMPLES}" \
      --data_bsize="${BATCH_SIZE}" \
      --seed="${seed}" \
      --enable_downstream=false \
      --export_path="${export_path}" \
      "${recon_flags[@]}"
  } 2>&1 | tee "${log_path}"
  local end_epoch
  end_epoch=$(date +%s)
  local metadata_path="${run_dir}/export/checkpoint.pth.json"
  local raw_meta=""
  local calib_meta=""
  if [[ "${recon}" == "true" ]]; then
    calib_meta="${metadata_path}"
  else
    raw_meta="${metadata_path}"
  fi
  write_meta "${meta_path}" \
    "run_id=${run_id}" "status=complete" "workstream=${workstream}" "variant=${variant}" \
    "model_name=${model_name}" "model_path=${model_path}" "target_sparsity=${target_sparsity}" \
    "seed=${seed}" "gpu_ids=${gpu}" "gpu_count=1" "train_episodes=0" \
    "n_samples=${N_SAMPLES}" "batch_size=${BATCH_SIZE}" "candidate_dir=" \
    "raw_metadata_path=${raw_meta}" "calib_metadata_path=${calib_meta}" \
    "wikitext2_path=${WIKITEXT2_PATH}" "wikitext2_config=${WIKITEXT2_CONFIG}" \
    "commit_hash=${COMMIT_HASH}" "log_path=${log_path}" "start_epoch=${start_epoch}" \
    "search_start_epoch=${start_epoch}" "search_end_epoch=${start_epoch}" \
    "compile_start_epoch=${start_epoch}" "compile_end_epoch=${end_epoch}" "end_epoch=${end_epoch}" \
    "calibration_recipe=${calib_recipe}" "notes=export baseline"
}

pids=()
labels=()
gpu_cursor=0

next_gpu() {
  local gpu="${GPU_ARRAY[$((gpu_cursor % ${#GPU_ARRAY[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))
  echo "${gpu}"
}

wait_for_slot() {
  while [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; do
    local pid="${pids[0]}"
    local label="${labels[0]}"
    if ! wait "${pid}"; then
      echo "FAILED: ${label}" >&2
      exit 1
    fi
    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
  done
}

wait_all() {
  while [[ "${#pids[@]}" -gt 0 ]]; do
    local pid="${pids[0]}"
    local label="${labels[0]}"
    if ! wait "${pid}"; then
      echo "FAILED: ${label}" >&2
      exit 1
    fi
    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
  done
}

launch_search() {
  wait_for_slot
  local gpu
  gpu=$(next_gpu)
  local label="$2:$3:$4:$6"
  run_search_job "${gpu}" "$@" &
  pids+=("$!")
  labels+=("${label}")
  echo "=> launched ${label} on gpu=${gpu}"
}

launch_export() {
  wait_for_slot
  local gpu
  gpu=$(next_gpu)
  local label="$2:$3:$4:$6"
  run_export_job "${gpu}" "$@" &
  pids+=("$!")
  labels+=("${label}")
  echo "=> launched ${label} on gpu=${gpu}"
}

echo "===== FastForward Journal P0 ====="
echo "commit=${COMMIT_HASH}"
echo "out_root=${OUT_ROOT}"
echo "gpu_ids=${GPU_IDS}"
echo "max_parallel=${MAX_PARALLEL}"

if [[ "${RUN_CLEAN}" == "true" ]]; then
  echo "===== CLEAN MAIN-TABLE VERIFICATION: 3 models x 2 sparsities ====="
  for spec in "${model_specs[@]}"; do
    IFS='|' read -r model_name model_path train_episodes <<< "${spec}"
    for target_sparsity in 0.20 0.30; do
      launch_search "clean_main" "full_ff" "${model_name}" "${model_path}" "${target_sparsity}" "${SEED}" "${train_episodes}" "true" "true" "exact_projector" ""
    done
  done
  wait_all
fi

if [[ "${RUN_DENSE_UNIFORM}" == "true" ]]; then
  echo "===== DENSE + UNIFORM BASELINES FOR CLEAN MODELS ====="
  for spec in "${model_specs[@]}"; do
    IFS='|' read -r model_name model_path _train_episodes <<< "${spec}"
    launch_export "clean_baseline" "dense_raw" "${model_name}" "${model_path}" "0.00" "1.0" "false" "${SEED}"
    for target_sparsity in 0.20 0.30; do
      preserve_ratio=$(awk "BEGIN {printf \"%.6f\", 1 - ${target_sparsity}}")
      launch_export "clean_baseline" "uniform_calib" "${model_name}" "${model_path}" "${target_sparsity}" "${preserve_ratio}" "true" "${SEED}"
    done
  done
  wait_all
fi

if [[ "${RUN_ABLATION}" == "true" ]]; then
  echo "===== ONE-MODEL CORE ABLATION ====="
  ablation_episodes=$(episodes_for_model "${ABLATION_MODEL_NAME}")
  ablation_20_label=$(safe_label "${ABLATION_MODEL_NAME}")
  transfer_agent=$(find "${OUT_ROOT}/runs/clean_main_full_ff_${ablation_20_label}_sp20_seed${SEED}/search/logs" -name rl.pth.tar 2>/dev/null | sort | tail -1 || true)
  if [[ -z "${transfer_agent}" ]]; then
    echo "WARN: no 20% transfer agent found for ${ABLATION_MODEL_NAME}; full_transfer will run without transfer." >&2
  fi
  launch_search "core_ablation" "full_transfer" "${ABLATION_MODEL_NAME}" "${ABLATION_MODEL}" "0.30" "${SEED}" "${ablation_episodes}" "true" "true" "exact_projector" "${transfer_agent}"
  launch_search "core_ablation" "wo_transfer" "${ABLATION_MODEL_NAME}" "${ABLATION_MODEL}" "0.30" "${SEED}" "${ablation_episodes}" "true" "true" "exact_projector" ""
  launch_search "core_ablation" "wo_sparsity_curriculum" "${ABLATION_MODEL_NAME}" "${ABLATION_MODEL}" "0.30" "${SEED}" "${ablation_episodes}" "false" "true" "exact_projector" "${transfer_agent}"
  launch_search "core_ablation" "wo_fidelity_curriculum" "${ABLATION_MODEL_NAME}" "${ABLATION_MODEL}" "0.30" "${SEED}" "${ablation_episodes}" "true" "false" "exact_projector" "${transfer_agent}"
  launch_search "core_ablation" "legacy_soft_no_exact_projector" "${ABLATION_MODEL_NAME}" "${ABLATION_MODEL}" "0.30" "${SEED}" "${ablation_episodes}" "true" "true" "legacy_soft" "${transfer_agent}"
  preserve_ratio=$(awk "BEGIN {printf \"%.6f\", 1 - 0.30}")
  launch_export "core_ablation" "uniform_calib" "${ABLATION_MODEL_NAME}" "${ABLATION_MODEL}" "0.30" "${preserve_ratio}" "true" "${SEED}"
  wait_all
fi

"${PYTHON_BIN}" scripts/collect_fastforward_journal_p0.py "${OUT_ROOT}" --output-dir "${OUT_ROOT}/manifest"

echo "===== DONE ====="
echo "out_root=${OUT_ROOT}"
echo "manifest=${OUT_ROOT}/manifest/journal_p0_manifest.md"
echo "csv=${OUT_ROOT}/manifest/journal_p0_manifest.csv"
