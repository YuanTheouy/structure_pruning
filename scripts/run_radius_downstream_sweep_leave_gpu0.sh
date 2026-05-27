#!/usr/bin/env bash
set -Eeuo pipefail

# Long-running sweep wrapper. It intentionally leaves GPU 0 unused.
# Runs each setting sequentially, while each setting uses GPUs 1-7 for
# candidate-level probe/downstream parallelism.
#
# Override SETTINGS to choose the queue. Format per item:
#   model_name|model_path|seed|search_gpu|train_episode
#
# Example:
#   SETTINGS="opt-1.3b|/workspace/Models/opt-1.3b|6025|6|1000 opt-2.7b|/workspace/Models/opt-2.7b|7025|5|1000" \
#     bash scripts/run_radius_downstream_sweep_leave_gpu0.sh

cd /workspace/structure_pruning

GPU_IDS="${GPU_IDS:-1 2 3 4 5 6 7}"
DOWNSTREAM_LIMIT="${DOWNSTREAM_LIMIT:-100}"
TOP_K="${TOP_K:-40}"
N_SAMPLES="${N_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-50}"

DEFAULT_SETTINGS=(
  "opt-1.3b|/workspace/Models/opt-1.3b|6025|6|1000"
  "opt-2.7b|/workspace/Models/opt-2.7b|7025|5|1000"
)

if [[ -n "${SETTINGS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_SETTINGS=($SETTINGS)
else
  RUN_SETTINGS=("${DEFAULT_SETTINGS[@]}")
fi

echo "===== SWEEP CONFIG ====="
echo "GPU_IDS=$GPU_IDS"
echo "DOWNSTREAM_LIMIT=$DOWNSTREAM_LIMIT"
printf 'settings:\n'
printf '  %s\n' "${RUN_SETTINGS[@]}"

for spec in "${RUN_SETTINGS[@]}"; do
  IFS='|' read -r model_name model_path seed search_gpu train_episode <<< "$spec"
  if [[ -z "$model_name" || -z "$model_path" || -z "$seed" || -z "$search_gpu" || -z "$train_episode" ]]; then
    echo "Bad setting spec: $spec" >&2
    exit 2
  fi

  root="/workspace/ckpts/pas_informative_radius/${model_name}_seed${seed}_ff${train_episode}_growth_rep1"
  mkdir -p "$root"
  log="$root/run_all.log"

  echo
  echo "===== RUN $model_name seed=$seed search_gpu=$search_gpu train_episode=$train_episode ====="
  echo "log=$log"

  MODEL_NAME="$model_name" \
  MODEL="$model_path" \
  SEED="$seed" \
  GPU_SEARCH="$search_gpu" \
  GPU_IDS="$GPU_IDS" \
  TRAIN_EPISODE="$train_episode" \
  TOP_K="$TOP_K" \
  N_SAMPLES="$N_SAMPLES" \
  BATCH_SIZE="$BATCH_SIZE" \
  DOWNSTREAM_LIMIT="$DOWNSTREAM_LIMIT" \
  ROOT="$root" \
    bash scripts/run_opt13b_seed5025_radius_downstream.sh 2>&1 | tee "$log"

  echo "===== DONE $model_name seed=$seed ====="
  echo "summary=$root/FINAL_SUMMARY.md"
done

echo "===== SWEEP DONE ====="
