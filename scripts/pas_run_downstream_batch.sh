#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-python3}
MODEL=""
MODEL_NAME=""
DATASET="wikitext2"
SEED=""
CANDIDATE_POOL=""
TARGET_SIGMA="0.30"
PROBE_SIGMA="0.35"
HELDOUT_SIGMA="0.40"
OUTPUT_DIR=""
RECOVERY_TABLE=""
TASKS="piqa,hellaswag,winogrande,boolq"
LIMIT="100"
BATCH_SIZE="4"
EVAL_NUM_SAMPLES="64"
RECOVERY_METHOD="ffn_only_ridge_reconstruction"
RECON_SAMPLE="16"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
MAX_CANDIDATES=""
DRY_RUN="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --candidate-pool) CANDIDATE_POOL="$2"; shift 2 ;;
    --target-sigma) TARGET_SIGMA="$2"; shift 2 ;;
    --probe-sigma) PROBE_SIGMA="$2"; shift 2 ;;
    --heldout-sigma) HELDOUT_SIGMA="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --recovery-table) RECOVERY_TABLE="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --eval-num-samples) EVAL_NUM_SAMPLES="$2"; shift 2 ;;
    --recovery-method) RECOVERY_METHOD="$2"; shift 2 ;;
    --recon-sample) RECON_SAMPLE="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
    --force) FORCE="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$SEED" || -z "$CANDIDATE_POOL" || -z "$OUTPUT_DIR" || -z "$RECOVERY_TABLE" ]]; then
  echo "Missing required args: --model --seed --candidate-pool --output-dir --recovery-table" >&2
  exit 2
fi

MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
mkdir -p "$OUTPUT_DIR/downstream_eval" "$OUTPUT_DIR/logs"
LAUNCH_TSV="$OUTPUT_DIR/downstream_launch.tsv"

"$PYTHON_BIN" - \
  "$PYTHON_BIN" \
  "$MODEL" \
  "$MODEL_NAME" \
  "$DATASET" \
  "$SEED" \
  "$CANDIDATE_POOL" \
  "$TARGET_SIGMA" \
  "$PROBE_SIGMA" \
  "$HELDOUT_SIGMA" \
  "$OUTPUT_DIR" \
  "$RECOVERY_TABLE" \
  "$TASKS" \
  "$LIMIT" \
  "$BATCH_SIZE" \
  "$EVAL_NUM_SAMPLES" \
  "$RECOVERY_METHOD" \
  "$RECON_SAMPLE" \
  "$GPU_IDS" \
  "$MAX_CANDIDATES" \
  "$FORCE" <<'PY'
import csv
import json
import shlex
import sys
import time
from pathlib import Path

(
    python_bin,
    model,
    model_name,
    dataset,
    seed,
    candidate_pool,
    target_sigma,
    probe_sigma,
    heldout_sigma,
    output_dir,
    recovery_table,
    tasks,
    limit,
    batch_size,
    eval_num_samples,
    recovery_method,
    recon_sample,
    gpu_ids_raw,
    max_candidates,
    force,
) = sys.argv[1:]

repo = Path.cwd()
output = Path(output_dir)
gpu_ids = [gpu for gpu in gpu_ids_raw.replace(",", " ").split() if gpu]
if not gpu_ids:
    raise SystemExit("No GPU ids provided")

candidate_jsonl = Path(candidate_pool) / "candidates.jsonl"
with candidate_jsonl.open("r", encoding="utf-8") as handle:
    candidates = {}
    for line in handle:
        if not line.strip():
            continue
        candidate = json.loads(line)
        candidates[candidate["candidate_id"]] = candidate

with Path(recovery_table).open("r", encoding="utf-8") as handle:
    recovery_rows = [
        row for row in csv.DictReader(handle)
        if row.get("notes", "") == "same_protocol_recovery"
    ]

if max_candidates:
    recovery_rows = recovery_rows[: int(max_candidates)]

commands_by_gpu = {gpu: [] for gpu in gpu_ids}
assignments = []
for index, row in enumerate(recovery_rows):
    cid = row["candidate_id"]
    candidate = candidates.get(cid)
    if not candidate:
        raise SystemExit(f"Candidate {cid} not found in {candidate_jsonl}")
    gpu = gpu_ids[index % len(gpu_ids)]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cid)
    run_dir = output / "downstream_eval" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_candidate.json"
    export_path = run_dir / "checkpoint.pth.tar"
    final_policy_path = run_dir / "final_policy.json"
    downstream_output = run_dir / "downstream_results.json"
    payload = {
        "selected_mode": "stress_recovery_downstream",
        "candidate": candidate,
        "candidate_id": cid,
        "selection_tags": row.get("selection_tags", ""),
        "recovery_method": recovery_method,
        "source_recovery_table": recovery_table,
        "target_sparsity": float(target_sigma),
        "local_probe_sigma": float(probe_sigma),
        "heldout_sigma": float(heldout_sigma),
        "gpu_assignment": gpu,
    }
    best_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assignments.append({"candidate_id": cid, "gpu": gpu, "downstream_output": str(downstream_output)})
    if downstream_output.exists() and force.lower() != "true":
        commands_by_gpu[gpu].append(f"echo 'skip existing {downstream_output}'")
        continue
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "PYTHONUNBUFFERED=1",
        python_bin,
        "-u",
        str(repo / "amc_searchPPO.py"),
        "--job=compile",
        f"--model={model}",
        f"--model_name={model_name}",
        f"--dataset_name={dataset}",
        f"--preserve_ratio={1.0 - float(target_sigma):.6f}",
        f"--final_sparsity={target_sigma}",
        f"--best_candidate_path={best_path}",
        f"--export_path={export_path}",
        f"--final_policy_path={final_policy_path}",
        "--structure",
        "--prune=para",
        "--lbound=0.1",
        "--rbound=1.0",
        f"--n_samples={eval_num_samples}",
        "--data_bsize=8",
        f"--seed={seed}",
        "--gpu_id=0",
        "--recon",
        "--recon_ffn_only",
        f"--recon_sample={recon_sample}",
        "--enable_downstream=true",
        "--delayed_downstream_eval",
        f"--downstream_output={downstream_output}",
        f"--downstream_tasks={tasks}",
        f"--downstream_limit={limit}",
        f"--downstream_batch_size={batch_size}",
    ]
    commands_by_gpu[gpu].append(f"echo '=> downstream {cid} on visible GPU {gpu}'")
    commands_by_gpu[gpu].append(" ".join(shlex.quote(part) for part in cmd))

launch_rows = []
for gpu in gpu_ids:
    safe_gpu = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in gpu)
    command_path = output / f"downstream_commands_gpu{safe_gpu}.sh"
    log_path = output / "logs" / f"downstream_gpu{safe_gpu}.log"
    with command_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n")
        if commands_by_gpu[gpu]:
            handle.write("\n".join(commands_by_gpu[gpu]))
            handle.write("\n")
        else:
            handle.write("echo 'no candidates assigned'\n")
    command_path.chmod(0o755)
    launch_rows.append((gpu, str(command_path), str(log_path), len(commands_by_gpu[gpu])))

with (output / "downstream_launch.tsv").open("w", encoding="utf-8") as handle:
    for row in launch_rows:
        handle.write("\t".join(map(str, row)) + "\n")

manifest = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": model,
    "model_name": model_name,
    "dataset": dataset,
    "seed": seed,
    "candidate_pool": candidate_pool,
    "recovery_table": recovery_table,
    "target_sigma": float(target_sigma),
    "local_probe_sigma": float(probe_sigma),
    "heldout_sigma": float(heldout_sigma),
    "tasks": tasks,
    "limit": int(limit),
    "batch_size": int(batch_size),
    "eval_num_samples": int(eval_num_samples),
    "recovery_method": recovery_method,
    "recon_sample": int(recon_sample),
    "gpu_ids": gpu_ids,
    "candidate_count": len(recovery_rows),
    "same_protocol_for_all_candidates": True,
    "assignments": assignments,
    "artifacts": {
        "launch_tsv": str(output / "downstream_launch.tsv"),
        "logs_dir": str(output / "logs"),
        "downstream_eval_dir": str(output / "downstream_eval"),
    },
}
with (output / "downstream_multigpu_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(f"Wrote {output / 'downstream_launch.tsv'}")
print(f"Wrote {output / 'downstream_multigpu_manifest.json'}")
PY

echo "Wrote downstream shard commands under: $OUTPUT_DIR"
if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

echo "Downstream execution is gated by P1. If P1 passed, re-run with RUN_DOWNSTREAM_NOW=true."
if [[ "${RUN_DOWNSTREAM_NOW:-false}" != "true" ]]; then
  exit 0
fi

pids=()
status=0
while IFS=$'\t' read -r gpu command_path log_path command_count; do
  echo "=> Downstream shard GPU ${gpu}: ${command_path}, log ${log_path}, commands ${command_count}"
  bash "$command_path" > "$log_path" 2>&1 &
  pids+=("$!")
done < "$LAUNCH_TSV"

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one downstream shard failed. Inspect logs under $OUTPUT_DIR/logs" >&2
  exit "$status"
fi

echo "Downstream shards completed. Outputs under $OUTPUT_DIR/downstream_eval"
