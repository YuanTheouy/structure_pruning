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
RECOVERY_SUBSET=""
BATCH_SIZE="8"
NUM_SAMPLES="64"
RECOVERY_METHOD="ffn_only_ridge_reconstruction"
RECON_SAMPLE="16"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
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
    --recovery-subset) RECOVERY_SUBSET="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --recovery-method) RECOVERY_METHOD="$2"; shift 2 ;;
    --recon-sample) RECON_SAMPLE="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --force) FORCE="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$SEED" || -z "$CANDIDATE_POOL" || -z "$OUTPUT_DIR" || -z "$RECOVERY_SUBSET" ]]; then
  echo "Missing required args: --model --seed --candidate-pool --output-dir --recovery-subset" >&2
  exit 2
fi

MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
mkdir -p "$OUTPUT_DIR/recovery_eval" "$OUTPUT_DIR/logs"
LAUNCH_TSV="$OUTPUT_DIR/recovery_launch.tsv"

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
  "$RECOVERY_SUBSET" \
  "$BATCH_SIZE" \
  "$NUM_SAMPLES" \
  "$RECOVERY_METHOD" \
  "$RECON_SAMPLE" \
  "$GPU_IDS" \
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
    recovery_subset,
    batch_size,
    num_samples,
    recovery_method,
    recon_sample,
    gpu_ids_raw,
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

with Path(recovery_subset).open("r", encoding="utf-8") as handle:
    subset_rows = list(csv.DictReader(handle))

launch_rows = []
commands_by_gpu = {gpu: [] for gpu in gpu_ids}
assignments = []

for index, row in enumerate(subset_rows):
    cid = row["candidate_id"]
    candidate = candidates.get(cid)
    if not candidate:
        raise SystemExit(f"Candidate {cid} not found in {candidate_jsonl}")
    gpu = gpu_ids[index % len(gpu_ids)]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cid)
    run_dir = output / "recovery_eval" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_candidate.json"
    export_path = run_dir / "checkpoint.pth.tar"
    final_policy_path = run_dir / "final_policy.json"
    metadata_path = Path(str(export_path).rsplit(".", 1)[0] + ".json")
    payload = {
        "selected_mode": "stress_recovery_subset",
        "candidate": candidate,
        "candidate_id": cid,
        "selection_tags": row.get("selection_tags", ""),
        "recovery_method": recovery_method,
        "target_sparsity": float(target_sigma),
        "local_probe_sigma": float(probe_sigma),
        "heldout_sigma": float(heldout_sigma),
        "gpu_assignment": gpu,
    }
    best_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assignments.append({"candidate_id": cid, "gpu": gpu, "metadata_path": str(metadata_path)})

    if metadata_path.exists() and force.lower() != "true":
        commands_by_gpu[gpu].append(f"echo 'skip existing {metadata_path}'")
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
        f"--n_samples={num_samples}",
        f"--data_bsize={batch_size}",
        f"--seed={seed}",
        "--gpu_id=0",
        "--enable_downstream=false",
        "--recon",
        "--recon_ffn_only",
        f"--recon_sample={recon_sample}",
    ]
    commands_by_gpu[gpu].append(f"echo '=> recovery {cid} on visible GPU {gpu}'")
    commands_by_gpu[gpu].append(" ".join(shlex.quote(part) for part in cmd))

for gpu in gpu_ids:
    safe_gpu = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in gpu)
    command_path = output / f"recovery_commands_gpu{safe_gpu}.sh"
    log_path = output / "logs" / f"recovery_gpu{safe_gpu}.log"
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

with (output / "recovery_launch.tsv").open("w", encoding="utf-8") as handle:
    for row in launch_rows:
        handle.write("\t".join(map(str, row)) + "\n")

manifest = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": model,
    "model_name": model_name,
    "dataset": dataset,
    "seed": seed,
    "candidate_pool": candidate_pool,
    "recovery_subset": recovery_subset,
    "target_sigma": float(target_sigma),
    "local_probe_sigma": float(probe_sigma),
    "heldout_sigma": float(heldout_sigma),
    "num_samples": int(num_samples),
    "batch_size": int(batch_size),
    "recovery_method": recovery_method,
    "recon_sample": int(recon_sample),
    "gpu_ids": gpu_ids,
    "same_protocol_for_all_candidates": True,
    "candidate_count": len(subset_rows),
    "assignments": assignments,
    "artifacts": {
        "launch_tsv": str(output / "recovery_launch.tsv"),
        "logs_dir": str(output / "logs"),
        "recovery_eval_dir": str(output / "recovery_eval"),
    },
}
with (output / "recovery_multigpu_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(f"Wrote {output / 'recovery_launch.tsv'}")
print(f"Wrote {output / 'recovery_multigpu_manifest.json'}")
PY

echo "Wrote recovery shard commands under: $OUTPUT_DIR"
if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

pids=()
status=0
while IFS=$'\t' read -r gpu command_path log_path command_count; do
  echo "=> Recovery shard GPU ${gpu}: ${command_path}, log ${log_path}, commands ${command_count}"
  bash "$command_path" > "$log_path" 2>&1 &
  pids+=("$!")
done < "$LAUNCH_TSV"

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one recovery shard failed. Inspect logs under $OUTPUT_DIR/logs" >&2
  exit "$status"
fi

echo "Recovery shards completed. Outputs under $OUTPUT_DIR/recovery_eval"
