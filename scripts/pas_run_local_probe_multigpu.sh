#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-python3}
MODEL=""
MODEL_NAME=""
DATASET="wikitext2"
SEED=""
CANDIDATE_POOL=""
OUTPUT_DIR=""
PROBE_SPARSITY=""
DELTA=""
BASE_SPARSITY=""
PROJECTION_MODE="nested_from_base"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
NUM_SAMPLES="64"
BATCH_SIZE="8"
CANDIDATE_TOP_K="0"
FORCE="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --candidate-pool) CANDIDATE_POOL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --probe-sparsity) PROBE_SPARSITY="$2"; shift 2 ;;
    --delta) DELTA="$2"; shift 2 ;;
    --base-sparsity) BASE_SPARSITY="$2"; shift 2 ;;
    --projection-mode) PROJECTION_MODE="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --candidate-top-k) CANDIDATE_TOP_K="$2"; shift 2 ;;
    --force) FORCE="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$SEED" || -z "$CANDIDATE_POOL" || -z "$OUTPUT_DIR" || -z "$PROBE_SPARSITY" || -z "$DELTA" ]]; then
  echo "Missing required args: --model --seed --candidate-pool --output-dir --probe-sparsity --delta" >&2
  exit 2
fi

MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL")}
BASE_SPARSITY=${BASE_SPARSITY:-$("$PYTHON_BIN" - "$PROBE_SPARSITY" "$DELTA" <<'PY'
import sys
probe = float(sys.argv[1])
delta = float(sys.argv[2])
print(f"{max(0.0, probe - delta):.6f}")
PY
)}

mkdir -p "$OUTPUT_DIR/shards" "$OUTPUT_DIR/logs"

"$PYTHON_BIN" - \
  "$PYTHON_BIN" \
  "$MODEL" \
  "$MODEL_NAME" \
  "$DATASET" \
  "$SEED" \
  "$CANDIDATE_POOL" \
  "$OUTPUT_DIR" \
  "$PROBE_SPARSITY" \
  "$DELTA" \
  "$BASE_SPARSITY" \
  "$PROJECTION_MODE" \
  "$GPU_IDS" \
  "$NUM_SAMPLES" \
  "$BATCH_SIZE" \
  "$CANDIDATE_TOP_K" \
  "$FORCE" \
  "$DRY_RUN" <<'PY'
import json
import shlex
import subprocess
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
    output_dir,
    probe_sparsity,
    delta,
    base_sparsity,
    projection_mode,
    gpu_ids_raw,
    num_samples,
    batch_size,
    candidate_top_k,
    force,
    dry_run,
) = sys.argv[1:]

repo = Path.cwd()
output = Path(output_dir)
gpu_ids = [gpu for gpu in gpu_ids_raw.replace(",", " ").split() if gpu]
if not gpu_ids:
    raise SystemExit("No GPU ids provided")

num_shards = len(gpu_ids)
preserve_ratio = 1.0 - float(base_sparsity)
candidate_path = Path(candidate_pool) / "candidates.jsonl"
if not candidate_path.exists():
    raise SystemExit(f"Missing candidate jsonl: {candidate_path}")

probe_csvs = []
manifest = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": model,
    "model_name": model_name,
    "dataset": dataset,
    "seed": seed,
    "candidate_pool": candidate_pool,
    "probe_sparsity": float(probe_sparsity),
    "delta": float(delta),
    "base_sparsity": float(base_sparsity),
    "projection_mode": projection_mode,
    "num_shards": num_shards,
    "gpu_ids": gpu_ids,
    "shards": [],
}

commands_by_gpu = {}
for shard_id, gpu in enumerate(gpu_ids):
    shard_dir = output / "shards" / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    probe_csv = shard_dir / "probe_results.csv"
    probe_jsonl = shard_dir / "probe_results.jsonl"
    probe_csvs.append(str(probe_csv))
    if force.lower() == "true":
        for path in (probe_csv, probe_jsonl):
            if path.exists():
                path.unlink()

    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "PYTHONUNBUFFERED=1",
        python_bin,
        "-u",
        str(repo / "amc_searchPPO.py"),
        "--job=probe",
        f"--model={model}",
        f"--model_name={model_name}",
        f"--dataset_name={dataset}",
        f"--preserve_ratio={preserve_ratio:.6f}",
        f"--candidates_path={candidate_path}",
        f"--candidate_top_k={candidate_top_k}",
        f"--probe_sparsity={probe_sparsity}",
        f"--ew_delta={delta}",
        f"--projection_mode={projection_mode}",
        f"--projection_base_sparsity={base_sparsity}",
        f"--probe_output={probe_csv}",
        f"--probe_jsonl_output={probe_jsonl}",
        f"--num_shards={num_shards}",
        f"--shard_id={shard_id}",
        "--structure",
        "--prune=para",
        "--lbound=0.1",
        "--rbound=1.0",
        f"--n_samples={num_samples}",
        f"--data_bsize={batch_size}",
        f"--seed={seed}",
        "--gpu_id=0",
        "--enable_downstream=false",
    ]
    commands_by_gpu[gpu] = cmd
    command_path = output / f"probe_commands_gpu{gpu}.sh"
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(cmd) + "\n", encoding="utf-8")
    command_path.chmod(0o755)
    manifest["shards"].append(
        {
            "shard_id": shard_id,
            "gpu": gpu,
            "command_path": str(command_path),
            "log_path": str(output / "logs" / f"probe_gpu{gpu}.log"),
            "probe_csv": str(probe_csv),
        }
    )

(output / "local_probe_multigpu_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {output / 'local_probe_multigpu_manifest.json'}")
for shard in manifest["shards"]:
    print(f"=> Probe shard GPU {shard['gpu']}: {shard['command_path']}, log {shard['log_path']}")

if dry_run.lower() == "true":
    raise SystemExit(0)

processes = []
for gpu, cmd in commands_by_gpu.items():
    log_path = output / "logs" / f"probe_gpu{gpu}.log"
    handle = log_path.open("w", encoding="utf-8")
    processes.append((gpu, handle, subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)))

failed = False
for gpu, handle, process in processes:
    rc = process.wait()
    handle.close()
    if rc != 0:
        print(f"=> Probe shard on GPU {gpu} failed; see {output / 'logs' / f'probe_gpu{gpu}.log'}", file=sys.stderr)
        failed = True
if failed:
    raise SystemExit(1)

merge_cmd = [python_bin, str(repo / "scripts" / "merge_ew_probe_results.py"), "--output_dir", str(output)] + probe_csvs
subprocess.run(merge_cmd, check=True)
print(f"Local probe complete: {output / 'probe_results.csv'}")
PY
