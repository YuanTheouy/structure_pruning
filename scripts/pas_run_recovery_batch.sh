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
RECOVERY_METHOD="ridge_reconstruction"
RECON_SAMPLE="16"
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
mkdir -p "$OUTPUT_DIR/recovery_eval"
COMMANDS="$OUTPUT_DIR/recovery_commands.sh"

"$PYTHON_BIN" - "$MODEL" "$MODEL_NAME" "$DATASET" "$SEED" "$CANDIDATE_POOL" "$TARGET_SIGMA" "$PROBE_SIGMA" "$HELDOUT_SIGMA" "$OUTPUT_DIR" "$RECOVERY_SUBSET" "$BATCH_SIZE" "$NUM_SAMPLES" "$RECOVERY_METHOD" "$RECON_SAMPLE" "$FORCE" > "$COMMANDS" <<'PY'
import csv
import json
import shlex
import sys
from pathlib import Path

(
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
    force,
) = sys.argv[1:]

repo = Path.cwd()
candidate_jsonl = Path(candidate_pool) / "candidates.jsonl"
with candidate_jsonl.open("r", encoding="utf-8") as handle:
    candidates = {json.loads(line)["candidate_id"]: json.loads(line) for line in handle if line.strip()}

with Path(recovery_subset).open("r", encoding="utf-8") as handle:
    subset_rows = list(csv.DictReader(handle))

print("#!/usr/bin/env bash")
print("set -euo pipefail")
for row in subset_rows:
    cid = row["candidate_id"]
    candidate = candidates.get(cid)
    if not candidate:
        raise SystemExit(f"Candidate {cid} not found in {candidate_jsonl}")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cid)
    run_dir = Path(output_dir) / "recovery_eval" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_candidate.json"
    export_path = run_dir / "checkpoint.pth.tar"
    final_policy_path = run_dir / "final_policy.json"
    payload = {
        "selected_mode": "stress_recovery_subset",
        "candidate": candidate,
        "candidate_id": cid,
        "selection_tags": row.get("selection_tags", ""),
        "recovery_method": recovery_method,
        "target_sparsity": float(target_sigma),
        "local_probe_sigma": float(probe_sigma),
        "heldout_sigma": float(heldout_sigma),
    }
    best_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path = Path(str(export_path).rsplit(".", 1)[0] + ".json")
    if metadata_path.exists() and force.lower() != "true":
        print(f"echo 'skip existing {metadata_path}'")
        continue
    cmd = [
        sys.executable,
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
        "--enable_downstream=false",
        "--recon",
        f"--recon_sample={recon_sample}",
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
PY

chmod +x "$COMMANDS"
echo "Wrote recovery commands: $COMMANDS"
if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

bash "$COMMANDS"
