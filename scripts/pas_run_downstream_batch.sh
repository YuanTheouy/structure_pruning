#!/usr/bin/env bash
set -euo pipefail

MODEL=""
DATASET="wikitext2"
SEED=""
CANDIDATE_POOL=""
TARGET_SIGMA="0.30"
PROBE_SIGMA="0.35"
HELDOUT_SIGMA="0.40"
OUTPUT_DIR=""
CHECKPOINT_TABLE=""
TASKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --candidate-pool) CANDIDATE_POOL="$2"; shift 2 ;;
    --target-sigma) TARGET_SIGMA="$2"; shift 2 ;;
    --probe-sigma) PROBE_SIGMA="$2"; shift 2 ;;
    --heldout-sigma) HELDOUT_SIGMA="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint-table) CHECKPOINT_TABLE="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$SEED" || -z "$CANDIDATE_POOL" || -z "$OUTPUT_DIR" || -z "$CHECKPOINT_TABLE" ]]; then
  echo "Missing required args: --model --seed --candidate-pool --output-dir --checkpoint-table" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
COMMANDS="$OUTPUT_DIR/downstream_commands.sh"

python3 - "$MODEL" "$CHECKPOINT_TABLE" "$TASKS" "$OUTPUT_DIR" > "$COMMANDS" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

model, checkpoint_table, tasks, output_dir = sys.argv[1:]
print("#!/usr/bin/env bash")
print("set -euo pipefail")
with Path(checkpoint_table).open("r", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        ckpt = row.get("checkpoint_path") or row.get("artifact_path")
        cid = row.get("candidate_id", "candidate")
        if not ckpt:
            continue
        out = Path(output_dir) / "lm_eval" / cid
        cmd = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model}",
            "--tasks",
            tasks,
            "--output_path",
            str(out),
        ]
        print("# TODO: adapt model_args to load the exported static checkpoint if required by the local evaluator.")
        print(" ".join(shlex.quote(part) for part in cmd))
PY

chmod +x "$COMMANDS"
echo "Wrote downstream commands: $COMMANDS"
if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

echo "Downstream execution is intentionally gated. Review $COMMANDS and adapt checkpoint loading before running."
