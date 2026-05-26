#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end OPT-1.3B replicate for the radius/downstream story.
#
# What it runs:
#   1. Single PPO search, one trajectory, 1000 episodes.
#   2. Fixed-5% path probes for 30/35/40 and 30.25/30.5/31.
#   3. Downstream@30 on top15-by-L30 plus key PAS/oracle candidates.
#   4. Final markdown summary under $ROOT/FINAL_SUMMARY.md.
#
# Defaults match the 2026-05-26 seed4025 pilot. Override through env vars:
#   SEED=5025 GPU_SEARCH=6 GPU_IDS="0 1 2 3 4 5 6 7" bash scripts/run_opt13b_seed5025_radius_downstream.sh

cd /workspace/structure_pruning

SEED="${SEED:-5025}"
MODEL="${MODEL:-/workspace/Models/opt-1.3b}"
MODEL_NAME="${MODEL_NAME:-opt-1.3b}"
TARGET="${TARGET:-0.30}"
PRESERVE="${PRESERVE:-0.700000}"
GPU_SEARCH="${GPU_SEARCH:-6}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
TOP_K="${TOP_K:-40}"
N_SAMPLES="${N_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-50}"
TRAIN_EPISODE="${TRAIN_EPISODE:-1000}"
DOWNSTREAM_LIMIT="${DOWNSTREAM_LIMIT:-100}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-4}"
TASKS="${TASKS:-piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

RUN_ID="${RUN_ID:-ff_single_seed${SEED}_ep${TRAIN_EPISODE}_gpu${GPU_SEARCH}_growth5to100_rep1}"
SEARCH="${SEARCH:-/workspace/ckpts/${MODEL_NAME}/sparsity_${TARGET}/${RUN_ID}}"
CAND="${CAND:-${SEARCH}/candidates}"
ROOT="${ROOT:-/workspace/ckpts/pas_informative_radius/${MODEL_NAME}_seed${SEED}_ff1000_growth_rep1}"
PATH40="${PATH40:-${ROOT}/path30_35_40_fixed5}"
LOCAL="${LOCAL:-${ROOT}/local_radius_fixed5}"
DOWN="${DOWN:-${ROOT}/downstream30_local_radius}"

export PYTHON_BIN MODEL MODEL_NAME SEED CAND GPU_IDS TOP_K N_SAMPLES BATCH_SIZE ROOT PATH40 LOCAL DOWN

mkdir -p "$SEARCH" "$ROOT" "$PATH40" "$LOCAL" "$DOWN"

echo "===== ENV CHECK ====="
echo "python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY' || pip install 'tokenizers>=0.20,<0.21'
import tokenizers
parts = tuple(int(x) for x in tokenizers.__version__.split(".")[:2])
print("tokenizers", tokenizers.__version__)
raise SystemExit(0 if (0, 20) <= parts < (0, 21) else 1)
PY

"$PYTHON_BIN" - <<'PY'
import sys
import transformers
import tokenizers
import datasets
print(sys.executable)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("datasets ok")
PY

unset HF_DATASETS_OFFLINE
unset HF_ENDPOINT
export HF_HOME="${HF_HOME:-/workspace/datasets/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/datasets/.cache/huggingface/datasets}"
export WIKITEXT2_PATH="${WIKITEXT2_PATH:-/workspace/datasets/wikitext/wikitext-2-raw-v1}"
export WIKITEXT2_CONFIG="${WIKITEXT2_CONFIG:-wikitext-2-raw-v1}"

echo "===== CONFIG ====="
cat <<EOF
SEED=$SEED
MODEL=$MODEL
MODEL_NAME=$MODEL_NAME
GPU_SEARCH=$GPU_SEARCH
GPU_IDS=$GPU_IDS
SEARCH=$SEARCH
CAND=$CAND
ROOT=$ROOT
PATH40=$PATH40
LOCAL=$LOCAL
DOWN=$DOWN
TASKS=$TASKS
EOF

echo "===== PHASE 1: SINGLE SEARCH ====="
if [[ -s "$CAND/candidates.jsonl" ]] && [[ "$(wc -l < "$CAND/candidates.jsonl")" -ge "$TOP_K" ]]; then
  echo "SKIP search: existing $CAND/candidates.jsonl"
else
  mkdir -p "$CAND"
  CUDA_VISIBLE_DEVICES="$GPU_SEARCH" "$PYTHON_BIN" -u amc_searchPPO.py \
    --model "$MODEL" \
    --model_name "$MODEL_NAME" \
    --dataset_name wikitext2 \
    --preserve_ratio "$PRESERVE" \
    --structure \
    --prune para \
    --lbound 0.1 \
    --rbound 1.0 \
    --n_samples "$N_SAMPLES" \
    --data_bsize "$BATCH_SIZE" \
    --seed "$SEED" \
    --gpu_id 0 \
    --train_episode "$TRAIN_EPISODE" \
    --num_collect 15 \
    --learning_epoch 10 \
    --lr_a 5e-4 \
    --clip_param 0.2 \
    --entropy_coef 0.01 \
    --save_candidates \
    --candidate_save_mode topk \
    --candidate_top_k "$TOP_K" \
    --candidate_dir "$CAND" \
    --run_id "$RUN_ID" \
    --use_dataset_growth \
    --dataset_initial_ratio 0.05 \
    --dataset_final_ratio 1.0 \
    --dataset_growth_start_episode 0 \
    --dataset_growth_end_episode "$TRAIN_EPISODE" \
    --eval_ppl_batch_size 8 \
    --enable_downstream=false \
    2>&1 | tee "$SEARCH/search.log"
fi

echo "===== TOP SEARCH CANDIDATES ====="
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CAND"]) / "candidates.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
rows = sorted(rows, key=lambda row: float(row.get("endpoint_ppl", 1e99)))
for row in rows[:20]:
    print(
        f"step={row.get('step')} "
        f"ppl={float(row.get('endpoint_ppl')):.4f} "
        f"logppl={float(row.get('endpoint_logppl')):.4f} "
        f"id={row['candidate_id']}"
    )
PY

run_probe() {
  local name="$1"
  local center="$2"
  local delta="$3"
  local preserve="$4"
  local out="$5"

  mkdir -p "$out/shards"
  echo "===== PROBE $name center=$center delta=$delta preserve=$preserve ====="

  CENTER="$center" DELTA_RUN="$delta" PRESERVE_RUN="$preserve" OUT="$out" "$PYTHON_BIN" - <<'PY'
import os
import pathlib
import subprocess
import sys

py = os.environ["PYTHON_BIN"]
gpus = os.environ["GPU_IDS"].split()
out = pathlib.Path(os.environ["OUT"])
center = os.environ["CENTER"]
delta = os.environ["DELTA_RUN"]
preserve = os.environ["PRESERVE_RUN"]
procs = []

for shard_id, gpu in enumerate(gpus):
    shard = out / "shards" / f"shard_{shard_id}"
    shard.mkdir(parents=True, exist_ok=True)
    log = shard / "probe.log"
    cmd = [
        py, "-u", "amc_searchPPO.py",
        "--job=probe",
        f"--model={os.environ['MODEL']}",
        f"--model_name={os.environ['MODEL_NAME']}",
        "--dataset_name=wikitext2",
        f"--preserve_ratio={preserve}",
        "--structure",
        "--prune=para",
        "--lbound=0.1",
        "--rbound=1.0",
        f"--n_samples={os.environ['N_SAMPLES']}",
        f"--data_bsize={os.environ['BATCH_SIZE']}",
        f"--candidate_dir={os.environ['CAND']}",
        f"--candidate_top_k={os.environ['TOP_K']}",
        f"--probe_sparsity={center}",
        f"--ew_delta={delta}",
        "--projection_mode=nested_from_base",
        "--projection_base_sparsity=0.30",
        f"--probe_output={shard / 'probe_results.csv'}",
        f"--probe_jsonl_output={shard / 'probe_results.jsonl'}",
        f"--num_shards={len(gpus)}",
        f"--shard_id={shard_id}",
        "--gpu_id=0",
        f"--seed={os.environ['SEED']}",
        "--use_dataset_growth",
        "--dataset_initial_ratio=0.05",
        "--dataset_final_ratio=0.05",
        "--dataset_growth_start_episode=0",
        "--dataset_growth_end_episode=1",
        "--enable_downstream=false",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    handle = open(log, "w")
    proc = subprocess.Popen(cmd, cwd="/workspace/structure_pruning", env=env, stdout=handle, stderr=subprocess.STDOUT)
    procs.append((proc, handle, log, gpu, shard_id))
    print(f"started shard={shard_id} gpu={gpu} log={log}", flush=True)

failed = False
for proc, handle, log, gpu, shard_id in procs:
    rc = proc.wait()
    handle.close()
    if rc != 0:
        failed = True
        print(f"FAILED shard={shard_id} gpu={gpu} log={log}", flush=True)

raise SystemExit(1 if failed else 0)
PY

  "$PYTHON_BIN" scripts/merge_ew_probe_results.py \
    --output_dir "$out" \
    "$out"/shards/shard_*/probe_results.csv
}

echo "===== PHASE 2: PATH 30/35/40 ====="
run_probe path40 0.35 0.05 0.650000 "$PATH40"

echo "===== PHASE 3: LOCAL RADIUS ====="
run_probe delta0025 0.30125 0.00125 0.700000 "$LOCAL/delta0025"
run_probe delta0050 0.30250 0.00250 0.700000 "$LOCAL/delta0050"
run_probe delta0100 0.30500 0.00500 0.700000 "$LOCAL/delta0100"

echo "===== PHASE 4: BUILD LOCAL/PATH TABLES ====="
"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

local = Path(os.environ["LOCAL"])
path40 = Path(os.environ["PATH40"])
down = Path(os.environ["DOWN"])
down.mkdir(parents=True, exist_ok=True)

def read(path):
    return {row["candidate_id"]: row for row in csv.DictReader(open(path))}

r0025 = read(local / "delta0025" / "probe_results.csv")
r0050 = read(local / "delta0050" / "probe_results.csv")
r0100 = read(local / "delta0100" / "probe_results.csv")
r40 = read(path40 / "probe_results.csv")
ids = sorted(set(r0025) & set(r0050) & set(r0100) & set(r40))

rows = []
for cid in ids:
    a, b, c, h = r0025[cid], r0050[cid], r0100[cid], r40[cid]
    l30, l35, l40 = float(h["ell_minus"]), float(h["ell_0"]), float(h["ell_plus"])
    l3025, l3050, l31 = float(a["ell_plus"]), float(b["ell_plus"]), float(c["ell_plus"])
    rows.append({
        "candidate_id": cid,
        "step": h.get("step", ""),
        "L30": l30,
        "PPL30": float(h["ppl_minus"]),
        "L3025": l3025,
        "S3025": l3025 - l30,
        "L3050": l3050,
        "S3050": l3050 - l30,
        "L31": l31,
        "S31": l31 - l30,
        "L35": l35,
        "S35": l35 - l30,
        "L40": l40,
        "PPL40": float(h["ppl_plus"]),
        "Delta40": l40 - l30,
    })

best40 = min(row["L40"] for row in rows)
for row in rows:
    row["Regret40"] = row["L40"] - best40

out_csv = local / "local_radius_vs_40_fixed5.csv"
with open(out_csv, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["L40"]))

picked = {}

def add(tag, row):
    current = picked.get(row["candidate_id"])
    if current is not None:
        tags = set(current.get("selection_tags", "").split(","))
        tags.add(tag)
        current["selection_tags"] = ",".join(sorted(item for item in tags if item))
        return
    copied = dict(row)
    copied["selection_tags"] = tag
    picked[row["candidate_id"]] = copied

for row in sorted(rows, key=lambda item: item["L30"])[:15]:
    add("top15_by_L30", row)
for tag, row in [
    ("FF_Endpoint", min(rows, key=lambda item: item["L30"])),
    ("PAS_S3025", min(rows, key=lambda item: item["S3025"])),
    ("PAS_S3050", min(rows, key=lambda item: item["S3050"])),
    ("PAS_S31", min(rows, key=lambda item: item["S31"])),
    ("PAS_S35", min(rows, key=lambda item: item["S35"])),
    ("Oracle40_analysis_only", min(rows, key=lambda item: item["L40"])),
]:
    add(tag, row)

subset = down / "downstream30_local_radius_subset_table.csv"
fields = [
    "candidate_id", "selection_tags", "step",
    "L30", "PPL30", "S3025", "S3050", "S31", "S35",
    "L40", "PPL40", "Regret40",
]
with open(subset, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in sorted(picked.values(), key=lambda item: item["L30"]):
        writer.writerow({key: row.get(key, "") for key in fields})

print("WROTE", out_csv)
print("WROTE", subset)
print("downstream candidate_count", len(picked))
PY

echo "===== PHASE 5: DOWNSTREAM@30 ====="
PYTHON="$PYTHON_BIN" RUN_DOWNSTREAM_NOW=true bash scripts/pas_run_downstream_batch.sh \
  --model "$MODEL" \
  --model-name "$MODEL_NAME" \
  --dataset wikitext2 \
  --seed "$SEED" \
  --candidate-pool "$CAND" \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --candidate-table "$DOWN/downstream30_local_radius_subset_table.csv" \
  --output-dir "$DOWN" \
  --gpu-ids "$GPU_IDS" \
  --tasks "$TASKS" \
  --limit "$DOWNSTREAM_LIMIT" \
  --batch-size "$DOWNSTREAM_BATCH_SIZE" \
  --eval-num-samples 64 \
  --recovery-method no_recovery

echo "===== PHASE 6: COLLECT DOWNSTREAM ====="
"$PYTHON_BIN" scripts/pas_collect_downstream_results.py \
  --model "$MODEL" \
  --dataset wikitext2 \
  --seed "$SEED" \
  --candidate-pool "$CAND" \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --candidate-table "$DOWN/downstream30_local_radius_subset_table.csv" \
  --downstream-dir "$DOWN" \
  --output-dir "$DOWN"

echo "===== PHASE 7: FINAL SUMMARY ====="
"$PYTHON_BIN" - <<'PY'
import csv
import math
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
down = Path(os.environ["DOWN"])
seed = os.environ["SEED"]
summary_path = down / f"downstream_candidate_summary_opt13b_seed{seed}.csv"

subset = {row["candidate_id"]: row for row in csv.DictReader(open(down / "downstream30_local_radius_subset_table.csv"))}
summary = {row["candidate_id"]: row for row in csv.DictReader(open(summary_path))}
rows = []
for cid, meta in subset.items():
    if cid not in summary:
        continue
    row = dict(meta)
    row["avg_pruned_score"] = summary[cid].get("avg_pruned_score", "")
    row["task_count"] = summary[cid].get("task_count", "")
    rows.append(row)

def num(value):
    try:
        return float(value)
    except Exception:
        return float("nan")

def ok(value):
    return math.isfinite(value)

def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if ok(x) and ok(y)]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)

def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out

def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if ok(x) and ok(y)]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    return pearson(ranks(list(xs)), ranks(list(ys)))

def residualize(y, x):
    pairs = [(i, yi, xi) for i, (yi, xi) in enumerate(zip(y, x)) if ok(yi) and ok(xi)]
    out = [float("nan")] * len(y)
    if len(pairs) < 3:
        return out
    xs = [item[2] for item in pairs]
    ys = [item[1] for item in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((xi - mx) ** 2 for xi in xs)
    if vx <= 0:
        return out
    beta = sum((xi - mx) * (yi - my) for yi, xi in zip(ys, xs)) / vx
    alpha = my - beta * mx
    for i, yi, xi in pairs:
        out[i] = yi - (alpha + beta * xi)
    return out

def fmt(value):
    if isinstance(value, float):
        return "" if not ok(value) else f"{value:.6g}"
    return str(value)

metrics = ["S3025", "S3050", "S31", "S35"]
y = [num(row["avg_pruned_score"]) for row in rows]
l30 = [num(row["L30"]) for row in rows]
yres = residualize(y, l30)
corr = []
for metric in metrics:
    xs = [num(row[metric]) for row in rows]
    xres = residualize(xs, l30)
    corr.append((metric, pearson(xs, y), spearman(xs, y), pearson(xres, yres)))

selections = [
    ("FF-Endpoint", min(rows, key=lambda row: num(row["L30"]))),
    ("PAS-S3025", min(rows, key=lambda row: num(row["S3025"]))),
    ("PAS-S3050", min(rows, key=lambda row: num(row["S3050"]))),
    ("PAS-S31", min(rows, key=lambda row: num(row["S31"]))),
    ("PAS-S35", min(rows, key=lambda row: num(row["S35"]))),
    ("Oracle-Downstream30", max(rows, key=lambda row: num(row["avg_pruned_score"]))),
    ("Oracle40", min(rows, key=lambda row: num(row["L40"]))),
]

joined = down / "local_radius_downstream30_joined.csv"
with open(joined, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: num(row["avg_pruned_score"]), reverse=True))

md = root / "FINAL_SUMMARY.md"
with open(md, "w") as handle:
    handle.write(f"# OPT-1.3B Seed{seed} Replicate Summary\n\n")
    handle.write("## Correlation With Downstream@30\n\n")
    handle.write("| metric | pearson | spearman | partial_corr(metric, downstream | L30) |\n")
    handle.write("| --- | --- | --- | --- |\n")
    for metric, p, s, pc in corr:
        handle.write(f"| {metric} | {fmt(p)} | {fmt(s)} | {fmt(pc)} |\n")
    handle.write("\n## Selection Summary\n\n")
    handle.write("| rule | step | avg_pruned_score | L30 | S3025 | S3050 | S31 | S35 | L40 | Regret40 | candidate |\n")
    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for rule, row in selections:
        handle.write(
            f"| {rule} | {row['step']} | {fmt(num(row['avg_pruned_score']))} | {fmt(num(row['L30']))} | "
            f"{fmt(num(row['S3025']))} | {fmt(num(row['S3050']))} | {fmt(num(row['S31']))} | "
            f"{fmt(num(row['S35']))} | {fmt(num(row['L40']))} | {fmt(num(row['Regret40']))} | {row['candidate_id']} |\n"
        )
    handle.write("\n## Top Candidates By Downstream\n\n")
    handle.write("| rank | step | avg_pruned_score | L30 | S3025 | S3050 | S31 | S35 | L40 | Regret40 | candidate |\n")
    handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for i, row in enumerate(sorted(rows, key=lambda item: num(item["avg_pruned_score"]), reverse=True)[:20], 1):
        handle.write(
            f"| {i} | {row['step']} | {fmt(num(row['avg_pruned_score']))} | {fmt(num(row['L30']))} | "
            f"{fmt(num(row['S3025']))} | {fmt(num(row['S3050']))} | {fmt(num(row['S31']))} | "
            f"{fmt(num(row['S35']))} | {fmt(num(row['L40']))} | {fmt(num(row['Regret40']))} | {row['candidate_id']} |\n"
        )

print("WROTE", md)
print(md.read_text())
PY

echo "===== ALL DONE ====="
echo "$ROOT/FINAL_SUMMARY.md"
