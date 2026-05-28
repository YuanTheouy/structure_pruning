#!/usr/bin/env python3
"""Offline replay for Progressive PAS lookahead.

The script does not change PPO rewards. It replays saved FastForward candidates
from a progressive search, probes stage -> next-stage nested projections, and
compares a gated PAS lookahead selector with the stage endpoint selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def as_float(value, default=float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_num(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_gpu_ids(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def safe_float_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def candidate_path(candidate_dir: Path) -> Path:
    all_path = candidate_dir / "all_candidates.jsonl"
    if all_path.exists():
        return all_path
    return candidate_dir / "candidates.jsonl"


def candidate_key(candidate_id: str, sparsity: float, projection_mode: str, base_sparsity: float, n_samples: int) -> str:
    return "|".join(
        [
            candidate_id,
            f"{sparsity:.6f}",
            projection_mode,
            f"{base_sparsity:.6f}",
            str(n_samples),
        ]
    )


def merge_probe_csvs(paths: list[Path], output_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        for row in read_csv(path):
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not rows:
        raise RuntimeError(f"No probe rows found for {output_path}")
    rows.sort(key=lambda row: (as_int(row.get("shard_id")), row.get("candidate_id", "")))
    write_csv(output_path, rows, fieldnames)
    return rows


def run_probe_batch(
    *,
    args: argparse.Namespace,
    label: str,
    candidates: list[dict],
    center: float,
    delta: float,
    base_sparsity: float,
    output_dir: Path,
) -> list[dict[str, str]]:
    output_csv = output_dir / "probe_results.csv"
    existing = read_csv(output_csv)
    existing_ids = {row.get("candidate_id") for row in existing}
    candidate_ids = {row.get("candidate_id") for row in candidates}
    if existing and candidate_ids.issubset(existing_ids):
        print(f"=> reuse {label}: {output_csv}")
        return [row for row in existing if row.get("candidate_id") in candidate_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input_candidates"
    write_jsonl(input_dir / "candidates.jsonl", candidates)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        gpu_ids = ["0"]
    procs = []
    shard_csvs = []
    for shard_id, gpu in enumerate(gpu_ids):
        shard = output_dir / "shards" / f"shard_{shard_id}"
        shard.mkdir(parents=True, exist_ok=True)
        shard_csv = shard / "probe_results.csv"
        shard_csvs.append(shard_csv)
        log_path = shard / "probe.log"
        cmd = [
            sys.executable,
            "-u",
            str(repo_root() / "amc_searchPPO.py"),
            "--job=probe",
            f"--model={args.model}",
            f"--model_name={args.model_name}",
            f"--dataset_name={args.dataset}",
            f"--preserve_ratio={1.0 - center:.6f}",
            "--structure",
            "--prune=para",
            "--lbound=0.1",
            "--rbound=1.0",
            f"--n_samples={args.n_samples}",
            f"--data_bsize={args.batch_size}",
            f"--candidate_dir={input_dir}",
            "--candidate_top_k=0",
            f"--probe_sparsity={center}",
            f"--ew_delta={delta}",
            f"--projection_mode={args.projection_mode}",
            f"--projection_base_sparsity={base_sparsity}",
            f"--probe_output={shard_csv}",
            f"--probe_jsonl_output={shard / 'probe_results.jsonl'}",
            f"--num_shards={len(gpu_ids)}",
            f"--shard_id={shard_id}",
            "--gpu_id=0",
            f"--seed={args.seed}",
            "--enable_downstream=false",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
        log_handle = log_path.open("a", encoding="utf-8")
        print(f"=> probe {label}: shard={shard_id} gpu={gpu} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=repo_root(), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        procs.append((proc, log_handle, log_path))

    failed = False
    for proc, handle, log_path in procs:
        rc = proc.wait()
        handle.close()
        if rc != 0:
            failed = True
            print(f"FAILED probe shard log={log_path}", file=sys.stderr)
    if failed:
        raise RuntimeError(f"At least one probe shard failed for {label}")

    return merge_probe_csvs(shard_csvs, output_csv)


def append_cache(cache_rows: dict[str, dict], probe_rows: list[dict[str, str]], projection_mode: str, base_sparsity: float, n_samples: int) -> None:
    points = [
        ("minus", "sparsity_minus", "logppl_minus", "ppl_minus"),
        ("zero", "sparsity_zero", "logppl_zero", "ppl_zero"),
        ("plus", "sparsity_plus", "logppl_plus", "ppl_plus"),
    ]
    for row in probe_rows:
        cid = row.get("candidate_id", "")
        for tag, sparsity_key, log_key, ppl_key in points:
            sparsity = as_float(row.get(sparsity_key))
            if not math.isfinite(sparsity):
                continue
            key = candidate_key(cid, sparsity, projection_mode, base_sparsity, n_samples)
            cache_rows[key] = {
                "cache_key": key,
                "candidate_id": cid,
                "sparsity": f"{sparsity:.6f}",
                "projection_mode": projection_mode,
                "base_sparsity": f"{base_sparsity:.6f}",
                "n_samples": n_samples,
                "point": tag,
                "L": row.get(log_key, ""),
                "PPL": row.get(ppl_key, ""),
            }


def probe_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows if row.get("candidate_id")}


def choose_min(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    return min(rows, key=lambda row: (as_float(row.get(key)), as_float(row.get("logppl_zero")), row.get("candidate_id", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Progressive PAS lookahead replay")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", "--model_name", dest="model_name", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--target-sparsity", type=float, default=0.30)
    parser.add_argument("--heldout-sparsity", type=float, default=0.40)
    parser.add_argument("--stages", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--prefix-steps", default="300,500,700,1000,1500,2000,5000")
    parser.add_argument("--stage-window", type=float, default=0.015)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--projection-mode", default="nested_from_base", choices=["nested_from_base", "current"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval-l40", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_dir = Path(args.candidate_dir)
    source_path = candidate_path(candidate_dir)
    candidates = read_jsonl(source_path)
    if not candidates:
        raise RuntimeError(f"No candidates found under {candidate_dir}")
    expected_prefix = f"{args.model_name}_seed{args.seed}"
    bad_ids = [
        row.get("candidate_id", "")
        for row in candidates[: min(20, len(candidates))]
        if expected_prefix not in row.get("candidate_id", "")
    ]
    if bad_ids:
        raise RuntimeError(
            "candidate pool mismatch: expected candidate_id containing "
            f"{expected_prefix!r}, saw {bad_ids[:5]!r}; source={source_path}"
        )
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    stages = parse_float_list(args.stages)
    prefixes = parse_int_list(args.prefix_steps)
    if len(stages) < 2:
        raise RuntimeError("--stages must contain at least two values")

    cache_path = out_dir / "progressive_pas_eval_cache.csv"
    cache_rows = {row["cache_key"]: row for row in read_csv(cache_path) if row.get("cache_key")}
    selection_rows: list[dict] = []
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "candidate_source": str(source_path),
        "stage_expansions": [],
        "skipped": [],
        "regret40_note": "Regret40 is computed against the best L40 among the selected rules for each prefix/stage.",
    }

    for prefix in prefixes:
        prefix_candidates = [row for row in candidates if as_int(row.get("step")) <= prefix]
        search_endpoint_evals = len(prefix_candidates)
        for stage, next_stage in zip(stages[:-1], stages[1:]):
            window = args.stage_window
            stage_candidates = [
                row for row in prefix_candidates
                if abs(as_float(row.get("current_sparsity")) - stage) <= window
            ]
            expanded = False
            if len(stage_candidates) < args.top_k:
                window = 2.0 * args.stage_window
                expanded = True
                stage_candidates = [
                    row for row in prefix_candidates
                    if abs(as_float(row.get("current_sparsity")) - stage) <= window
                ]
                manifest["stage_expansions"].append(
                    {"prefix_step": prefix, "stage": stage, "expanded_window": window, "count": len(stage_candidates)}
                )
            if not stage_candidates:
                manifest["skipped"].append({"prefix_step": prefix, "stage": stage, "reason": "no candidates"})
                continue

            stage_candidates.sort(key=lambda row: (as_float(row.get("endpoint_logppl")), as_int(row.get("step")), row.get("candidate_id", "")))
            top_candidates = stage_candidates[: args.top_k]
            label = f"prefix{prefix}_stage{safe_float_label(stage)}_to_{safe_float_label(next_stage)}"
            probe_dir = out_dir / "probes" / label
            stage_rows = run_probe_batch(
                args=args,
                label=label,
                candidates=top_candidates,
                center=stage,
                delta=next_stage - stage,
                base_sparsity=stage,
                output_dir=probe_dir,
            )
            append_cache(cache_rows, stage_rows, args.projection_mode, stage, args.n_samples)

            endpoint = choose_min(stage_rows, "logppl_zero")
            oracle = choose_min(stage_rows, "logppl_plus")
            endpoint_L = as_float(endpoint.get("logppl_zero"))
            band = [row for row in stage_rows if as_float(row.get("logppl_zero")) <= endpoint_L + args.epsilon]
            if not band:
                band = [endpoint]
            pas_raw = choose_min(band, "logppl_plus")
            endpoint_price = as_float(pas_raw.get("logppl_zero")) - endpoint_L
            lookahead_gain = as_float(endpoint.get("logppl_plus")) - as_float(pas_raw.get("logppl_plus"))
            gate_passed = endpoint_price <= args.epsilon and lookahead_gain >= args.margin
            pas_selected = pas_raw if gate_passed else endpoint

            rule_items = [
                ("FF-stage-endpoint", endpoint, False),
                ("PAS-lookahead", pas_selected, gate_passed),
                ("Oracle-next-stage", oracle, False),
            ]
            selected_candidates = []
            seen_ids = set()
            for _, row, _ in rule_items:
                cid = row["candidate_id"]
                if cid not in seen_ids:
                    selected_candidates.append(candidates_by_id[cid])
                    seen_ids.add(cid)

            final_label = f"{label}_final30_40"
            final_rows = run_probe_batch(
                args=args,
                label=final_label,
                candidates=selected_candidates,
                center=args.target_sparsity,
                delta=args.heldout_sparsity - args.target_sparsity,
                base_sparsity=stage,
                output_dir=out_dir / "final_probes" / final_label,
            )
            append_cache(cache_rows, final_rows, args.projection_mode, stage, args.n_samples)
            final_by_id = probe_by_id(final_rows)
            finite_l40 = [as_float(row.get("logppl_plus")) for row in final_rows if is_num(row.get("logppl_plus"))]
            best_l40 = min(finite_l40) if finite_l40 else float("nan")

            extra_probe_evals = 3 * (len(top_candidates) + len(selected_candidates))
            for rule, row, rule_gate_passed in rule_items:
                cid = row["candidate_id"]
                final = final_by_id.get(cid, {})
                l40 = as_float(final.get("logppl_plus"))
                cand = candidates_by_id.get(cid, {})
                selection_rows.append(
                    {
                        "model": args.model_name,
                        "seed": cand.get("seed", args.seed),
                        "prefix_step": prefix,
                        "stage": stage,
                        "next_stage": next_stage,
                        "rule": rule,
                        "candidate_id": cid,
                        "source_step": cand.get("step", row.get("step", "")),
                        "source_current_sparsity": cand.get("current_sparsity", ""),
                        "L_stage": row.get("logppl_zero", ""),
                        "L_next": row.get("logppl_plus", ""),
                        "S_stage_to_next": as_float(row.get("logppl_plus")) - as_float(row.get("logppl_zero")),
                        "L30": final.get("logppl_zero", ""),
                        "PPL30": final.get("ppl_zero", ""),
                        "L40": final.get("logppl_plus", ""),
                        "PPL40": final.get("ppl_plus", ""),
                        "Regret40": l40 - best_l40 if math.isfinite(l40) and math.isfinite(best_l40) else "",
                        "endpoint_price": endpoint_price if rule == "PAS-lookahead" else "",
                        "lookahead_gain": lookahead_gain if rule == "PAS-lookahead" else "",
                        "gate_passed": rule_gate_passed if rule == "PAS-lookahead" else "",
                        "search_endpoint_evals": search_endpoint_evals,
                        "extra_probe_evals": extra_probe_evals,
                        "total_eval_budget": search_endpoint_evals + extra_probe_evals,
                        "stage_window": window,
                        "stage_window_expanded": expanded,
                    }
                )

            write_csv(cache_path, list(cache_rows.values()))
            write_csv(out_dir / "progressive_pas_selection.csv", selection_rows)

    write_csv(cache_path, list(cache_rows.values()))
    selection_csv = out_dir / "progressive_pas_selection.csv"
    if not selection_rows:
        raise RuntimeError(
            "No selection rows generated. Check candidate current_sparsity values, "
            "stage windows, prefix steps, and candidate source."
        )
    write_csv(selection_csv, selection_rows)
    manifest_path = out_dir / "progressive_pas_manifest.json"
    write_json(manifest_path, manifest)

    md = out_dir / "progressive_pas_summary.md"
    with md.open("w", encoding="utf-8") as handle:
        handle.write("# Progressive PAS Lookahead Replay\n\n")
        handle.write(f"- model: `{args.model_name}`\n")
        handle.write(f"- candidate source: `{source_path}`\n")
        handle.write(f"- prefixes: `{args.prefix_steps}`\n")
        handle.write(f"- stages: `{args.stages}`\n")
        handle.write(f"- gpu ids: `{args.gpu_ids}`\n\n")
        handle.write("## Selection Rows\n\n")
        handle.write(f"- rows: `{len(selection_rows)}`\n")
        handle.write(f"- csv: `{selection_csv}`\n")
        handle.write(f"- cache: `{cache_path}`\n\n")
        handle.write("## Last-Stage Snapshot\n\n")
        handle.write("| prefix | rule | candidate | L30 | PPL30 | L40 | Regret40 | gate_passed |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        last_stage = stages[-2]
        for row in selection_rows:
            if abs(as_float(row["stage"]) - last_stage) > 1e-9:
                continue
            handle.write(
                f"| {row['prefix_step']} | {row['rule']} | {row['candidate_id']} | "
                f"{row['L30']} | {row['PPL30']} | {row['L40']} | {row['Regret40']} | {row['gate_passed']} |\n"
            )
    print(f"WROTE {selection_csv}")
    print(f"WROTE {cache_path}")
    print(f"WROTE {md}")
    print(f"WROTE {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
