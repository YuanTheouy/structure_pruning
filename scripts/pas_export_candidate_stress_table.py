#!/usr/bin/env python3
"""Export candidate-level PAS stress table from probe and held-out artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L30/L35/L40 candidate stress table.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--target-sigma", type=float, default=0.30)
    parser.add_argument("--probe-sigma", type=float, default=0.35)
    parser.add_argument("--heldout-sigma", type=float, default=0.40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-results", default=None)
    parser.add_argument("--heldout-results", default=None)
    parser.add_argument("--candidate-pool-id", default=None)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def model_tag(model_name: str) -> str:
    return model_name.replace("-", "").replace(".", "").replace("/", "_")


def sigma_dir(value: float) -> str:
    return f"sparsity_{value:.2f}"


def default_pas_dir(args: argparse.Namespace) -> Path:
    model_name = args.model_name or Path(args.model).name
    return Path("/workspace/ckpts") / model_name / sigma_dir(args.target_sigma) / f"p0_pas_seed{args.seed}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def as_float(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rank_map(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    usable = [row for row in rows if row.get(key) not in ("", None)]
    ordered = sorted(usable, key=lambda row: (float(row[key]), str(row["candidate_id"])))
    return {str(row["candidate_id"]): index for index, row in enumerate(ordered, start=1)}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pas_dir = default_pas_dir(args)
    probe_path = Path(args.probe_results) if args.probe_results else pas_dir / "probe_results.csv"
    heldout_path = Path(args.heldout_results) if args.heldout_results else pas_dir / "heldout_results.csv"
    model_name = args.model_name or Path(args.model).name
    pool_id = args.candidate_pool_id or f"{model_tag(model_name)}_seed{args.seed}"

    if args.dry_run:
        print(f"Would read probe results: {probe_path}")
        print(f"Would read held-out results: {heldout_path}")
        print(f"Would write stress table under: {output_dir}")
        return 0

    probe_rows = read_csv(probe_path)
    heldout_rows = read_csv(heldout_path)
    heldout_by_id = {row["candidate_id"]: row for row in heldout_rows}
    rows: list[dict[str, object]] = []
    for probe in probe_rows:
        candidate_id = probe.get("candidate_id", "")
        heldout = heldout_by_id.get(candidate_id, {})
        l30 = as_float(probe, "ell_0", as_float(probe, "logppl_zero"))
        l35 = as_float(probe, "ell_plus", as_float(probe, "logppl_plus"))
        l40 = as_float(heldout, "ell_h")
        if l30 is None or l35 is None or l40 is None:
            notes = "missing_required_loss"
        else:
            notes = "probe_target_and_local; heldout_40_analysis_only"
        row = {
            "model": model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "candidate_pool_id": pool_id,
            "candidate_id": candidate_id,
            "target_sigma": args.target_sigma,
            "local_probe_sigma": args.probe_sigma,
            "heldout_sigma": args.heldout_sigma,
            "L30": l30 if l30 is not None else "",
            "PPL30": as_float(probe, "ppl_0", as_float(probe, "ppl_zero", math.exp(l30) if l30 is not None else None)),
            "L35": l35 if l35 is not None else "",
            "PPL35": as_float(probe, "ppl_plus", math.exp(l35) if l35 is not None else None),
            "L40": l40 if l40 is not None else "",
            "PPL40": as_float(heldout, "ppl_h", math.exp(l40) if l40 is not None else None),
            "S35": (l35 - l30) if l30 is not None and l35 is not None else "",
            "Delta40": (l40 - l30) if l30 is not None and l40 is not None else "",
            "Regret40": "",
            "endpoint_rank": "",
            "sensitivity_rank": "",
            "heldout_rank": "",
            "artifact_path": f"{probe_path};{heldout_path}",
            "notes": notes,
        }
        rows.append(row)

    l40_values = [float(row["L40"]) for row in rows if row["L40"] not in ("", None)]
    best_l40 = min(l40_values) if l40_values else float("nan")
    for row in rows:
        if row["L40"] not in ("", None):
            row["Regret40"] = float(row["L40"]) - best_l40

    endpoint_ranks = rank_map(rows, "L30")
    sensitivity_ranks = rank_map(rows, "S35")
    heldout_ranks = rank_map(rows, "L40")
    for row in rows:
        candidate_id = str(row["candidate_id"])
        row["endpoint_rank"] = endpoint_ranks.get(candidate_id, "")
        row["sensitivity_rank"] = sensitivity_ranks.get(candidate_id, "")
        row["heldout_rank"] = heldout_ranks.get(candidate_id, "")

    out_csv = output_dir / f"candidate_stress_table_{model_tag(model_name)}_seed{args.seed}.csv"
    write_csv(out_csv, rows)
    manifest_path = output_dir / f"candidate_stress_manifest_{model_tag(model_name)}.json"
    write_json(
        manifest_path,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": args.model,
            "model_name": model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "candidate_pool": args.candidate_pool,
            "candidate_pool_id": pool_id,
            "target_sigma": args.target_sigma,
            "local_probe_sigma": args.probe_sigma,
            "heldout_sigma": args.heldout_sigma,
            "probe_results": str(probe_path),
            "heldout_results": str(heldout_path),
            "candidate_count": len(rows),
            "heldout_usage": "analysis_only_not_selection_tuning_filtering_or_early_stopping",
            "artifacts": {"candidate_stress_table": str(out_csv)},
        },
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
