#!/usr/bin/env python3
"""Collect same-protocol recovery results for PAS stress subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect recovery table and analysis.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--target-sigma", type=float, default=0.30)
    parser.add_argument("--probe-sigma", type=float, default=0.35)
    parser.add_argument("--heldout-sigma", type=float, default=0.40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recovery-subset", required=True)
    parser.add_argument("--recovery-dir", required=True)
    parser.add_argument("--recovery-method", default="ridge_reconstruction")
    parser.add_argument("--recovery-steps", default="")
    parser.add_argument("--lora-rank", default="")
    parser.add_argument("--lora-alpha", default="")
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--recovery-seed", default="")
    return parser.parse_args()


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


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def f(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs)
    y = np.asarray(ys)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: list[float]) -> list[float]:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    out[order] = np.arange(1, len(values) + 1, dtype=float)
    return out.tolist()


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def regression(rows: list[dict[str, object]]) -> dict[str, float]:
    data = []
    for row in rows:
        y = f(row.get("L30_recovered"))
        l30 = f(row.get("L30_raw"))
        s35 = f(row.get("S35"))
        if y is not None and l30 is not None and s35 is not None:
            data.append((y, l30, s35))
    if len(data) < 3:
        return {"n": len(data), "beta_L30_raw": float("nan"), "beta_S35": float("nan"), "r2": float("nan")}
    y = np.asarray([item[0] for item in data])
    design = np.column_stack([np.ones(len(data)), [item[1] for item in data], [item[2] for item in data]])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"n": len(data), "beta_L30_raw": float(beta[1]), "beta_S35": float(beta[2]), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        print(f"Would collect recovery metadata from {args.recovery_dir}")
        return 0

    subset_rows = read_csv(Path(args.recovery_subset))
    rows: list[dict[str, object]] = []
    for subset in subset_rows:
        cid = subset["candidate_id"]
        run_dir = Path(args.recovery_dir) / "recovery_eval" / safe_id(cid)
        metadata_path = run_dir / "checkpoint.pth.json"
        if not metadata_path.exists():
            rows.append({"candidate_id": cid, "selection_tags": subset.get("selection_tags", ""), "notes": f"missing_metadata:{metadata_path}"})
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        ppl = float(metadata["ppl"])
        ell = math.log(ppl) if ppl > 0 else float("inf")
        raw = f(subset.get("L30_raw"))
        rows.append(
            {
                "model": Path(args.model).name,
                "dataset": args.dataset,
                "seed": args.seed,
                "candidate_pool_id": subset.get("candidate_pool_id", ""),
                "candidate_id": cid,
                "selection_tags": subset.get("selection_tags", ""),
                "L30_raw": subset.get("L30_raw", ""),
                "PPL30_raw": subset.get("PPL30_raw", ""),
                "S35": subset.get("S35", ""),
                "L40_raw": subset.get("L40_raw", ""),
                "Regret40": subset.get("Regret40", ""),
                "recovery_method": args.recovery_method,
                "recovery_steps": args.recovery_steps,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "recovery_seed": args.recovery_seed or args.seed,
                "L30_recovered": ell,
                "PPL30_recovered": ppl,
                "RecoveryGain": raw - ell if raw is not None else "",
                "artifact_path": str(metadata_path),
                "notes": "same_protocol_recovery",
            }
        )

    table_path = output_dir / f"recovery_table_opt27b_seed{args.seed}.csv"
    write_csv(table_path, rows)

    analysis_rows: list[dict[str, object]] = []
    for y_key in ("L30_recovered", "RecoveryGain"):
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x = f(row.get("S35"))
            y = f(row.get(y_key))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        analysis_rows.append({"metric": f"Pearson(S35,{y_key})", "value": pearson(xs, ys), "n": len(xs)})
        analysis_rows.append({"metric": f"Spearman(S35,{y_key})", "value": spearman(xs, ys), "n": len(xs)})
    analysis_rows.append({"metric": "linear_regression:L30_recovered~L30_raw+S35", **regression(rows)})
    analysis_path = output_dir / f"recovery_analysis_opt27b_seed{args.seed}.csv"
    write_csv(analysis_path, analysis_rows)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "recovery_method": args.recovery_method,
        "same_protocol_for_all_candidates": True,
        "artifacts": {"recovery_table": str(table_path), "recovery_analysis": str(analysis_path)},
    }
    manifest_path = output_dir / "recovery_manifest_opt27b.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {table_path}")
    print(f"Wrote {analysis_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
