#!/usr/bin/env python3
"""Analyze whether PAS local stress S35 predicts held-out regret."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze S35 correlations and regressions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", default="pooled")
    parser.add_argument("--candidate-pool", default="")
    parser.add_argument("--target-sigma", type=float, default=0.30)
    parser.add_argument("--probe-sigma", type=float, default=0.35)
    parser.add_argument("--heldout-sigma", type=float, default=0.40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stress-tables", nargs="+", required=True)
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


def to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def valid_arrays(rows: list[dict[str, str]], x_key: str, y_key: str, control_key: str | None = None):
    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    for row in rows:
        x = to_float(row, x_key)
        y = to_float(row, y_key)
        c = to_float(row, control_key) if control_key else 0.0
        if x is None or y is None or (control_key and c is None):
            continue
        xs.append(x)
        ys.append(y)
        cs.append(c)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(cs, dtype=float)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    out[order] = np.arange(1, len(values) + 1, dtype=float)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(ranks(x), ranks(y))


def residualize(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_corr(rows: list[dict[str, str]], x_key: str, y_key: str, control_key: str) -> float:
    x, y, c = valid_arrays(rows, x_key, y_key, control_key)
    if len(x) < 3:
        return float("nan")
    return pearson(residualize(x, c), residualize(y, c))


def regression(rows: list[dict[str, str]], y_key: str) -> dict[str, float]:
    s35, y, l30 = valid_arrays(rows, "S35", y_key, "L30")
    if len(y) < 3:
        return {"n": len(y), "intercept": float("nan"), "beta_L30": float("nan"), "beta_S35": float("nan"), "r2": float("nan")}
    design = np.column_stack([np.ones(len(y)), l30, s35])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": len(y),
        "intercept": float(beta[0]),
        "beta_L30": float(beta[1]),
        "beta_S35": float(beta[2]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def metric_rows(label: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for y_key in ("Regret40", "Delta40"):
        x, y, _ = valid_arrays(rows, "S35", y_key)
        out.append({"scope": label, "metric": f"Pearson(S35,{y_key})", "value": pearson(x, y), "n": len(x)})
        out.append({"scope": label, "metric": f"Spearman(S35,{y_key})", "value": spearman(x, y), "n": len(x)})
    out.append({"scope": label, "metric": "partial_corr(S35,Regret40|L30)", "value": partial_corr(rows, "S35", "Regret40", "L30"), "n": len(valid_arrays(rows, "S35", "Regret40", "L30")[0])})
    out.append({"scope": label, "metric": "partial_corr(S35,L40|L30)", "value": partial_corr(rows, "S35", "L40", "L30"), "n": len(valid_arrays(rows, "S35", "L40", "L30")[0])})
    for y_key in ("L40", "Regret40"):
        reg = regression(rows, y_key)
        out.append({"scope": label, "metric": f"linear_regression:{y_key}~L30+S35", **reg})
    return out


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PAS Stress Correlation Analysis\n\n")
        handle.write("| scope | metric | value | n | beta_L30 | beta_S35 | r2 |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row.get('scope','')} | {row.get('metric','')} | {row.get('value','')} | {row.get('n','')} | "
                f"{row.get('beta_L30','')} | {row.get('beta_S35','')} | {row.get('r2','')} |\n"
            )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        print("Would analyze stress tables:")
        for table in args.stress_tables:
            print(f"  {table}")
        return 0

    all_rows: list[dict[str, str]] = []
    result_rows: list[dict[str, object]] = []
    for table in args.stress_tables:
        rows = read_csv(Path(table))
        all_rows.extend(rows)
        result_rows.extend(metric_rows(Path(table).stem, rows))
    if len(args.stress_tables) > 1:
        result_rows.extend(metric_rows("pooled", all_rows))

    csv_path = output_dir / "stress_correlation_opt27b.csv"
    md_path = output_dir / "stress_correlation_opt27b.md"
    write_csv(csv_path, result_rows)
    write_md(md_path, result_rows)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "dataset": args.dataset,
        "target_sigma": args.target_sigma,
        "probe_sigma": args.probe_sigma,
        "heldout_sigma": args.heldout_sigma,
        "stress_tables": args.stress_tables,
        "artifacts": {"stress_correlation_csv": str(csv_path), "stress_correlation_md": str(md_path)},
    }
    with (output_dir / "stress_correlation_manifest_opt27b.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
