#!/usr/bin/env python3
"""Make PAS stress/recovery evidence figures."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PAS stress/recovery figures.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--target-sigma", type=float, default=0.30)
    parser.add_argument("--probe-sigma", type=float, default=0.35)
    parser.add_argument("--heldout-sigma", type=float, default=0.40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stress-table", required=True)
    parser.add_argument("--recovery-table", default="")
    parser.add_argument("--downstream-table", default="")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


def scatter(rows, x_key, y_key, path: Path, xlabel: str, ylabel: str, title: str) -> None:
    xs = []
    ys = []
    colors = []
    for row in rows:
        x = f(row, x_key)
        y = f(row, y_key)
        c = f(row, "endpoint_rank") or f(row, "L30") or 0.0
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        colors.append(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.6, 4.0))
    plt.scatter(xs, ys, c=colors, cmap="viridis", alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label="endpoint rank / L30")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    if args.dry_run:
        print(f"Would write figures under {out}")
        return 0

    stress_rows = read_csv(args.stress_table)
    scatter(
        stress_rows,
        "S35",
        "Regret40",
        out / "img" / "stress_predicts_regret.pdf",
        "S35 = L35 - L30",
        "Regret40",
        "Local stress predicts held-out regret",
    )
    scatter(
        stress_rows,
        "S35",
        "Regret40",
        out / "img" / "sensitivity_correlation.pdf",
        "S35 = L35 - L30",
        "Regret40",
        "Sensitivity correlation",
    )

    if args.recovery_table and Path(args.recovery_table).exists():
        recovery_rows = read_csv(args.recovery_table)
        scatter(
            recovery_rows,
            "S35",
            "L30_recovered",
            out / "img" / "stress_predicts_recovery.pdf",
            "S35 = L35 - L30",
            "L30 recovered",
            "Local stress predicts recovery quality",
        )
        scatter(
            recovery_rows,
            "L30_raw",
            "L30_recovered",
            out / "img" / "endpoint_vs_recovered_ranking.pdf",
            "L30 raw",
            "L30 recovered",
            "Endpoint vs recovered quality",
        )

    if args.downstream_table and Path(args.downstream_table).exists():
        downstream_rows = read_csv(args.downstream_table)
        scatter(
            downstream_rows,
            "S35",
            "drop",
            out / "img" / "stress_predicts_downstream.pdf",
            "S35 = L35 - L30",
            "downstream drop",
            "Local stress predicts downstream drop",
        )

    print(f"Wrote figures under {out / 'img'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
