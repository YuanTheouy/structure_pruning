#!/usr/bin/env python3
"""Measure correlation between Early-Warning curvature and future degradation."""

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze curvature/future-degradation correlation.")
    parser.add_argument("--probe_results", required=True, help="Path to probe_results.csv")
    parser.add_argument("--warning_sparsity", type=float, default=0.30)
    parser.add_argument("--future_sparsity", type=float, default=0.40)
    parser.add_argument("--collapse_threshold", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_csv", default=None, help="Output correlation summary CSV")
    parser.add_argument("--output_tex", default=None, help="Optional LaTeX table path")
    return parser.parse_args()


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr(x, y):
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()
    probe_path = Path(args.probe_results)
    output_dir = Path(args.output_dir) if args.output_dir else probe_path.parent
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "correlation_table.csv"

    curvature = []
    future = []
    slope = []
    endpoint = []
    with open(probe_path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curvature.append(float(row["curvature"]))
            future.append(float(row["future_degradation"]))
            slope.append(float(row["slope"]))
            endpoint.append(float(row["logppl_zero"]))

    metrics = {
        "curvature": np.asarray(curvature, dtype=np.float64),
        "slope": np.asarray(slope, dtype=np.float64),
        "endpoint_logppl": np.asarray(endpoint, dtype=np.float64),
    }
    future = np.asarray(future, dtype=np.float64)

    rows = []
    for name, values in metrics.items():
        row = {
            "metric": name,
            "target": "future_degradation",
            "pearson": corr(values, future),
            "spearman": corr(rankdata(values), rankdata(future)),
            "n": len(future),
            "auroc": "",
        }
        if args.collapse_threshold is not None:
            try:
                from sklearn.metrics import roc_auc_score

                labels = (future >= args.collapse_threshold).astype(np.int32)
                if len(set(labels.tolist())) > 1:
                    row["auroc"] = float(roc_auc_score(labels, values))
            except Exception:
                row["auroc"] = ""
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_tex = Path(args.output_tex) if args.output_tex else None
    if output_tex:
        with open(output_tex, "w", encoding="utf-8") as handle:
            handle.write("\\begin{tabular}{lrrrr}\n")
            handle.write("\\toprule\nMetric & Pearson & Spearman & AUROC & N \\\\\n\\midrule\n")
            for row in rows:
                auroc = row["auroc"] if row["auroc"] == "" else f"{row['auroc']:.3f}"
                handle.write(f"{row['metric']} & {row['pearson']:.3f} & {row['spearman']:.3f} & {auroc} & {row['n']} \\\\\n")
            handle.write("\\bottomrule\n\\end{tabular}\n")

    try:
        import matplotlib.pyplot as plt

        scatter_pdf = output_dir / "curvature_scatter.pdf"
        plt.figure(figsize=(5.2, 4.2))
        plt.scatter(metrics["curvature"], future, s=36, alpha=0.85)
        if len(future) >= 2:
            fit = np.polyfit(metrics["curvature"], future, deg=1)
            xs = np.linspace(float(np.min(metrics["curvature"])), float(np.max(metrics["curvature"])), 100)
            plt.plot(xs, fit[0] * xs + fit[1], color="black", linewidth=1.2)
        plt.xlabel("Log-PPL curvature")
        plt.ylabel("Future degradation")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(scatter_pdf)
        print(f"Wrote {scatter_pdf}")
    except Exception as exc:
        print(f"Skipping curvature_scatter.pdf: {exc}")

    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
