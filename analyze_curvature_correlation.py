#!/usr/bin/env python3
"""Measure Early-Warning signal correlation with a separated future target."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze curvature/future-degradation correlation.")
    parser.add_argument("--probe_results", required=True, help="Path to probe_results.csv")
    parser.add_argument(
        "--future_results_csv",
        default=None,
        help=(
            "Optional high_sparsity_results.csv. When provided, the target is "
            "logPPL(future_sparsity) - logPPL(target_sparsity), separated from "
            "the local 0.25/0.30/0.35 probe."
        ),
    )
    parser.add_argument("--warning_sparsity", type=float, default=0.30)
    parser.add_argument(
        "--target_sparsity",
        type=float,
        default=None,
        help="Baseline sparsity for the future target. Defaults to --warning_sparsity.",
    )
    parser.add_argument("--future_sparsity", type=float, default=0.40)
    parser.add_argument("--sparsity_tolerance", type=float, default=1e-6)
    parser.add_argument("--collapse_threshold", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_csv", default=None, help="Output correlation summary CSV")
    parser.add_argument("--joined_output_csv", default=None, help="Output joined target rows CSV")
    parser.add_argument("--output_tex", default=None, help="Optional LaTeX table path")
    return parser.parse_args()


def fmt_sparsity(value):
    return f"{float(value):.2f}"


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value, *, field):
    if value in (None, ""):
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def row_logppl(row):
    if row.get("logppl") not in (None, ""):
        return float(row["logppl"])
    if row.get("ppl") not in (None, ""):
        return math.log(float(row["ppl"]))
    raise ValueError(f"Row has neither logppl nor ppl: {row}")


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


def probe_by_candidate(rows):
    result = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if candidate_id:
            result[candidate_id] = row
    return result


def load_separated_future_rows(probe_rows, future_rows, target_sparsity, future_sparsity, tolerance):
    probes = probe_by_candidate(probe_rows)
    future_by_candidate = {}

    for row in future_rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id or row.get("sparsity") in (None, ""):
            continue
        sparsity = float(row["sparsity"])
        if abs(sparsity - target_sparsity) <= tolerance:
            point = "target"
        elif abs(sparsity - future_sparsity) <= tolerance:
            point = "future"
        else:
            continue

        entry = future_by_candidate.setdefault(candidate_id, {"methods": set()})
        method = row.get("method") or row.get("mode")
        if method:
            entry["methods"].add(method)
        entry.setdefault(point, {"row": row, "logppl": row_logppl(row)})

    target_name = f"future_degradation_{fmt_sparsity(future_sparsity)}_minus_{fmt_sparsity(target_sparsity)}"
    joined = []
    missing_probe = 0
    missing_point = 0

    for candidate_id, entry in sorted(future_by_candidate.items()):
        probe = probes.get(candidate_id)
        if probe is None:
            missing_probe += 1
            continue
        if "target" not in entry or "future" not in entry:
            missing_point += 1
            continue

        target_logppl = float(entry["target"]["logppl"])
        future_logppl = float(entry["future"]["logppl"])
        local_probe_degradation = (
            probe.get("local_probe_degradation")
            or probe.get("probe_plus_minus_zero_degradation")
            or probe.get("future_degradation_probe_plus_minus_zero")
            or probe.get("future_degradation")
            or ""
        )
        joined.append(
            {
                "candidate_id": candidate_id,
                "methods": ";".join(sorted(entry["methods"])),
                "probe_center_sparsity": probe.get("center_sparsity", ""),
                "target_sparsity": fmt_sparsity(target_sparsity),
                "future_sparsity": fmt_sparsity(future_sparsity),
                "target_logppl": target_logppl,
                "future_logppl": future_logppl,
                target_name: future_logppl - target_logppl,
                "curvature": parse_float(probe.get("curvature"), field="curvature"),
                "slope": parse_float(probe.get("slope"), field="slope"),
                "endpoint_logppl": parse_float(probe.get("logppl_zero"), field="logppl_zero"),
                "probe_logppl_minus": probe.get("logppl_minus", ""),
                "probe_logppl_zero": probe.get("logppl_zero", ""),
                "probe_logppl_plus": probe.get("logppl_plus", ""),
                "local_probe_degradation": local_probe_degradation,
                "target_source_checkpoint": entry["target"]["row"].get("checkpoint_path", ""),
                "future_source_checkpoint": entry["future"]["row"].get("checkpoint_path", ""),
            }
        )

    print(
        "Joined separated future target rows: "
        f"{len(joined)} usable, {missing_probe} without probe row, {missing_point} without both sparsity points"
    )
    return target_name, joined


def load_local_probe_rows(probe_rows):
    if not probe_rows:
        return "local_probe_degradation", []

    first = probe_rows[0]
    zero = first.get("sparsity_zero") or first.get("center_sparsity") or ""
    plus = first.get("sparsity_plus") or ""
    if zero and plus:
        target_name = f"local_probe_degradation_{fmt_sparsity(plus)}_minus_{fmt_sparsity(zero)}"
    else:
        target_name = "local_probe_degradation_sigma_plus_delta_minus_sigma"

    print(
        "WARNING: --future_results_csv was not provided; using local probe degradation only, "
        "not the paper-level separated future target.",
        file=sys.stderr,
    )

    rows = []
    for probe in probe_rows:
        local_probe_degradation = (
            probe.get("local_probe_degradation")
            or probe.get("probe_plus_minus_zero_degradation")
            or probe.get("future_degradation_probe_plus_minus_zero")
            or probe.get("future_degradation")
        )
        rows.append(
            {
                "candidate_id": probe.get("candidate_id", ""),
                "methods": "",
                "probe_center_sparsity": probe.get("center_sparsity", ""),
                "target_sparsity": probe.get("sparsity_zero", ""),
                "future_sparsity": probe.get("sparsity_plus", ""),
                "target_logppl": probe.get("logppl_zero", ""),
                "future_logppl": probe.get("logppl_plus", ""),
                target_name: parse_float(local_probe_degradation, field="local_probe_degradation"),
                "curvature": parse_float(probe.get("curvature"), field="curvature"),
                "slope": parse_float(probe.get("slope"), field="slope"),
                "endpoint_logppl": parse_float(probe.get("logppl_zero"), field="logppl_zero"),
                "probe_logppl_minus": probe.get("logppl_minus", ""),
                "probe_logppl_zero": probe.get("logppl_zero", ""),
                "probe_logppl_plus": probe.get("logppl_plus", ""),
                "local_probe_degradation": local_probe_degradation,
                "target_source_checkpoint": "",
                "future_source_checkpoint": "",
            }
        )
    return target_name, rows


def build_correlation_rows(joined_rows, target_name, collapse_threshold):
    target = np.asarray([float(row[target_name]) for row in joined_rows], dtype=np.float64)
    metrics = {
        "curvature": np.asarray([float(row["curvature"]) for row in joined_rows], dtype=np.float64),
        "slope": np.asarray([float(row["slope"]) for row in joined_rows], dtype=np.float64),
        "endpoint_logppl": np.asarray([float(row["endpoint_logppl"]) for row in joined_rows], dtype=np.float64),
    }

    rows = []
    for name, values in metrics.items():
        row = {
            "metric": name,
            "target": target_name,
            "pearson": corr(values, target),
            "spearman": corr(rankdata(values), rankdata(target)),
            "n": len(target),
            "auroc": "",
        }
        if collapse_threshold is not None:
            try:
                from sklearn.metrics import roc_auc_score

                labels = (target >= collapse_threshold).astype(np.int32)
                if len(set(labels.tolist())) > 1:
                    row["auroc"] = float(roc_auc_score(labels, values))
            except Exception:
                row["auroc"] = ""
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrr}\n")
        handle.write("\\toprule\nMetric & Pearson & Spearman & AUROC & N \\\\\n\\midrule\n")
        for row in rows:
            auroc = row["auroc"] if row["auroc"] == "" else f"{row['auroc']:.3f}"
            handle.write(
                f"{row['metric']} & {row['pearson']:.3f} & {row['spearman']:.3f} & "
                f"{auroc} & {row['n']} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def write_scatter(output_dir, rows, target_name):
    try:
        import matplotlib.pyplot as plt

        curvature = np.asarray([float(row["curvature"]) for row in rows], dtype=np.float64)
        target = np.asarray([float(row[target_name]) for row in rows], dtype=np.float64)

        scatter_pdf = output_dir / "curvature_scatter.pdf"
        plt.figure(figsize=(5.2, 4.2))
        plt.scatter(curvature, target, s=36, alpha=0.85)
        if len(target) >= 2:
            fit = np.polyfit(curvature, target, deg=1)
            xs = np.linspace(float(np.min(curvature)), float(np.max(curvature)), 100)
            plt.plot(xs, fit[0] * xs + fit[1], color="black", linewidth=1.2)
        plt.xlabel("Log-PPL curvature")
        plt.ylabel(target_name.replace("_", " "))
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(scatter_pdf)
        print(f"Wrote {scatter_pdf}")
    except Exception as exc:
        print(f"Skipping curvature_scatter.pdf: {exc}")


def main():
    args = parse_args()
    probe_path = Path(args.probe_results)
    output_dir = Path(args.output_dir) if args.output_dir else probe_path.parent
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "correlation_table.csv"
    joined_output_csv = (
        Path(args.joined_output_csv) if args.joined_output_csv else output_dir / "correlation_joined.csv"
    )
    target_sparsity = args.target_sparsity if args.target_sparsity is not None else args.warning_sparsity

    probe_rows = read_csv_rows(probe_path)
    if args.future_results_csv:
        future_rows = read_csv_rows(Path(args.future_results_csv))
        target_name, joined_rows = load_separated_future_rows(
            probe_rows,
            future_rows,
            target_sparsity,
            args.future_sparsity,
            args.sparsity_tolerance,
        )
    else:
        target_name, joined_rows = load_local_probe_rows(probe_rows)

    if not joined_rows:
        raise RuntimeError("No joined rows available for correlation analysis.")

    correlation_rows = build_correlation_rows(joined_rows, target_name, args.collapse_threshold)
    write_csv(joined_output_csv, joined_rows)
    write_csv(output_csv, correlation_rows)

    output_tex = Path(args.output_tex) if args.output_tex else None
    if output_tex:
        write_tex(output_tex, correlation_rows)

    write_scatter(output_dir, joined_rows, target_name)
    print(f"Wrote {joined_output_csv}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
