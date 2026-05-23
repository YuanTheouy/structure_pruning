#!/usr/bin/env python3
"""Analyze small-radius PAS local-delta probes against stress/downstream targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PAS local-delta probe correlations.")
    parser.add_argument("--local-delta-csv", required=True, help="CSV with L30/L3025/L3050/S3025/S3050 rows.")
    parser.add_argument("--stress-table", required=True, help="Candidate stress table with S35/L40/Regret40.")
    parser.add_argument("--downstream-summary", required=True, help="Candidate downstream@30 summary CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="opt-2.7b")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", default="3025")
    parser.add_argument("--endpoint-close-top-k", default="8,13")
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


def to_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def get_float(row: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = to_float(row.get(key))
            if value is not None:
                return value
    return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def residualize(values: list[float], control: list[float]) -> list[float] | None:
    if len(values) != len(control) or len(values) < 3:
        return None
    mc = sum(control) / len(control)
    mv = sum(values) / len(values)
    denom = sum((c - mc) ** 2 for c in control)
    if denom <= 0:
        return None
    beta = sum((c - mc) * (v - mv) for c, v in zip(control, values)) / denom
    alpha = mv - beta * mc
    return [v - (alpha + beta * c) for v, c in zip(values, control)]


def partial_corr(xs: list[float], ys: list[float], control: list[float]) -> float | None:
    rx = residualize(xs, control)
    ry = residualize(ys, control)
    if rx is None or ry is None:
        return None
    return pearson(rx, ry)


def regression(xs: list[float], ys: list[float], control: list[float]) -> dict[str, object]:
    if len(xs) < 3 or len(xs) != len(ys) or len(xs) != len(control):
        return {"intercept": "", "beta_L30": "", "beta_predictor": "", "r2": ""}
    # Two-feature least squares written directly to avoid adding scipy/sklearn.
    x1 = control
    x2 = xs
    cols = [
        [1.0] * len(ys),
        x1,
        x2,
    ]
    gram = [[sum(a * b for a, b in zip(c1, c2)) for c2 in cols] for c1 in cols]
    rhs = [sum(c * y for c, y in zip(col, ys)) for col in cols]
    beta = solve_3x3(gram, rhs)
    if beta is None:
        return {"intercept": "", "beta_L30": "", "beta_predictor": "", "r2": ""}
    pred = [beta[0] + beta[1] * c + beta[2] * x for c, x in zip(control, xs)]
    ybar = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    return {
        "intercept": beta[0],
        "beta_L30": beta[1],
        "beta_predictor": beta[2],
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else "",
    }


def solve_3x3(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    a = [row[:] + [b] for row, b in zip(matrix, rhs)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        a[col] = [v / div for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [v - factor * p for v, p in zip(a[row], a[col])]
    return [a[i][n] for i in range(n)]


def collect_arrays(
    rows: list[dict[str, object]],
    predictor: str,
    target: str,
    control: str,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    for row in rows:
        x = get_float(row, predictor)
        y = get_float(row, target)
        c = get_float(row, control)
        if x is None or y is None or c is None:
            continue
        xs.append(x)
        ys.append(y)
        cs.append(c)
    return xs, ys, cs


def metric_rows(rows: list[dict[str, object]], scope: str, control_key: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    predictors = [
        ("S3025", "local_30.25_minus_30"),
        ("S3050", "local_30.50_minus_30"),
        ("S31", "local_31.00_minus_30"),
        ("S35", "stress_35_minus_30"),
    ]
    targets = [
        ("avg_pruned_score", "higher_is_better", "local flatness story expects negative slope-score relation"),
        ("L40", "lower_is_better", "stress story expects positive slope-loss relation"),
        ("Regret40", "lower_is_better", "stress story expects positive slope-regret relation"),
        ("Delta40", "lower_is_better", "stress story expects positive slope-degradation relation"),
    ]
    for predictor, predictor_note in predictors:
        for target, direction, target_note in targets:
            xs, ys, cs = collect_arrays(rows, predictor, target, control_key)
            if len(xs) < 3:
                continue
            reg = regression(xs, ys, cs)
            out.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "target": target,
                    "control": control_key,
                    "n": len(xs),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                    "partial_corr": partial_corr(xs, ys, cs),
                    **reg,
                    "direction": direction,
                    "predictor_note": predictor_note,
                    "target_note": target_note,
                }
            )
    return out


def normalize_local(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {"candidate_id": row["candidate_id"]}
    for key in (
        "L30",
        "L3025",
        "L3050",
        "L31",
        "S3025",
        "S3050",
        "S31",
        "PPL30",
        "PPL3025",
        "PPL3050",
        "PPL31",
    ):
        out[f"{key}_local" if key == "L30" else key] = row.get(key, "")
    return out


def normalize_stress(row: dict[str, str]) -> dict[str, object]:
    l30 = row.get("L30", row.get("L30_raw", ""))
    l40 = row.get("L40", row.get("L40_raw", ""))
    delta40 = row.get("Delta40", "")
    if delta40 in ("", None) and l30 not in ("", None) and l40 not in ("", None):
        l30_f = to_float(l30)
        l40_f = to_float(l40)
        delta40 = l40_f - l30_f if l30_f is not None and l40_f is not None else ""
    return {
        "candidate_id": row["candidate_id"],
        "selection_tags": row.get("selection_tags", ""),
        "L30_stress": l30,
        "PPL30_raw": row.get("PPL30_raw", row.get("PPL30", "")),
        "S35": row.get("S35", ""),
        "L40": l40,
        "Regret40": row.get("Regret40", ""),
        "Delta40": delta40,
    }


def normalize_downstream(row: dict[str, str]) -> dict[str, object]:
    return {
        "candidate_id": row["candidate_id"],
        "checkpoint_type": row.get("checkpoint_type", ""),
        "task_count": row.get("task_count", ""),
        "avg_pruned_score": row.get("avg_pruned_score", ""),
        "avg_retention": row.get("avg_retention", ""),
        "avg_drop": row.get("avg_drop", ""),
    }


def write_md(path: Path, analysis_rows: list[dict[str, object]], joined_rows: list[dict[str, object]]) -> None:
    def fmt(value: object) -> str:
        number = to_float(value)
        if number is None:
            return str(value)
        return f"{number:.6g}"

    best_local_downstream = [
        row for row in analysis_rows
        if row.get("scope") == "all_candidates"
        and row.get("target") == "avg_pruned_score"
        and row.get("predictor") in {"S3025", "S3050", "S31"}
    ]
    stress_rows = [
        row for row in analysis_rows
        if row.get("scope") == "all_candidates"
        and row.get("target") in {"L40", "Regret40"}
        and row.get("predictor") == "S35"
    ]
    negative_counts = {
        key: sum(1 for row in joined_rows if (get_float(row, key) is not None and get_float(row, key) < 0))
        for key in ("S3025", "S3050", "S31")
        if any(get_float(row, key) is not None for row in joined_rows)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PAS Local-Delta Probe Analysis\n\n")
        handle.write("This analysis joins the small-radius PPL probe, the existing stress table, and the existing downstream@30 summary.\n\n")
        handle.write("## Quick Read\n\n")
        handle.write(f"- Joined candidates: `{len(joined_rows)}`.\n")
        negative_text = ", ".join(f"`{key}={value}`" for key, value in negative_counts.items())
        handle.write(f"- Negative local slopes: {negative_text}.\n")
        handle.write("- A local-flatness downstream story would need a stable negative relation between local slope and downstream@30 score after controlling endpoint `L30`.\n")
        handle.write("- A stress story expects a positive relation between slope and `L40`/`Regret40` after controlling endpoint `L30`.\n\n")
        handle.write("## All-Candidate Downstream Rows\n\n")
        handle.write("| predictor | target | control | n | pearson | spearman | partial_corr | beta_predictor | r2 |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in best_local_downstream:
            handle.write(
                f"| {row['predictor']} | {row['target']} | {row['control']} | {row['n']} | "
                f"{fmt(row.get('pearson'))} | {fmt(row.get('spearman'))} | {fmt(row.get('partial_corr'))} | "
                f"{fmt(row.get('beta_predictor'))} | {fmt(row.get('r2'))} |\n"
            )
        handle.write("\n## S35 Stress Rows\n\n")
        handle.write("| predictor | target | control | n | pearson | spearman | partial_corr | beta_predictor | r2 |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in stress_rows:
            handle.write(
                f"| {row['predictor']} | {row['target']} | {row['control']} | {row['n']} | "
                f"{fmt(row.get('pearson'))} | {fmt(row.get('spearman'))} | {fmt(row.get('partial_corr'))} | "
                f"{fmt(row.get('beta_predictor'))} | {fmt(row.get('r2'))} |\n"
            )
        handle.write("\n## Full Metric Table\n\n")
        handle.write("| scope | predictor | target | control | n | pearson | spearman | partial_corr | beta_predictor | r2 |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in analysis_rows:
            handle.write(
                f"| {row['scope']} | {row['predictor']} | {row['target']} | {row['control']} | {row['n']} | "
                f"{fmt(row.get('pearson'))} | {fmt(row.get('spearman'))} | {fmt(row.get('partial_corr'))} | "
                f"{fmt(row.get('beta_predictor'))} | {fmt(row.get('r2'))} |\n"
            )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    local_rows = {row["candidate_id"]: normalize_local(row) for row in read_csv(Path(args.local_delta_csv))}
    stress_rows = {row["candidate_id"]: normalize_stress(row) for row in read_csv(Path(args.stress_table))}
    downstream_rows = {row["candidate_id"]: normalize_downstream(row) for row in read_csv(Path(args.downstream_summary))}
    common_ids = sorted(set(local_rows) & set(stress_rows) & set(downstream_rows))
    if not common_ids:
        raise SystemExit("No common candidate_id rows across local delta, stress, and downstream inputs.")

    joined: list[dict[str, object]] = []
    for cid in common_ids:
        row: dict[str, object] = {"candidate_id": cid}
        row.update(stress_rows[cid])
        row.update(local_rows[cid])
        row.update(downstream_rows[cid])
        joined.append(row)

    for row in joined:
        local_l30 = get_float(row, "L30_local")
        stress_l30 = get_float(row, "L30_stress")
        row["L30_control_local"] = local_l30 if local_l30 is not None else ""
        row["L30_control_stress"] = stress_l30 if stress_l30 is not None else ""

    analysis: list[dict[str, object]] = []
    scopes = [("all_candidates", joined)]
    ordered = sorted(joined, key=lambda row: (get_float(row, "L30_stress") if get_float(row, "L30_stress") is not None else float("inf"), str(row["candidate_id"])))
    for raw_k in [part.strip() for part in args.endpoint_close_top_k.split(",") if part.strip()]:
        k = int(raw_k)
        scopes.append((f"top{k}_by_L30_stress", ordered[:k]))

    for scope, rows in scopes:
        analysis.extend(metric_rows(rows, scope, "L30_control_local"))
        analysis.extend(metric_rows(rows, scope, "L30_control_stress"))

    joined_csv = output_dir / "local_delta_joined.csv"
    analysis_csv = output_dir / "local_delta_analysis.csv"
    analysis_md = output_dir / "local_delta_analysis.md"
    manifest_json = output_dir / "local_delta_manifest.json"
    write_csv(joined_csv, joined)
    write_csv(analysis_csv, analysis)
    write_md(analysis_md, analysis, joined)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "inputs": {
            "local_delta_csv": args.local_delta_csv,
            "stress_table": args.stress_table,
            "downstream_summary": args.downstream_summary,
        },
        "joined_candidates": len(joined),
        "endpoint_close_top_k": args.endpoint_close_top_k,
        "interpretation_guardrail": "Treat local flatness as supported only if local slopes show a stable downstream@30 relation after controlling L30; otherwise keep S35 as a stress signal.",
        "artifacts": {
            "local_delta_joined": str(joined_csv),
            "local_delta_analysis": str(analysis_csv),
            "local_delta_analysis_md": str(analysis_md),
            "local_delta_manifest": str(manifest_json),
        },
    }
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {joined_csv}")
    print(f"Wrote {analysis_csv}")
    print(f"Wrote {analysis_md}")
    print(f"Wrote {manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
