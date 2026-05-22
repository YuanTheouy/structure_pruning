#!/usr/bin/env python3
"""Compare local-probe PAS selection rules against downstream@30 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


TASK_ORDER = ["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "boolq"]
BASE_RULES = ["FF-Endpoint", "PAS-S30.25", "PAS-S30.50", "PAS-S35", "Oracle-Downstream30", "Oracle-L40"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PAS local selection and downstream@30.")
    parser.add_argument("--local-delta-table", required=True)
    parser.add_argument("--downstream-summary", required=True)
    parser.add_argument("--stress-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-m", default="2,5,8,13")
    parser.add_argument("--epsilon", default="0.02,0.05,0.10")
    parser.add_argument("--model", default="opt-2.7b")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", default="3025")
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


def numeric(row: dict[str, object], key: str) -> float | None:
    return to_float(row.get(key))


def first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return value
    return ""


def normalize_local(row: dict[str, str]) -> dict[str, object]:
    out = {
        "candidate_id": row["candidate_id"],
        "L30": row.get("L30", ""),
        "L3025": row.get("L3025", ""),
        "L3050": row.get("L3050", ""),
        "S3025": row.get("S3025", ""),
        "S3050": row.get("S3050", ""),
        "PPL30": row.get("PPL30", ""),
        "PPL3025": row.get("PPL3025", ""),
        "PPL3050": row.get("PPL3050", ""),
    }
    for key in ("L31", "L3100", "S31", "S3100", "PPL31", "PPL3100"):
        if row.get(key) not in ("", None):
            out[key] = row.get(key, "")
    if "S31" not in out and "S3100" in out:
        out["S31"] = out["S3100"]
    if "L31" not in out and "L3100" in out:
        out["L31"] = out["L3100"]
    return out


def normalize_stress(row: dict[str, str]) -> dict[str, object]:
    l30 = first_value(row, "L30", "L30_raw", "ell_0")
    l40 = first_value(row, "L40", "L40_raw", "ell_h")
    delta40 = first_value(row, "Delta40")
    if not delta40:
        l30_f = to_float(l30)
        l40_f = to_float(l40)
        if l30_f is not None and l40_f is not None:
            delta40 = str(l40_f - l30_f)
    return {
        "candidate_id": row["candidate_id"],
        "selection_tags": row.get("selection_tags", ""),
        "S35": first_value(row, "S35", "slope"),
        "L40": l40,
        "Regret40": first_value(row, "Regret40", "regret"),
        "Delta40": delta40,
    }


def normalize_downstream(row: dict[str, str]) -> dict[str, object]:
    return {
        "candidate_id": row["candidate_id"],
        "avg_pruned_score": row.get("avg_pruned_score", ""),
        "task_count": row.get("task_count", ""),
    }


def read_task_scores(downstream_summary: Path) -> dict[str, dict[str, object]]:
    """Best-effort pivot from sibling downstream_retention_opt27b.csv."""
    retention_path = downstream_summary.with_name("downstream_retention_opt27b.csv")
    if not retention_path.exists():
        return {}
    scores: dict[str, dict[str, object]] = {}
    for row in read_csv(retention_path):
        cid = row.get("candidate_id", "")
        task = row.get("task", "")
        if not cid or not task or task == "average":
            continue
        value = first_value(row, "pruned_score", "score")
        if value == "":
            continue
        scores.setdefault(cid, {})[task] = value
    return scores


def join_rows(
    local_rows: list[dict[str, str]],
    stress_rows: list[dict[str, str]],
    downstream_rows: list[dict[str, str]],
    task_scores: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    local = {row["candidate_id"]: normalize_local(row) for row in local_rows}
    stress = {row["candidate_id"]: normalize_stress(row) for row in stress_rows}
    downstream = {row["candidate_id"]: normalize_downstream(row) for row in downstream_rows}
    common = sorted(set(local) & set(stress) & set(downstream))
    rows: list[dict[str, object]] = []
    for cid in common:
        row: dict[str, object] = {"candidate_id": cid}
        row.update(local[cid])
        row.update(stress[cid])
        row.update(downstream[cid])
        for task, value in task_scores.get(cid, {}).items():
            row[task] = value
        rows.append(row)
    return rows


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def scope_rows(rows: list[dict[str, object]], top_m: list[int], epsilons: list[float]) -> list[tuple[str, list[dict[str, object]]]]:
    ordered = sorted(rows, key=lambda row: (numeric(row, "L30") if numeric(row, "L30") is not None else float("inf"), str(row["candidate_id"])))
    scopes: list[tuple[str, list[dict[str, object]]]] = []
    for m in top_m:
        scopes.append((f"top_m={m}", ordered[:m]))
    best_l30 = numeric(ordered[0], "L30") if ordered else None
    if best_l30 is not None:
        for eps in epsilons:
            subset = [row for row in ordered if numeric(row, "L30") is not None and numeric(row, "L30") <= best_l30 + eps]
            scopes.append((f"epsilon={eps:g}", subset))
    scopes.append(("all_candidates", ordered))
    return scopes


def available_rules(rows: list[dict[str, object]]) -> list[str]:
    rules = BASE_RULES[:]
    if any(numeric(row, "S31") is not None or numeric(row, "S3100") is not None for row in rows):
        rules.insert(3, "PAS-S31")
    return rules


def choose_candidate(rows: list[dict[str, object]], rule: str) -> tuple[dict[str, object] | None, object]:
    specs = {
        "FF-Endpoint": ("L30", False),
        "PAS-S30.25": ("S3025", False),
        "PAS-S30.50": ("S3050", False),
        "PAS-S31": ("S31", False),
        "PAS-S35": ("S35", False),
        "Oracle-Downstream30": ("avg_pruned_score", True),
        "Oracle-L40": ("L40", False),
    }
    key, maximize = specs[rule]
    valid = [row for row in rows if numeric(row, key) is not None]
    if not valid:
        return None, ""
    selected = sorted(
        valid,
        key=lambda row: (
            -numeric(row, key) if maximize else numeric(row, key),
            numeric(row, "L30") if numeric(row, "L30") is not None else float("inf"),
            str(row["candidate_id"]),
        ),
    )[0]
    return selected, selected.get(key, "")


def selection_table(scopes: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    existing_tasks = {key for _, subset in scopes for row in subset for key in row if key in TASK_ORDER}
    task_columns = [task for task in TASK_ORDER if task in existing_tasks]
    for scope_name, subset in scopes:
        for rule in available_rules(subset):
            selected, score = choose_candidate(subset, rule)
            if selected is None:
                rows.append({"rule": rule, "selection_scope": scope_name, "scope_size": len(subset), "candidate_id": "", "selection_score": ""})
                continue
            row = {
                "rule": rule,
                "selection_scope": scope_name,
                "scope_size": len(subset),
                "candidate_id": selected["candidate_id"],
                "selection_score": score,
                "analysis_only_oracle": rule.startswith("Oracle"),
                "L30": selected.get("L30", ""),
                "S3025": selected.get("S3025", ""),
                "S3050": selected.get("S3050", ""),
                "S31": selected.get("S31", selected.get("S3100", "")),
                "S35": selected.get("S35", ""),
                "L40": selected.get("L40", ""),
                "Regret40": selected.get("Regret40", ""),
                "Delta40": selected.get("Delta40", ""),
                "avg_pruned_score": selected.get("avg_pruned_score", ""),
                "task_count": selected.get("task_count", ""),
            }
            for task in task_columns:
                row[task] = selected.get(task, "")
            rows.append(row)
    return rows


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
    ranked = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for pos in range(i, j):
            ranked[order[pos]] = rank
        i = j
    return ranked


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def residualize(values: list[float], controls: list[float]) -> list[float] | None:
    if len(values) != len(controls) or len(values) < 3:
        return None
    mc = sum(controls) / len(controls)
    mv = sum(values) / len(values)
    denom = sum((c - mc) ** 2 for c in controls)
    if denom <= 0:
        return None
    beta = sum((c - mc) * (v - mv) for c, v in zip(controls, values)) / denom
    alpha = mv - beta * mc
    return [v - (alpha + beta * c) for v, c in zip(values, controls)]


def partial_corr(xs: list[float], ys: list[float], controls: list[float]) -> float | None:
    rx = residualize(xs, controls)
    ry = residualize(ys, controls)
    if rx is None or ry is None:
        return None
    return pearson(rx, ry)


def correlation_table(scopes: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    has_s31 = any(numeric(row, "S31") is not None or numeric(row, "S3100") is not None for _, subset in scopes for row in subset)
    predictors = ["S3025", "S3050"] + (["S31"] if has_s31 else []) + ["S35"]
    targets = [
        ("avg_pruned_score", "higher_is_better", "negative_if_local_slope_means_fragility"),
        ("L40", "lower_is_better", "positive_if_signal_predicts_stress_loss"),
        ("Regret40", "lower_is_better", "positive_if_signal_predicts_stress_regret"),
    ]
    for scope_name, subset in scopes:
        for predictor in predictors:
            for target, target_direction, expected_relation in targets:
                xs: list[float] = []
                ys: list[float] = []
                controls: list[float] = []
                for row in subset:
                    x = numeric(row, predictor)
                    y = numeric(row, target)
                    c = numeric(row, "L30")
                    if x is None or y is None or c is None:
                        continue
                    xs.append(x)
                    ys.append(y)
                    controls.append(c)
                output.append(
                    {
                        "selection_scope": scope_name,
                        "scope_size": len(subset),
                        "predictor": predictor,
                        "target": target,
                        "target_direction": target_direction,
                        "expected_relation": expected_relation,
                        "control": "L30",
                        "n": len(xs),
                        "pearson": pearson(xs, ys),
                        "spearman": spearman(xs, ys),
                        "partial_corr": partial_corr(xs, ys, controls),
                    }
                )
    return output


def fmt(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.6g}"


def write_selection_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "selection_scope",
        "rule",
        "scope_size",
        "candidate_id",
        "L30",
        "S3025",
        "S3050",
        "S31",
        "S35",
        "L40",
        "Regret40",
        "avg_pruned_score",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PAS Local Selection Downstream Table\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |\n")


def write_corr_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "selection_scope",
        "predictor",
        "target",
        "target_direction",
        "n",
        "pearson",
        "spearman",
        "partial_corr",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PAS Local Signal Correlation\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    local_path = Path(args.local_delta_table)
    downstream_path = Path(args.downstream_summary)
    stress_path = Path(args.stress_table)
    rows = join_rows(
        read_csv(local_path),
        read_csv(stress_path),
        read_csv(downstream_path),
        read_task_scores(downstream_path),
    )
    if not rows:
        raise SystemExit("No shared candidate_id rows across local, stress, and downstream tables.")
    scopes = scope_rows(rows, parse_int_list(args.top_m), parse_float_list(args.epsilon))
    selected_rows = selection_table(scopes)
    corr_rows = correlation_table(scopes)

    selection_csv = output_dir / "local_selection_downstream_table.csv"
    selection_md = output_dir / "local_selection_downstream_table.md"
    corr_csv = output_dir / "local_signal_correlation.csv"
    corr_md = output_dir / "local_signal_correlation.md"
    manifest_json = output_dir / "local_selection_manifest.json"
    write_csv(selection_csv, selected_rows)
    write_selection_md(selection_md, selected_rows)
    write_csv(corr_csv, corr_rows)
    write_corr_md(corr_md, corr_rows)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "inputs": {
            "local_delta_table": str(local_path),
            "downstream_summary": str(downstream_path),
            "stress_table": str(stress_path),
        },
        "joined_candidates": len(rows),
        "selection_scopes": [name for name, _ in scopes],
        "selection_rules": available_rules(rows),
        "analysis_only_rules": ["Oracle-Downstream30", "Oracle-L40"],
        "artifacts": {
            "local_selection_downstream_table": str(selection_csv),
            "local_selection_downstream_table_md": str(selection_md),
            "local_signal_correlation": str(corr_csv),
            "local_signal_correlation_md": str(corr_md),
            "local_selection_manifest": str(manifest_json),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {selection_csv}")
    print(f"Wrote {selection_md}")
    print(f"Wrote {corr_csv}")
    print(f"Wrote {corr_md}")
    print(f"Wrote {manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
