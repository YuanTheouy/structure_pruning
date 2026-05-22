#!/usr/bin/env python3
"""Collect downstream retention rows from task result CSV/JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PAS downstream retention results.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--target-sigma", type=float, default=0.30)
    parser.add_argument("--probe-sigma", type=float, default=0.35)
    parser.add_argument("--heldout-sigma", type=float, default=0.40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidate-table", default="")
    parser.add_argument("--recovery-table", default="")
    parser.add_argument("--endpoint-close-top-k", default="8,13")
    parser.add_argument("--task-results", nargs="*", default=[])
    parser.add_argument("--downstream-dir", default="")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No downstream rows to write: {path}")
    fieldnames = []
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


def metric_rows(candidate_rows: list[dict[str, object]], scope: str) -> list[dict[str, object]]:
    metrics = []
    targets = [
        ("avg_pruned_score", "higher_is_better"),
        ("avg_retention", "higher_is_better"),
        ("avg_drop", "lower_is_better"),
    ]
    for target, direction in targets:
        xs: list[float] = []
        ys: list[float] = []
        controls: list[float] = []
        for row in candidate_rows:
            s35 = to_float(row.get("S35"))
            y = to_float(row.get(target))
            l30 = to_float(row.get("L30_raw"))
            if s35 is None or y is None:
                continue
            xs.append(s35)
            ys.append(y)
            controls.append(l30 if l30 is not None else 0.0)
        if len(xs) < 3:
            continue
        metrics.append({
            "scope": scope,
            "metric": f"Pearson(S35,{target})",
            "value": pearson(xs, ys),
            "n": len(xs),
            "direction": direction,
            "notes": "raw candidate-level downstream aggregation",
        })
        metrics.append({
            "scope": scope,
            "metric": f"Spearman(S35,{target})",
            "value": spearman(xs, ys),
            "n": len(xs),
            "direction": direction,
            "notes": "raw candidate-level downstream aggregation",
        })
        if all(to_float(row.get("L30_raw")) is not None for row in candidate_rows if to_float(row.get(target)) is not None and to_float(row.get("S35")) is not None):
            metrics.append({
                "scope": scope,
                "metric": f"partial_corr(S35,{target}|L30_raw)",
                "value": partial_corr(xs, ys, controls),
                "n": len(xs),
                "direction": direction,
                "notes": "endpoint-controlled relation",
            })
    return metrics


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("Would collect downstream task result files:")
        for path in args.task_results:
            print(f"  {path}")
        return 0

    candidate_table = args.candidate_table or args.recovery_table
    if not candidate_table:
        raise SystemExit("Pass --candidate-table with the P0 stress table, or --recovery-table for legacy runs.")
    candidate_meta = {row["candidate_id"]: row for row in read_csv(Path(candidate_table))}
    task_results = list(args.task_results)
    if args.downstream_dir:
        task_results.extend(str(path) for path in Path(args.downstream_dir).glob("downstream_eval/*/downstream_results.json"))
    rows: list[dict[str, object]] = []
    for result_path in task_results:
        path = Path(result_path)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload_checkpoint_type = payload.get("checkpoint_type", "")
            if not payload_checkpoint_type and "raw_no_recovery" in str(payload.get("selected_mode", "")):
                payload_checkpoint_type = "raw_no_recovery"
            if not payload_checkpoint_type:
                payload_checkpoint_type = "recovered"
            if "downstream_results" in payload:
                cid = payload.get("candidate_id")
                raw_rows = []
                for row in (payload.get("downstream_results") or {}).get("rows", []):
                    raw_rows.append({
                        "candidate_id": cid,
                        "checkpoint_type": payload_checkpoint_type,
                        "task": row.get("task", ""),
                        "metric": row.get("metric", ""),
                        "pruned_score": row.get("score", ""),
                        "notes": (payload.get("downstream_results") or {}).get("eval_type", ""),
                    })
            else:
                raw_rows = payload.get("rows", [])
        else:
            raw_rows = read_csv(path)
        for raw in raw_rows:
            cid = raw.get("candidate_id")
            rec = candidate_meta.get(cid, {})
            dense_raw = raw.get("dense_score", "")
            pruned_raw = raw.get("pruned_score", raw.get("score", ""))
            dense = float(dense_raw) if dense_raw not in ("", None) else None
            pruned = float(pruned_raw) if pruned_raw not in ("", None) else None
            rows.append(
                {
                    "model": Path(args.model).name,
                    "seed": args.seed,
                    "candidate_id": cid,
                    "selection_tags": rec.get("selection_tags", ""),
                    "checkpoint_type": raw.get("checkpoint_type", "recovered"),
                    "task": raw.get("task", ""),
                    "metric": raw.get("metric", ""),
                    "dense_score": dense if dense is not None else "",
                    "pruned_score": pruned if pruned is not None else "",
                    "retention": pruned / dense if dense else "",
                    "drop": dense - pruned if dense is not None and pruned is not None else "",
                    "L30_raw": rec.get("L30_raw", rec.get("L30", "")),
                    "PPL30_raw": rec.get("PPL30_raw", rec.get("PPL30", "")),
                    "S35": rec.get("S35", ""),
                    "L40_raw": rec.get("L40_raw", rec.get("L40", "")),
                    "Regret40": rec.get("Regret40", ""),
                    "artifact_path": result_path,
                    "notes": raw.get("notes", ""),
                }
            )

    out_csv = Path(args.output_dir) / "downstream_retention_opt27b.csv"
    write_csv(out_csv, rows)
    by_candidate: dict[str, dict[str, object]] = {}
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        item = by_candidate.setdefault(
            cid,
            {
                "candidate_id": cid,
                "selection_tags": row.get("selection_tags", ""),
                "checkpoint_type": row.get("checkpoint_type", ""),
                "L30_raw": row.get("L30_raw", ""),
                "PPL30_raw": row.get("PPL30_raw", ""),
                "S35": row.get("S35", ""),
                "L40_raw": row.get("L40_raw", ""),
                "Regret40": row.get("Regret40", ""),
                "scores": [],
                "retentions": [],
                "drops": [],
            },
        )
        if row.get("task") == "average":
            continue
        score = to_float(row.get("pruned_score"))
        retention = to_float(row.get("retention"))
        drop = to_float(row.get("drop"))
        if score is not None:
            item["scores"].append(score)
        if retention is not None:
            item["retentions"].append(retention)
        if drop is not None:
            item["drops"].append(drop)
    candidate_summary = []
    for item in by_candidate.values():
        scores = item.pop("scores")
        retentions = item.pop("retentions")
        drops = item.pop("drops")
        item["task_count"] = len(scores)
        item["avg_pruned_score"] = sum(scores) / len(scores) if scores else ""
        item["avg_retention"] = sum(retentions) / len(retentions) if retentions else ""
        item["avg_drop"] = sum(drops) / len(drops) if drops else ""
        candidate_summary.append(item)
    if candidate_summary:
        summary_csv = Path(args.output_dir) / f"downstream_candidate_summary_opt27b_seed{args.seed}.csv"
        write_csv(summary_csv, candidate_summary)
        analysis_csv = Path(args.output_dir) / f"downstream_analysis_opt27b_seed{args.seed}.csv"
        analysis_rows = metric_rows(candidate_summary, "all_candidates")
        ordered = sorted(
            [row for row in candidate_summary if to_float(row.get("L30_raw")) is not None],
            key=lambda row: (to_float(row.get("L30_raw")), str(row.get("candidate_id", ""))),
        )
        for raw_k in [part.strip() for part in args.endpoint_close_top_k.split(",") if part.strip()]:
            k = int(raw_k)
            analysis_rows.extend(metric_rows(ordered[:k], f"top{k}_by_L30_raw"))
        write_csv(analysis_csv, analysis_rows)
    else:
        summary_csv = ""
        analysis_csv = ""
    out_md = Path(args.output_dir) / "downstream_retention_opt27b.md"
    out_md.write_text(
        f"# Downstream Retention\n\nRows: {len(rows)}\nCSV: `{out_csv}`\n"
        f"Candidate summary: `{summary_csv}`\nAnalysis: `{analysis_csv}`\n",
        encoding="utf-8",
    )
    manifest = {
        "model": args.model,
        "seed": args.seed,
        "candidate_table": candidate_table,
        "downstream_dir": args.downstream_dir,
        "task_results": task_results,
        "endpoint_close_top_k": args.endpoint_close_top_k,
        "drop_note": "drop/retention are populated only when dense_score is provided in task result rows; raw lm-eval JSON provides downstream score.",
        "artifacts": {
            "downstream_retention": str(out_csv),
            "downstream_retention_md": str(out_md),
            "downstream_candidate_summary": str(summary_csv),
            "downstream_analysis": str(analysis_csv),
        },
    }
    with (Path(args.output_dir) / "downstream_manifest_opt27b.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
