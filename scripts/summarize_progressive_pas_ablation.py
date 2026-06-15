#!/usr/bin/env python3
"""Summarize same-pool Progressive PAS replay ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def num(value, default=float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value) -> str:
    value = num(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def setting_from_dir(path: Path) -> str:
    return path.name


def infer_params(setting: str, rows: list[dict[str, str]]) -> tuple[str, str, str]:
    top_k = rows[0].get("top_k", "") if rows else ""
    epsilon = ""
    margin = ""
    match = re.search(r"topk(\d+)_eps([0-9p]+)_margin([0-9p]+)", setting)
    if match:
        top_k = top_k or match.group(1)
        epsilon = match.group(2).replace("p", ".")
        margin = match.group(3).replace("p", ".")
    return top_k, epsilon, margin


def selection_index(selection_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in selection_rows:
        key = (row.get("prefix_step", ""), row.get("stage", ""))
        grouped.setdefault(key, {})[row.get("rule", "")] = row
    return grouped


def summarize_dir(path: Path) -> list[dict[str, str]]:
    promotion_rows = read_csv(path / "progressive_pas_promotion_gate.csv")
    selection_rows = read_csv(path / "progressive_pas_selection.csv")
    by_key = selection_index(selection_rows)
    setting = setting_from_dir(path)
    top_k, epsilon, margin = infer_params(setting, promotion_rows)
    rows: list[dict[str, str]] = []
    for row in promotion_rows:
        key = (row.get("prefix_step", ""), row.get("stage", ""))
        selected = by_key.get(key, {})
        ff = selected.get("FF-stage-endpoint", {})
        pas = selected.get("PAS-lookahead", {})
        endpoint_l30 = num(ff.get("L30"))
        pas_l30 = num(pas.get("L30"))
        endpoint_l40 = num(ff.get("L40"))
        pas_l40 = num(pas.get("L40"))
        raw_gain = row.get("pas_raw_lookahead_gain", row.get("lookahead_gain", ""))
        rows.append(
            {
                "setting": setting,
                "top_k": top_k,
                "epsilon": epsilon,
                "margin": margin,
                "prefix": row.get("prefix_step", ""),
                "stage": row.get("stage", ""),
                "next_stage": row.get("next_stage", ""),
                "candidate_count": row.get("candidate_count", ""),
                "endpoint_candidate": row.get("endpoint_candidate", ""),
                "pas_raw_candidate": row.get("pas_raw_candidate", ""),
                "same_candidate": row.get("same_candidate", str(row.get("endpoint_candidate", "") == row.get("pas_raw_candidate", ""))),
                "endpoint_L_stage": row.get("endpoint_L_stage", ""),
                "pas_raw_L_stage": row.get("pas_raw_L_stage", ""),
                "endpoint_L_next": row.get("endpoint_L_next", ""),
                "pas_raw_L_next": row.get("pas_raw_L_next", ""),
                "endpoint_price": row.get("endpoint_price", row.get("pas_raw_endpoint_price", "")),
                "raw_lookahead_gain": raw_gain,
                "promotion_decision": row.get("promotion_decision", ""),
                "hold_reason": row.get("hold_reason", ""),
                "PAS_minus_FF_L30": fmt(pas_l30 - endpoint_l30),
                "PAS_minus_FF_L40": fmt(pas_l40 - endpoint_l40),
                "extra_probe_evals": row.get("extra_probe_evals", pas.get("extra_probe_evals", "")),
                "total_eval_budget": pas.get("total_eval_budget", ""),
                "source_dir": str(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Progressive PAS ablation replay directories")
    parser.add_argument("replay_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for value in args.replay_dirs:
        rows.extend(summarize_dir(Path(value)))
    rows.sort(key=lambda row: (row["setting"], num(row["prefix"], 0), num(row["stage"], 0)))

    fieldnames = [
        "setting",
        "top_k",
        "epsilon",
        "margin",
        "prefix",
        "stage",
        "next_stage",
        "candidate_count",
        "endpoint_candidate",
        "pas_raw_candidate",
        "same_candidate",
        "endpoint_L_stage",
        "pas_raw_L_stage",
        "endpoint_L_next",
        "pas_raw_L_next",
        "endpoint_price",
        "raw_lookahead_gain",
        "promotion_decision",
        "hold_reason",
        "PAS_minus_FF_L30",
        "PAS_minus_FF_L40",
        "extra_probe_evals",
        "total_eval_budget",
        "source_dir",
    ]
    csv_path = out_dir / "progressive_pas_ablation_summary.csv"
    md_path = out_dir / "progressive_pas_ablation_summary.md"
    json_path = out_dir / "progressive_pas_ablation_manifest.json"
    write_csv(csv_path, rows, fieldnames)

    by_setting: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_setting.setdefault(row["setting"], []).append(row)
    setting_summary = []
    hard_stop_supported = None
    for setting, group in sorted(by_setting.items()):
        max_gain = max((num(row["raw_lookahead_gain"], float("-inf")) for row in group), default=float("nan"))
        promotes = sum(1 for row in group if row["promotion_decision"] == "PROMOTE")
        raw_positive = sum(1 for row in group if num(row["raw_lookahead_gain"]) > 0)
        item = {
            "setting": setting,
            "rows": len(group),
            "promotes": promotes,
            "raw_positive_count": raw_positive,
            "max_raw_lookahead_gain": fmt(max_gain),
        }
        setting_summary.append(item)
        if "topk50_eps0p10_margin0p00" in setting:
            hard_stop_supported = max_gain <= 0

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Progressive PAS Same-Pool Ablation Summary\n\n")
        handle.write("## Setting Summary\n\n")
        handle.write("| setting | rows | promotes | raw_positive_count | max_raw_lookahead_gain |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for item in setting_summary:
            handle.write(
                f"| {item['setting']} | {item['rows']} | {item['promotes']} | "
                f"{item['raw_positive_count']} | {item['max_raw_lookahead_gain']} |\n"
            )
        handle.write("\n## Hard Stop Check\n\n")
        if hard_stop_supported is True:
            handle.write("Progressive PAS main-line stopped: top_k=50, epsilon=0.10, margin=0.00 has no positive raw lookahead gain.\n")
        elif hard_stop_supported is False:
            handle.write("Progressive PAS main-line not stopped by the raw-gain rule; inspect L30 and replication criteria.\n")
        else:
            handle.write("Hard-stop setting top_k=50, epsilon=0.10, margin=0.00 was not found.\n")
        handle.write("\n## Gate Rows\n\n")
        handle.write("| setting | prefix | stage | next | candidates | raw_gain | promote | hold_reason | PAS_minus_FF_L30 | PAS_minus_FF_L40 |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['setting']} | {row['prefix']} | {row['stage']} | {row['next_stage']} | "
                f"{row['candidate_count']} | {row['raw_lookahead_gain']} | {row['promotion_decision']} | "
                f"{row['hold_reason']} | {row['PAS_minus_FF_L30']} | {row['PAS_minus_FF_L40']} |\n"
            )

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dirs": args.replay_dirs,
        "setting_summary": setting_summary,
        "hard_stop_topk50_eps010_margin0_raw_gain_zero": hard_stop_supported,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(csv_path)
    print(md_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
