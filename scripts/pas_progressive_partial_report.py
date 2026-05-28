#!/usr/bin/env python3
"""Summarize an in-progress Progressive PAS replay directory."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def safe_float_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def choose_min(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    return min(rows, key=lambda row: (num(row.get(key)), num(row.get("logppl_zero")), row.get("candidate_id", "")))


def row_key(row: dict[str, str]) -> tuple[int, float]:
    return (int(num(row.get("prefix_step"), -1)), round(num(row.get("stage")), 6))


def short_candidate(candidate_id: str) -> str:
    if not candidate_id:
        return ""
    match = re.search(r"step0*([0-9]+)", candidate_id)
    if match:
        return f"step{match.group(1)}"
    return candidate_id[-28:]


def infer_pas_raw(replay_dir: Path, row: dict[str, str], epsilon: float) -> dict[str, str]:
    if row.get("pas_raw_candidate"):
        return row
    prefix = int(num(row.get("prefix_step"), -1))
    stage = num(row.get("stage"))
    next_stage = num(row.get("next_stage"))
    if prefix < 0 or not math.isfinite(stage) or not math.isfinite(next_stage):
        return row
    label = f"prefix{prefix}_stage{safe_float_label(stage)}_to_{safe_float_label(next_stage)}"
    probe_rows = read_csv(replay_dir / "probes" / label / "probe_results.csv")
    if not probe_rows:
        return row
    endpoint = choose_min(probe_rows, "logppl_zero")
    endpoint_l = num(endpoint.get("logppl_zero"))
    band = [probe_row for probe_row in probe_rows if num(probe_row.get("logppl_zero")) <= endpoint_l + epsilon]
    if not band:
        band = [endpoint]
    pas_raw = choose_min(band, "logppl_plus")
    row = dict(row)
    row["pas_raw_candidate"] = pas_raw.get("candidate_id", "")
    row["pas_raw_L_stage"] = pas_raw.get("logppl_zero", "")
    row["pas_raw_L_next"] = pas_raw.get("logppl_plus", "")
    row["pas_raw_endpoint_price"] = str(num(pas_raw.get("logppl_zero")) - endpoint_l)
    row["pas_raw_lookahead_gain"] = str(num(endpoint.get("logppl_plus")) - num(pas_raw.get("logppl_plus")))
    return row


def write_table(handle, headers: list[str], rows: list[list[object]]) -> None:
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(str(item) for item in row) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a partial Progressive PAS replay report")
    parser.add_argument("replay_dir", help="Replay directory containing progressive_pas_*.csv files")
    parser.add_argument("--expected-batches", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    selection_csv = replay_dir / "progressive_pas_selection.csv"
    promotion_csv = replay_dir / "progressive_pas_promotion_gate.csv"
    probe_csvs = sorted((replay_dir / "probes").glob("*/probe_results.csv"))
    final_probe_csvs = sorted((replay_dir / "final_probes").glob("*/probe_results.csv"))

    promotion = [infer_pas_raw(replay_dir, row, args.epsilon) for row in read_csv(promotion_csv)]
    promotion_by_key = {row_key(row): row for row in promotion}
    selection = []
    for row in read_csv(selection_csv):
        if row.get("rule") == "PAS-lookahead" and not row.get("pas_raw_candidate"):
            promo = promotion_by_key.get(row_key(row), {})
            row = dict(row)
            for key in ("pas_raw_candidate", "pas_raw_L_stage", "pas_raw_L_next", "pas_raw_endpoint_price", "pas_raw_lookahead_gain"):
                if promo.get(key):
                    row[key] = promo[key]
        selection.append(row)
    out_path = Path(args.output) if args.output else replay_dir / "progressive_pas_partial_report.md"

    prefixes = sorted({int(num(row.get("prefix_step"), -1)) for row in promotion if num(row.get("prefix_step"), -1) >= 0})
    stages = sorted({num(row.get("stage")) for row in promotion if math.isfinite(num(row.get("stage")))})
    decisions = Counter(row.get("promotion_decision", "") for row in promotion)
    promote_count = decisions.get("PROMOTE", 0)
    hold_count = decisions.get("HOLD", 0)
    raw_improvements = [
        row for row in promotion
        if num(row.get("pas_raw_lookahead_gain", row.get("lookahead_gain", ""))) > 0
    ]
    best_raw = max(
        promotion,
        key=lambda row: num(row.get("pas_raw_lookahead_gain", row.get("lookahead_gain", "")), float("-inf")),
        default={},
    )
    if promote_count:
        verdict = "PAS has triggered at least one promotion."
    elif raw_improvements:
        verdict = "PAS has raw lookahead improvements, but no promotion yet."
    elif promotion:
        verdict = "No PAS acceleration signal yet: raw PAS keeps matching FF endpoint."
    else:
        verdict = "No promotion checks completed yet."

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Progressive PAS Partial Report\n\n")
        handle.write("## Simple Verdict\n\n")
        handle.write(f"- verdict: **{verdict}**\n")
        handle.write(f"- completed gate checks: `{len(promotion)}`\n")
        handle.write(f"- decisions: `PROMOTE={promote_count}`, `HOLD={hold_count}`\n")
        handle.write(f"- raw PAS improvements: `{len(raw_improvements)}`\n")
        handle.write(
            f"- best raw gain so far: `{fmt(best_raw.get('pas_raw_lookahead_gain', best_raw.get('lookahead_gain', '')))}"
            f"` at prefix `{best_raw.get('prefix_step', '')}`, stage `{fmt(best_raw.get('stage', ''))}`\n"
        )
        handle.write(f"- completed probe batches: `{len(probe_csvs) + len(final_probe_csvs)}` / `{args.expected_batches}`\n")
        handle.write(f"- prefixes seen: `{','.join(map(str, prefixes))}`\n\n")

        handle.write("## Compact Gate Table\n\n")
        compact_rows = []
        for row in sorted(promotion, key=lambda r: (num(r.get("prefix_step")), num(r.get("stage")))):
            raw_gain = num(row.get("pas_raw_lookahead_gain", row.get("lookahead_gain", "")))
            compact_rows.append(
                [
                    row.get("prefix_step", ""),
                    f"{fmt(row.get('stage', ''))}->{fmt(row.get('next_stage', ''))}",
                    row.get("promotion_decision", ""),
                    fmt(raw_gain),
                    short_candidate(row.get("pas_raw_candidate", "")),
                    short_candidate(row.get("pas_candidate", "")),
                    row.get("hold_reason", ""),
                ]
            )
        write_table(
            handle,
            ["prefix", "stage", "decision", "raw_gain", "raw", "selected", "why_hold"],
            compact_rows,
        )
        handle.write("\n")

        handle.write("## Run Progress\n\n")
        handle.write(f"- replay_dir: `{replay_dir}`\n")
        handle.write(f"- stage probe batches: `{len(probe_csvs)}`\n")
        handle.write(f"- final probe batches: `{len(final_probe_csvs)}`\n")
        handle.write(f"- selection rows: `{len(selection)}`\n")
        handle.write(f"- stages seen: `{','.join(fmt(s) for s in stages)}`\n\n")

        handle.write("## Decision Counts\n\n")
        write_table(
            handle,
            ["decision", "count"],
            [[decision or "(blank)", count] for decision, count in sorted(decisions.items())],
        )
        handle.write("\n")

        handle.write("## Detailed Gate Rows\n\n")
        promo_rows = []
        for row in sorted(promotion, key=lambda r: (num(r.get("prefix_step")), num(r.get("stage")))):
            promo_rows.append(
                [
                    row.get("prefix_step", ""),
                    fmt(row.get("stage", "")),
                    fmt(row.get("next_stage", "")),
                    row.get("promotion_mode", ""),
                    row.get("promotion_decision", ""),
                    row.get("episodes_saved_vs_full_prefix", ""),
                    row.get("online_gate_probe_evals", ""),
                    fmt(row.get("pas_raw_endpoint_price", row.get("endpoint_price", ""))),
                    fmt(row.get("pas_raw_lookahead_gain", row.get("lookahead_gain", ""))),
                    short_candidate(row.get("pas_raw_candidate", "")),
                    fmt(row.get("endpoint_price", "")),
                    fmt(row.get("lookahead_gain", "")),
                    row.get("hold_reason", ""),
                    short_candidate(row.get("pas_candidate", "")),
                ]
            )
        write_table(
            handle,
            ["prefix", "stage", "next", "mode", "decision", "saved_eps", "probe_evals", "raw_price", "raw_gain", "raw_candidate", "selected_price", "selected_gain", "hold_reason", "selected_candidate"],
            promo_rows,
        )
        handle.write("\n")

        if selection:
            handle.write("## Latest Prefix Selection\n\n")
            latest_prefix = max(int(num(row.get("prefix_step"), -1)) for row in selection)
            latest = [row for row in selection if int(num(row.get("prefix_step"), -1)) == latest_prefix]
            latest.sort(key=lambda r: (num(r.get("stage")), r.get("rule", "")))
            table_rows = []
            for row in latest:
                table_rows.append(
                    [
                        row.get("prefix_step", ""),
                        fmt(row.get("stage", "")),
                        row.get("rule", ""),
                        row.get("promotion_decision", ""),
                        short_candidate(row.get("pas_raw_candidate", "")),
                        fmt(row.get("pas_raw_lookahead_gain", "")),
                        fmt(row.get("L30", "")),
                        fmt(row.get("L40", "")),
                        fmt(row.get("Regret40", "")),
                        short_candidate(row.get("candidate_id", "")),
                    ]
                )
            write_table(
                handle,
                ["prefix", "stage", "rule", "promote", "pas_raw_candidate", "pas_raw_gain", "L30", "L40", "Regret40", "candidate"],
                table_rows,
            )
            handle.write("\n")

            by_stage: dict[tuple[int, float], list[dict[str, str]]] = defaultdict(list)
            for row in selection:
                by_stage[(int(num(row.get("prefix_step"), -1)), num(row.get("stage")))].append(row)
            handle.write("## PAS vs FF By Completed Checkpoint\n\n")
            rows = []
            for (prefix, stage), group in sorted(by_stage.items()):
                ff = next((row for row in group if row.get("rule") == "FF-stage-endpoint"), None)
                pas = next((row for row in group if row.get("rule") == "PAS-lookahead"), None)
                if not ff or not pas:
                    continue
                rows.append(
                    [
                        prefix,
                        fmt(stage),
                        fmt(num(pas.get("L30")) - num(ff.get("L30"))),
                        fmt(num(pas.get("L40")) - num(ff.get("L40"))),
                        pas.get("promotion_decision", ""),
                        pas.get("promotion_hold_reason", ""),
                    ]
                )
            write_table(handle, ["prefix", "stage", "PAS_minus_FF_L30", "PAS_minus_FF_L40", "decision", "hold_reason"], rows)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
