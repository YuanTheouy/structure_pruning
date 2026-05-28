#!/usr/bin/env python3
"""Summarize an in-progress Progressive PAS replay directory."""

from __future__ import annotations

import argparse
import csv
import math
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


def write_table(handle, headers: list[str], rows: list[list[object]]) -> None:
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(str(item) for item in row) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a partial Progressive PAS replay report")
    parser.add_argument("replay_dir", help="Replay directory containing progressive_pas_*.csv files")
    parser.add_argument("--expected-batches", type=int, default=60)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    selection_csv = replay_dir / "progressive_pas_selection.csv"
    promotion_csv = replay_dir / "progressive_pas_promotion_gate.csv"
    probe_csvs = sorted((replay_dir / "probes").glob("*/probe_results.csv"))
    final_probe_csvs = sorted((replay_dir / "final_probes").glob("*/probe_results.csv"))

    selection = read_csv(selection_csv)
    promotion = read_csv(promotion_csv)
    out_path = Path(args.output) if args.output else replay_dir / "progressive_pas_partial_report.md"

    prefixes = sorted({int(num(row.get("prefix_step"), -1)) for row in promotion if num(row.get("prefix_step"), -1) >= 0})
    stages = sorted({num(row.get("stage")) for row in promotion if math.isfinite(num(row.get("stage")))})
    decisions = Counter(row.get("promotion_decision", "") for row in promotion)

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Progressive PAS Partial Report\n\n")
        handle.write(f"- replay_dir: `{replay_dir}`\n")
        handle.write(f"- probe batches done: `{len(probe_csvs) + len(final_probe_csvs)}` / `{args.expected_batches}`\n")
        handle.write(f"- stage probe batches: `{len(probe_csvs)}`\n")
        handle.write(f"- final probe batches: `{len(final_probe_csvs)}`\n")
        handle.write(f"- promotion rows: `{len(promotion)}`\n")
        handle.write(f"- selection rows: `{len(selection)}`\n")
        handle.write(f"- prefixes seen: `{','.join(map(str, prefixes))}`\n")
        handle.write(f"- stages seen: `{','.join(fmt(s) for s in stages)}`\n\n")

        handle.write("## Promotion Decisions\n\n")
        write_table(
            handle,
            ["decision", "count"],
            [[decision or "(blank)", count] for decision, count in sorted(decisions.items())],
        )
        handle.write("\n")

        handle.write("## Promotion Gate Rows\n\n")
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
                    fmt(row.get("endpoint_price", "")),
                    fmt(row.get("lookahead_gain", "")),
                    row.get("hold_reason", ""),
                    row.get("pas_candidate", ""),
                ]
            )
        write_table(
            handle,
            ["prefix", "stage", "next", "mode", "decision", "saved_eps", "probe_evals", "endpoint_price", "lookahead_gain", "hold_reason", "pas_candidate"],
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
                        fmt(row.get("L30", "")),
                        fmt(row.get("L40", "")),
                        fmt(row.get("Regret40", "")),
                        row.get("candidate_id", ""),
                    ]
                )
            write_table(
                handle,
                ["prefix", "stage", "rule", "promote", "L30", "L40", "Regret40", "candidate"],
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
