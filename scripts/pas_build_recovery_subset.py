#!/usr/bin/env python3
"""Build fixed PAS recovery subset from a candidate stress table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed recovery subset R.")
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
    parser.add_argument("--selected-candidates-json", default="")
    parser.add_argument("--endpoint-top-n", type=int, default=8)
    parser.add_argument("--endpoint-window-n", type=int, default=20)
    parser.add_argument("--low-s35-n", type=int, default=4)
    parser.add_argument("--high-s35-n", type=int, default=4)
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


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("inf")


def selected_ids(path: str) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out = {}
    for rule, item in payload.items():
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id") or item.get("probe_row", {}).get("candidate_id")
        if candidate_id:
            out[rule] = candidate_id
    return out


def add_tag(tags: dict[str, set[str]], candidate_id: str, tag: str) -> None:
    tags.setdefault(candidate_id, set()).add(tag)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        print(f"Would build recovery subset from {args.stress_table}")
        return 0

    rows = read_csv(Path(args.stress_table))
    by_id = {row["candidate_id"]: row for row in rows}
    endpoint_sorted = sorted(rows, key=lambda row: (as_float(row, "L30"), row["candidate_id"]))
    endpoint_window = endpoint_sorted[: args.endpoint_window_n]
    low_s35 = sorted(endpoint_window, key=lambda row: (as_float(row, "S35"), row["candidate_id"]))[: args.low_s35_n]
    high_s35 = sorted(endpoint_window, key=lambda row: (-as_float(row, "S35"), row["candidate_id"]))[: args.high_s35_n]
    oracle40 = min(rows, key=lambda row: (as_float(row, "L40"), row["candidate_id"]))
    selected = selected_ids(args.selected_candidates_json)

    tags: dict[str, set[str]] = {}
    for row in endpoint_sorted[: args.endpoint_top_n]:
        add_tag(tags, row["candidate_id"], f"top{args.endpoint_top_n}_L30")
    for row in low_s35:
        add_tag(tags, row["candidate_id"], f"low{args.low_s35_n}_S35_within_top{args.endpoint_window_n}_L30")
    for row in high_s35:
        add_tag(tags, row["candidate_id"], f"high{args.high_s35_n}_S35_within_top{args.endpoint_window_n}_L30")
    add_tag(tags, oracle40["candidate_id"], "Oracle40")
    for rule, candidate_id in selected.items():
        add_tag(tags, candidate_id, rule)

    subset_rows = []
    for candidate_id in sorted(tags, key=lambda cid: (as_float(by_id[cid], "L30"), cid)):
        source = by_id[candidate_id]
        subset_rows.append(
            {
                "model": source.get("model", Path(args.model).name),
                "dataset": args.dataset,
                "seed": args.seed,
                "candidate_pool_id": source.get("candidate_pool_id", ""),
                "candidate_id": candidate_id,
                "selection_tags": ";".join(sorted(tags[candidate_id])),
                "L30_raw": source.get("L30", ""),
                "PPL30_raw": source.get("PPL30", ""),
                "S35": source.get("S35", ""),
                "L40_raw": source.get("L40", ""),
                "Regret40": source.get("Regret40", ""),
                "source_stress_table": args.stress_table,
            }
        )

    out_csv = output_dir / f"recovery_subset_opt27b_seed{args.seed}.csv"
    out_json = output_dir / f"recovery_subset_opt27b_seed{args.seed}.json"
    write_csv(out_csv, subset_rows)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump({"candidate_ids": [row["candidate_id"] for row in subset_rows], "rows": subset_rows}, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
