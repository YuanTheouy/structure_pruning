#!/usr/bin/env python3
"""Merge sharded Early-Warning probe outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded probe_results.csv/jsonl files.")
    parser.add_argument("probe_csvs", nargs="+")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for csv_value in args.probe_csvs:
        path = Path(csv_value)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)

    if not rows:
        raise RuntimeError("no probe rows found to merge")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (int(row.get("shard_id") or 0), row.get("candidate_id", "")))

    csv_path = output_dir / "probe_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = output_dir / "probe_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"=> Merged {len(rows)} probe rows into {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
