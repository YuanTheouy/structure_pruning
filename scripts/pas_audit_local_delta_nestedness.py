#!/usr/bin/env python3
"""Audit whether local-delta negative slopes line up with non-nested projections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path


SLOPE_PATHS = {
    "S3025": [(0.3000, 0.3025)],
    "S3050": [(0.3000, 0.3025), (0.3025, 0.3050)],
    "S31": [(0.3000, 0.3025), (0.3025, 0.3050), (0.3050, 0.3100)],
    "S3100": [(0.3000, 0.3025), (0.3025, 0.3050), (0.3050, 0.3100)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join local slopes with nestedness/mask-change diagnostics.")
    parser.add_argument("--local-delta-table", required=True)
    parser.add_argument("--nestedness-by-candidate", required=True)
    parser.add_argument("--mask-change-by-candidate", required=True)
    parser.add_argument("--nestedness-violations", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slope-columns", default="S3025,S3050,S31")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        rows = [{}]
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


def pair_key(row: dict[str, str]) -> tuple[str, float, float]:
    return (row["candidate_id"], round(float(row["sigma_a"]), 4), round(float(row["sigma_b"]), 4))


def pair_tuple(pair: tuple[float, float]) -> tuple[float, float]:
    return (round(pair[0], 4), round(pair[1], 4))


def sum_float(rows: list[dict[str, str]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            total += value
    return total


def max_float(rows: list[dict[str, str]], key: str) -> float:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_joined(
    local_rows: list[dict[str, str]],
    nested_rows: list[dict[str, str]],
    mask_rows: list[dict[str, str]],
    violation_rows: list[dict[str, str]],
    slope_columns: list[str],
) -> list[dict[str, object]]:
    nested_by_pair = defaultdict(list)
    mask_by_pair = defaultdict(list)
    violation_by_pair = defaultdict(list)
    for row in nested_rows:
        nested_by_pair[pair_key(row)].append(row)
    for row in mask_rows:
        mask_by_pair[pair_key(row)].append(row)
    for row in violation_rows:
        violation_by_pair[pair_key(row)].append(row)

    output: list[dict[str, object]] = []
    for row in local_rows:
        cid = row["candidate_id"]
        for slope in slope_columns:
            slope_value = to_float(row.get(slope))
            if slope_value is None:
                continue
            pairs = SLOPE_PATHS.get(slope)
            if not pairs:
                continue
            nested_path_rows: list[dict[str, str]] = []
            mask_path_rows: list[dict[str, str]] = []
            violation_path_rows: list[dict[str, str]] = []
            for sigma_a, sigma_b in pairs:
                key = (cid, *pair_tuple((sigma_a, sigma_b)))
                nested_path_rows.extend(nested_by_pair.get(key, []))
                mask_path_rows.extend(mask_by_pair.get(key, []))
                violation_path_rows.extend(violation_by_pair.get(key, []))
            violation_modules = int(sum_float(nested_path_rows, "num_violation_modules"))
            output.append(
                {
                    "candidate_id": cid,
                    "slope_column": slope,
                    "slope_value": slope_value,
                    "negative_slope": slope_value < 0,
                    "path_pairs": ";".join(f"{a:.4f}->{b:.4f}" for a, b in pairs),
                    "path_has_violation": violation_modules > 0,
                    "path_violation_modules": violation_modules,
                    "path_total_dimension_increase": sum_float(nested_path_rows, "total_dimension_increase"),
                    "path_max_dimension_increase": max_float(nested_path_rows, "max_dimension_increase"),
                    "path_changed_modules": sum_float(mask_path_rows, "num_changed_modules"),
                    "path_changed_head_modules": sum_float(mask_path_rows, "num_changed_head_modules"),
                    "path_changed_ffn_modules": sum_float(mask_path_rows, "num_changed_ffn_modules"),
                    "path_abs_dimension_change": sum_float(mask_path_rows, "total_abs_dimension_change"),
                    "path_removed_dimensions": sum_float(mask_path_rows, "total_removed_dimensions"),
                    "path_added_dimensions": sum_float(mask_path_rows, "total_added_dimensions"),
                    "path_changed_head_dimensions": sum_float(mask_path_rows, "total_changed_head_dimensions"),
                    "path_changed_ffn_dimensions": sum_float(mask_path_rows, "total_changed_ffn_dimensions"),
                    "violation_layers": ";".join(
                        sorted(
                            {
                                f"{v.get('module_type','')}@L{v.get('layer_index','')}:+{v.get('dimension_increase','')}"
                                for v in violation_path_rows
                            }
                        )
                    ),
                }
            )
    return output


def summary_rows(joined: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for slope in sorted({str(row["slope_column"]) for row in joined}):
        rows = [row for row in joined if row["slope_column"] == slope]
        neg = [row for row in rows if row["negative_slope"]]
        nonneg = [row for row in rows if not row["negative_slope"]]
        for label, subset in [("negative", neg), ("nonnegative", nonneg)]:
            values = {
                "slope_column": slope,
                "group": label,
                "n": len(subset),
                "fraction_with_violation": mean([1.0 if row["path_has_violation"] else 0.0 for row in subset]),
                "avg_violation_modules": mean([float(row["path_violation_modules"]) for row in subset]),
                "avg_added_dimensions": mean([float(row["path_added_dimensions"]) for row in subset]),
                "avg_removed_dimensions": mean([float(row["path_removed_dimensions"]) for row in subset]),
                "avg_abs_dimension_change": mean([float(row["path_abs_dimension_change"]) for row in subset]),
                "avg_changed_modules": mean([float(row["path_changed_modules"]) for row in subset]),
            }
            output.append(values)
    return output


def write_md(path: Path, joined: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    def fmt(value: object) -> str:
        number = to_float(value)
        if number is None:
            return str(value)
        return f"{number:.6g}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Local Delta Nestedness Audit\n\n")
        handle.write("This joins local PPL slopes with projection nestedness and mask-change diagnostics.\n\n")
        handle.write("## Summary\n\n")
        handle.write("| slope | group | n | frac violation | avg added dims | avg removed dims | avg abs dim change | avg changed modules |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in summary:
            handle.write(
                f"| {row['slope_column']} | {row['group']} | {row['n']} | {fmt(row['fraction_with_violation'])} | "
                f"{fmt(row['avg_added_dimensions'])} | {fmt(row['avg_removed_dimensions'])} | "
                f"{fmt(row['avg_abs_dimension_change'])} | {fmt(row['avg_changed_modules'])} |\n"
            )
        handle.write("\n## Negative-Slope Cases\n\n")
        handle.write("| slope | candidate_id | value | has violation | added dims | removed dims | violation layers |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in joined:
            if not row["negative_slope"]:
                continue
            handle.write(
                f"| {row['slope_column']} | {row['candidate_id']} | {fmt(row['slope_value'])} | "
                f"{row['path_has_violation']} | {fmt(row['path_added_dimensions'])} | "
                f"{fmt(row['path_removed_dimensions'])} | {row['violation_layers']} |\n"
            )


def main() -> int:
    args = parse_args()
    slope_columns = [part.strip() for part in args.slope_columns.split(",") if part.strip()]
    violation_rows = read_csv(Path(args.nestedness_violations)) if args.nestedness_violations else []
    joined = build_joined(
        read_csv(Path(args.local_delta_table)),
        read_csv(Path(args.nestedness_by_candidate)),
        read_csv(Path(args.mask_change_by_candidate)),
        violation_rows,
        slope_columns,
    )
    summary = summary_rows(joined)
    output_dir = Path(args.output_dir)
    joined_path = output_dir / "local_delta_nestedness_joined.csv"
    summary_path = output_dir / "local_delta_nestedness_summary.csv"
    md_path = output_dir / "local_delta_negative_cases.md"
    manifest_path = output_dir / "local_delta_nestedness_manifest.json"
    write_csv(joined_path, joined)
    write_csv(summary_path, summary)
    write_md(md_path, joined, summary)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "local_delta_table": args.local_delta_table,
            "nestedness_by_candidate": args.nestedness_by_candidate,
            "mask_change_by_candidate": args.mask_change_by_candidate,
            "nestedness_violations": args.nestedness_violations,
        },
        "slope_columns": slope_columns,
        "artifacts": {
            "local_delta_nestedness_joined": str(joined_path),
            "local_delta_nestedness_summary": str(summary_path),
            "local_delta_negative_cases": str(md_path),
            "local_delta_nestedness_manifest": str(manifest_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {joined_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
