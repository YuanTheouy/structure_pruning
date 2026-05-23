#!/usr/bin/env python3
"""Materialize local-delta score tables from nested probe outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S3025/S3050/S31 local-delta tables from probe_results.csv files.")
    parser.add_argument("--delta0025-probe-results", required=True, help="Probe results for center=0.3025, delta=0.0025.")
    parser.add_argument("--delta0050-probe-results", default="", help="Optional probe results for center=0.3050, delta=0.0050.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seed", default="")
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


def to_float(value: object) -> float:
    return float(value)


def optional_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def from_delta0025(row: dict[str, str]) -> dict[str, object]:
    l30 = to_float(row["ell_minus"])
    l3025 = to_float(row["ell_0"])
    l3050 = to_float(row["ell_plus"])
    return {
        "candidate_id": row["candidate_id"],
        "L30": l30,
        "L3025": l3025,
        "L3050": l3050,
        "S3025": l3025 - l30,
        "S3050": l3050 - l30,
        "PPL30": row.get("ppl_minus", ""),
        "PPL3025": row.get("ppl_0", ""),
        "PPL3050": row.get("ppl_plus", ""),
        "projection_mode": row.get("projection_mode", ""),
        "projection_base_sparsity": row.get("projection_base_sparsity", ""),
    }


def from_delta0050(row: dict[str, str]) -> dict[str, object]:
    l30 = to_float(row["ell_minus"])
    l3050 = to_float(row["ell_0"])
    l31 = to_float(row["ell_plus"])
    return {
        "candidate_id": row["candidate_id"],
        "L30": l30,
        "L3050": l3050,
        "L31": l31,
        "S3050": l3050 - l30,
        "S31": l31 - l30,
        "PPL30": row.get("ppl_minus", ""),
        "PPL3050": row.get("ppl_0", ""),
        "PPL31": row.get("ppl_plus", ""),
        "projection_mode": row.get("projection_mode", ""),
        "projection_base_sparsity": row.get("projection_base_sparsity", ""),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    delta0025_path = Path(args.delta0025_probe_results)
    local_rows = [from_delta0025(row) for row in read_csv(delta0025_path)]
    local_by_id = {str(row["candidate_id"]): row for row in local_rows}

    local_csv = output_dir / "local_delta_scores.csv"
    write_csv(local_csv, local_rows)

    artifacts: dict[str, str] = {"local_delta_scores": str(local_csv)}
    merged_rows: list[dict[str, object]] = []
    if args.delta0050_probe_results:
        s31_rows = [from_delta0050(row) for row in read_csv(Path(args.delta0050_probe_results))]
        s31_csv = output_dir / "local_delta_scores_s31.csv"
        write_csv(s31_csv, s31_rows)
        artifacts["local_delta_scores_s31"] = str(s31_csv)

        for row in s31_rows:
            cid = str(row["candidate_id"])
            base = local_by_id.get(cid)
            if base is None:
                continue
            base_l30 = optional_float(base.get("L30"))
            base_l3050 = optional_float(base.get("L3050"))
            row_l30 = optional_float(row.get("L30"))
            row_l3050 = optional_float(row.get("L3050"))
            merged_rows.append(
                {
                    "candidate_id": cid,
                    "L30": row.get("L30"),
                    "L3025": base.get("L3025"),
                    "L3050": row.get("L3050"),
                    "L31": row.get("L31"),
                    "S3025": base.get("S3025"),
                    "S3050": row.get("S3050"),
                    "S31": row.get("S31"),
                    "PPL30": row.get("PPL30"),
                    "PPL3025": base.get("PPL3025"),
                    "PPL3050": row.get("PPL3050"),
                    "PPL31": row.get("PPL31"),
                    "projection_mode": row.get("projection_mode") or base.get("projection_mode", ""),
                    "projection_base_sparsity": row.get("projection_base_sparsity") or base.get("projection_base_sparsity", ""),
                    "L30_abs_diff_between_runs": abs(row_l30 - base_l30) if row_l30 is not None and base_l30 is not None else "",
                    "L3050_abs_diff_between_runs": abs(row_l3050 - base_l3050) if row_l3050 is not None and base_l3050 is not None else "",
                }
            )
        merged_csv = output_dir / "local_delta_scores_with_s31.csv"
        write_csv(merged_csv, merged_rows)
        artifacts["local_delta_scores_with_s31"] = str(merged_csv)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "inputs": {
            "delta0025_probe_results": str(delta0025_path),
            "delta0050_probe_results": args.delta0050_probe_results,
        },
        "candidate_count": len(local_rows),
        "merged_candidate_count": len(merged_rows) if args.delta0050_probe_results else "",
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "local_delta_materialization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {local_csv}")
    if args.delta0050_probe_results:
        print(f"Wrote {artifacts['local_delta_scores_s31']}")
        print(f"Wrote {artifacts['local_delta_scores_with_s31']}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
