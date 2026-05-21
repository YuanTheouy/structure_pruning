#!/usr/bin/env python3
"""Collect downstream retention rows from task result CSV/JSON files."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--recovery-table", required=True)
    parser.add_argument("--task-results", nargs="*", default=[])
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


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("Would collect downstream task result files:")
        for path in args.task_results:
            print(f"  {path}")
        return 0

    recovery = {row["candidate_id"]: row for row in read_csv(Path(args.recovery_table))}
    rows: list[dict[str, object]] = []
    for result_path in args.task_results:
        path = Path(result_path)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_rows = payload.get("rows", [])
        else:
            raw_rows = read_csv(path)
        for raw in raw_rows:
            cid = raw.get("candidate_id")
            rec = recovery.get(cid, {})
            dense = float(raw.get("dense_score", "nan"))
            pruned = float(raw.get("pruned_score", "nan"))
            rows.append(
                {
                    "model": Path(args.model).name,
                    "seed": args.seed,
                    "candidate_id": cid,
                    "selection_tags": rec.get("selection_tags", ""),
                    "checkpoint_type": raw.get("checkpoint_type", "recovered"),
                    "task": raw.get("task", ""),
                    "dense_score": dense,
                    "pruned_score": pruned,
                    "retention": pruned / dense if dense else "",
                    "drop": dense - pruned,
                    "L30_raw": rec.get("L30_raw", ""),
                    "S35": rec.get("S35", ""),
                    "L30_recovered": rec.get("L30_recovered", ""),
                    "artifact_path": result_path,
                    "notes": raw.get("notes", ""),
                }
            )

    out_csv = Path(args.output_dir) / "downstream_retention_opt27b.csv"
    write_csv(out_csv, rows)
    out_md = Path(args.output_dir) / "downstream_retention_opt27b.md"
    out_md.write_text(f"# Downstream Retention\n\nRows: {len(rows)}\nCSV: `{out_csv}`\n", encoding="utf-8")
    manifest = {"model": args.model, "seed": args.seed, "task_results": args.task_results, "artifacts": {"downstream_retention": str(out_csv), "downstream_retention_md": str(out_md)}}
    with (Path(args.output_dir) / "downstream_manifest_opt27b.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
