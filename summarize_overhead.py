#!/usr/bin/env python3
"""Summarize PAS experiment overhead from available artifacts."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize PAS overhead artifacts.")
    parser.add_argument("--pas_dir", required=True, help="Directory containing PAS probe/held-out artifacts.")
    parser.add_argument("--recheck_dir", default=None, help="Optional selected-candidate recheck directory.")
    parser.add_argument("--final_eval_dir", default=None, help="Optional compensation-aligned final eval directory.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--search_gpu_hours", type=float, default=None)
    parser.add_argument("--search_wall_seconds", type=float, default=None)
    parser.add_argument("--search_gpu_count", type=float, default=None)
    parser.add_argument("--compensation_gpu_hours", type=float, default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def read_csv_rows(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not path or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sum_numeric(rows, key):
    total = 0.0
    count = 0
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            total += float(value)
            count += 1
        except (TypeError, ValueError):
            continue
    return total, count


def format_float(value):
    if value in ("", None):
        return ""
    return float(value)


def gpu_hours_from_seconds(seconds):
    if seconds in ("", None):
        return ""
    return float(seconds) / 3600.0


def make_row(stage, seconds="", gpu_hours="", status="measured", source="", notes=""):
    return {
        "stage": stage,
        "wall_seconds_available": format_float(seconds) if seconds not in ("", None) else "",
        "approx_gpu_hours": format_float(gpu_hours) if gpu_hours not in ("", None) else "",
        "gpu_hour_status": status,
        "artifact_source": source,
        "notes": notes,
    }


def main():
    args = parse_args()
    pas_dir = Path(args.pas_dir)
    output_dir = Path(args.output_dir) if args.output_dir else pas_dir
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "pas_overhead_summary.csv"
    output_json = Path(args.output_json) if args.output_json else output_dir / "pas_overhead_summary.json"

    probe_path = pas_dir / "probe_results.csv"
    heldout_path = pas_dir / "heldout_results.csv"
    manifest_path = pas_dir / "artifact_manifest.json"
    probe_rows = read_csv_rows(probe_path)
    heldout_rows = read_csv_rows(heldout_path)
    pas_manifest = read_json(manifest_path)

    rows = []
    if args.search_gpu_hours is not None:
        search_gpu_hours = args.search_gpu_hours
        search_status = "provided"
    elif args.search_wall_seconds is not None and args.search_gpu_count is not None:
        search_gpu_hours = args.search_wall_seconds * args.search_gpu_count / 3600.0
        search_status = "estimated_from_provided_wall_seconds_and_gpu_count"
    else:
        search_gpu_hours = ""
        search_status = "missing"
    rows.append(
        make_row(
            "candidate_search",
            seconds=args.search_wall_seconds if args.search_wall_seconds is not None else "",
            gpu_hours=search_gpu_hours,
            status=search_status,
            source=str(pas_manifest.get("candidate_dir", "")),
            notes="FastForward candidate generation; exact wall time not recorded unless provided.",
        )
    )

    probe_seconds, probe_count = sum_numeric(probe_rows, "eval_seconds_total")
    rows.append(
        make_row(
            "pas_probe_0.25_0.30_0.35",
            seconds=probe_seconds if probe_count else "",
            gpu_hours=gpu_hours_from_seconds(probe_seconds) if probe_count else "",
            status="measured_from_probe_rows" if probe_count else "missing",
            source=str(probe_path),
            notes=f"{probe_count} candidate probe timings; one GPU per probe process is assumed.",
        )
    )

    heldout_seconds, heldout_count = sum_numeric(heldout_rows, "eval_seconds_h")
    rows.append(
        make_row(
            "heldout_analysis_0.40",
            seconds=heldout_seconds if heldout_count else "",
            gpu_hours=gpu_hours_from_seconds(heldout_seconds) if heldout_count else "",
            status="measured_from_heldout_rows" if heldout_count else "missing_in_existing_artifacts",
            source=str(heldout_path),
            notes="0.40 held-out is analysis-only and not used for PAS selection.",
        )
    )

    if args.recheck_dir:
        recheck_dir = Path(args.recheck_dir)
        recheck_rows = read_csv_rows(recheck_dir / "selected_heldout_recheck.csv")
        recheck_manifest = read_json(recheck_dir / "selected_heldout_recheck_manifest.json")
        recheck_seconds, recheck_count = sum_numeric(recheck_rows, "eval_seconds_h")
        if not recheck_count and recheck_manifest.get("total_eval_seconds"):
            recheck_seconds = float(recheck_manifest["total_eval_seconds"])
            recheck_count = 1
        rows.append(
            make_row(
                "selected_candidate_recheck_0.40",
                seconds=recheck_seconds if recheck_count else "",
                gpu_hours=gpu_hours_from_seconds(recheck_seconds) if recheck_count else "",
                status="measured_from_recheck" if recheck_count else "missing_in_existing_artifacts",
                source=str(recheck_dir),
                notes="Only already selected candidates are re-evaluated; selection is unchanged.",
            )
        )

    if args.final_eval_dir:
        final_dir = Path(args.final_eval_dir)
        final_rows = read_csv_rows(final_dir / "pas_compensation_aligned_eval.csv")
        final_manifest = read_json(final_dir / "pas_compensation_aligned_manifest.json")
        final_seconds, final_count = sum_numeric(final_rows, "eval_seconds")
        if not final_count and final_manifest.get("total_eval_seconds"):
            final_seconds = float(final_manifest["total_eval_seconds"])
            final_count = 1
        if args.compensation_gpu_hours is not None:
            final_gpu_hours = args.compensation_gpu_hours
            status = "provided"
        else:
            final_gpu_hours = gpu_hours_from_seconds(final_seconds) if final_count else ""
            status = "measured_from_final_eval" if final_count else "missing_in_existing_artifacts"
        rows.append(
            make_row(
                "compensation_aligned_final_eval",
                seconds=final_seconds if final_count else "",
                gpu_hours=final_gpu_hours,
                status=status,
                source=str(final_dir),
                notes="FF-Endpoint and PAS-Slope must use identical recovery/calibration settings.",
            )
        )

    rows.append(
        make_row(
            "final_checkpoint_type",
            status="not_gpu_cost",
            source=str(pas_dir),
            notes="Static exact-budget pruned checkpoint.",
        )
    )
    rows.append(
        make_row(
            "inference_time_overhead",
            seconds=0.0,
            gpu_hours=0.0,
            status="by_design",
            source=str(pas_dir),
            notes="PAS is selection-time only; deployed model is a normal static checkpoint.",
        )
    )

    write_csv(output_csv, rows)
    manifest = {
        "pas_dir": str(pas_dir),
        "recheck_dir": args.recheck_dir,
        "final_eval_dir": args.final_eval_dir,
        "source_manifest": str(manifest_path),
        "artifacts": {
            "pas_overhead_summary_csv": str(output_csv),
            "pas_overhead_summary_json": str(output_json),
        },
        "notes": "Missing timings are explicit; provide search wall time/GPU count or rerun with timing-enabled scripts for measured GPU-hours.",
    }
    write_json(output_json, {"summary_rows": rows, "manifest": manifest})
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
