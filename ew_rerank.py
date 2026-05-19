#!/usr/bin/env python3
"""Rerank Early-Warning probed candidates."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank candidates by endpoint, slope, or curvature.")
    parser.add_argument("--probe_results", required=True)
    parser.add_argument("--mode", choices=["endpoint", "slope", "curvature"], default="curvature")
    parser.add_argument("--lambda_ew", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidates_jsonl", default=None)
    return parser.parse_args()


def read_jsonl(path):
    rows = {}
    if not path or not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["candidate_id"]] = row
    return rows


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.probe_results, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = read_jsonl(args.candidates_jsonl)

    scored = []
    for row in rows:
        endpoint = float(row["logppl_zero"])
        slope = float(row["slope"])
        curvature = float(row["curvature"])
        warning_penalty = max(0.0, curvature - args.tau)
        endpoint_score = endpoint
        slope_score = endpoint + args.lambda_ew * slope
        curvature_score = endpoint + args.lambda_ew * warning_penalty
        selected_score = {
            "endpoint": endpoint_score,
            "slope": slope_score,
            "curvature": curvature_score,
        }[args.mode]
        out = dict(row)
        out.update({
            "endpoint_score": endpoint_score,
            "slope_score": slope_score,
            "curvature_score": curvature_score,
            "warning_penalty": warning_penalty,
            "selection_score": selected_score,
            "selected_mode": args.mode,
            "lambda_ew": args.lambda_ew,
            "tau": args.tau,
        })
        scored.append(out)

    scored.sort(key=lambda row: float(row["selection_score"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    if not scored:
        raise RuntimeError(f"No probe rows found in {args.probe_results}")

    rerank_csv = output_dir / "rerank_results.csv"
    with open(rerank_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    best = scored[0]
    best_payload = {
        "selected_mode": args.mode,
        "selection_score": float(best["selection_score"]),
        "probe_row": best,
        "candidate": candidates.get(best["candidate_id"], {}),
    }
    write_json(output_dir / "best_candidate.json", best_payload)

    selected_payload = {}
    for mode, score_key in (
        ("endpoint", "endpoint_score"),
        ("slope", "slope_score"),
        ("curvature", "curvature_score"),
    ):
        mode_best = min(scored, key=lambda row: float(row[score_key]))
        selected_payload[f"{mode}_best"] = {
            "mode": mode,
            "score": float(mode_best[score_key]),
            "probe_row": mode_best,
            "candidate": candidates.get(mode_best["candidate_id"], {}),
        }
    write_json(output_dir / "selected_candidates.json", selected_payload)

    print(f"Wrote {rerank_csv}")
    print(f"Wrote {output_dir / 'best_candidate.json'}")
    print(f"Wrote {output_dir / 'selected_candidates.json'}")
    print("Top 10:")
    for row in scored[:10]:
        print(f"  rank={row['rank']} id={row['candidate_id']} score={float(row['selection_score']):.6f}")


if __name__ == "__main__":
    main()
