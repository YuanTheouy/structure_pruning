#!/usr/bin/env python3
"""Evaluate endpoint/slope/curvature selected candidates across sparsities."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate selected candidates at multiple sparsities.")
    parser.add_argument("--selected_candidates_json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.30, 0.35, 0.40])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--prune", default="para")
    parser.add_argument("--lbound", type=float, default=0.1)
    parser.add_argument("--rbound", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or Path(args.model).name

    with open(args.selected_candidates_json, "r", encoding="utf-8") as handle:
        selected = json.load(handle)

    rows = []
    for key, payload in selected.items():
        method = key.replace("_best", "")
        for sparsity in args.sparsities:
            run_dir = output_dir / method / f"sparsity_{sparsity:.2f}"
            run_dir.mkdir(parents=True, exist_ok=True)
            temp_best = run_dir / "best_candidate.json"
            with open(temp_best, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")

            export_path = run_dir / "checkpoint.pth.tar"
            cmd = [
                sys.executable,
                "-u",
                str(repo_root / "amc_searchPPO.py"),
                "--job=compile",
                f"--model={args.model}",
                f"--model_name={model_name}",
                f"--dataset_name={args.dataset}",
                f"--preserve_ratio={1.0 - sparsity:.6f}",
                f"--final_sparsity={sparsity}",
                f"--best_candidate_path={temp_best}",
                f"--export_path={export_path}",
                f"--final_policy_path={run_dir / 'final_policy.json'}",
                "--structure",
                f"--prune={args.prune}",
                f"--lbound={args.lbound}",
                f"--rbound={args.rbound}",
                f"--n_samples={args.num_samples}",
                f"--seed={args.seed}",
                "--enable_downstream=false",
            ]
            print(" ".join(cmd))
            rc = subprocess.call(cmd, cwd=repo_root)
            if rc != 0:
                raise SystemExit(rc)

            metadata_path = export_path.with_suffix(".json")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            rows.append({
                "method": method,
                "candidate_id": metadata.get("candidate_id"),
                "selected_score": payload.get("score", ""),
                "sparsity": sparsity,
                "actual_sparsity": metadata.get("actual_sparsity"),
                "ppl": metadata.get("ppl"),
                "logppl": math.log(float(metadata["ppl"])) if metadata.get("ppl") else "",
                "budget_error": metadata.get("budget_error"),
                "eval_time_sec": "",
                "checkpoint_path": str(export_path),
                "metadata_path": str(metadata_path),
            })

    results_csv = output_dir / "high_sparsity_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {results_csv}")


if __name__ == "__main__":
    main()
