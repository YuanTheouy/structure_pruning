#!/usr/bin/env python3
"""High-sample held-out recheck for already selected PAS candidates."""

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

from pas_evidence_pipeline import metadata_path_for_export, safe_id, write_csv, write_json


SKIP_KEYS = {"shortlist"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate the unique candidates selected by FF/PAS/Oracle at the held-out "
            "future budget. This does not change selection; it only rechecks selected "
            "candidates with more samples."
        )
    )
    parser.add_argument("--selected_candidates_json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--future_sparsity", type=float, default=0.40)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prune", default="para")
    parser.add_argument("--lbound", type=float, default=0.1)
    parser.add_argument("--rbound", type=float, default=1.0)
    return parser.parse_args()


def load_selected(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    by_candidate = {}
    for rule, item in payload.items():
        if rule in SKIP_KEYS or not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if not candidate_id and isinstance(item.get("probe_row"), dict):
            candidate_id = item["probe_row"].get("candidate_id")
        if not candidate_id:
            continue
        entry = by_candidate.setdefault(candidate_id, {"rules": [], "payload": item})
        entry["rules"].append(rule)
    return by_candidate


def build_compile_command(args, repo_root, best_path, export_path, final_policy_path):
    model_name = args.model_name or Path(args.model).name
    return [
        sys.executable,
        "-u",
        str(repo_root / "amc_searchPPO.py"),
        "--job=compile",
        f"--model={args.model}",
        f"--model_name={model_name}",
        f"--dataset_name={args.dataset}",
        f"--preserve_ratio={1.0 - args.future_sparsity:.6f}",
        f"--final_sparsity={args.future_sparsity}",
        f"--best_candidate_path={best_path}",
        f"--export_path={export_path}",
        f"--final_policy_path={final_policy_path}",
        "--structure",
        f"--prune={args.prune}",
        f"--lbound={args.lbound}",
        f"--rbound={args.rbound}",
        f"--n_samples={args.num_samples}",
        f"--data_bsize={args.batch_size}",
        f"--seed={args.seed}",
        "--enable_downstream=false",
    ]


def read_metadata(path):
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    ppl = float(metadata["ppl"])
    return metadata, math.log(ppl) if ppl > 0 else float("inf"), ppl


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_selected(Path(args.selected_candidates_json))
    if not selected:
        raise RuntimeError(f"No selected candidates found in {args.selected_candidates_json}")

    command_lines = []
    rows = []
    eval_root = output_dir / "heldout_recheck_eval"
    for candidate_id, entry in sorted(selected.items()):
        rules = entry["rules"]
        run_dir = eval_root / safe_id(candidate_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best_candidate.json"
        export_path = run_dir / "checkpoint.pth.tar"
        final_policy_path = run_dir / "final_policy.json"
        metadata_path = metadata_path_for_export(export_path)

        payload = dict(entry["payload"])
        payload["recheck_rules"] = rules
        payload["recheck_only"] = True
        payload["future_sparsity"] = args.future_sparsity
        write_json(best_path, payload)

        cmd = build_compile_command(args, repo_root, best_path, export_path, final_policy_path)
        command_lines.append(" ".join(shlex.quote(str(part)) for part in cmd))
        eval_seconds_h = ""
        if metadata_path.exists() and not args.force:
            pass
        elif args.dry_run:
            continue
        else:
            print(" ".join(shlex.quote(str(part)) for part in cmd))
            start_time = time.time()
            rc = subprocess.call(cmd, cwd=repo_root)
            eval_seconds_h = time.time() - start_time
            if rc != 0:
                raise SystemExit(rc)

        if metadata_path.exists():
            metadata, ell_h, ppl_h = read_metadata(metadata_path)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "rules": ";".join(rules),
                    "future_sparsity": args.future_sparsity,
                    "num_samples": args.num_samples,
                    "ell_h_recheck": ell_h,
                    "ppl_h_recheck": ppl_h,
                    "actual_sparsity_h": metadata.get("actual_sparsity", ""),
                    "budget_error_h": metadata.get("budget_error", ""),
                    "relative_budget_error_h": metadata.get("relative_budget_error", ""),
                    "checkpoint_path": str(export_path),
                    "metadata_path": str(metadata_path),
                    "eval_seconds_h": eval_seconds_h,
                }
            )

    commands_path = output_dir / "selected_heldout_recheck_commands.sh"
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\n")
        for line in command_lines:
            handle.write(line + "\n")

    if args.dry_run:
        print(f"Wrote dry-run commands to {commands_path}")
        return 0
    if not rows:
        raise RuntimeError("No recheck rows were produced.")

    csv_path = output_dir / "selected_heldout_recheck.csv"
    write_csv(csv_path, rows)

    best = min(rows, key=lambda row: (float(row["ell_h_recheck"]), row["candidate_id"]))
    regret_rows = []
    for row in rows:
        regret_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "rules": row["rules"],
                "ell_h_recheck": row["ell_h_recheck"],
                "ppl_h_recheck": row["ppl_h_recheck"],
                "best_rechecked_candidate_id": best["candidate_id"],
                "best_rechecked_ell_h": best["ell_h_recheck"],
                "regret_vs_best_rechecked": float(row["ell_h_recheck"]) - float(best["ell_h_recheck"]),
            }
        )
    regret_path = output_dir / "selected_heldout_recheck_regret.csv"
    write_csv(regret_path, regret_rows)

    manifest = {
        "selected_candidates_json": args.selected_candidates_json,
        "model": args.model,
        "model_name": args.model_name or Path(args.model).name,
        "dataset": args.dataset,
        "future_sparsity": args.future_sparsity,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "total_eval_seconds": sum(float(row["eval_seconds_h"]) for row in rows if row.get("eval_seconds_h") not in ("", None)),
        "artifacts": {
            "selected_heldout_recheck": str(csv_path),
            "selected_heldout_recheck_regret": str(regret_path),
            "commands": str(commands_path),
        },
    }
    write_json(output_dir / "selected_heldout_recheck_manifest.json", manifest)
    print(f"Wrote {csv_path}")
    print(f"Wrote {regret_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
