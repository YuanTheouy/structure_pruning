#!/usr/bin/env python3
"""Compile selected PAS policies under one aligned final-evaluation setting."""

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compile/evaluate selected candidates from selected_candidates.json with identical "
            "settings. Use this for FF-Endpoint vs PAS-Slope compensation-aligned checks."
        )
    )
    parser.add_argument("--selected_candidates_json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rules", default="FF-Endpoint,PAS-Slope")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--target_sparsity", type=float, default=0.30)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_recon", action="store_true")
    parser.add_argument("--recon_sample", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prune", default="para")
    parser.add_argument("--lbound", type=float, default=0.1)
    parser.add_argument("--rbound", type=float, default=1.0)
    return parser.parse_args()


def safe_id(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


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


def metadata_path_for_export(export_path):
    return Path(str(export_path).rsplit(".", 1)[0] + ".json")


def build_compile_command(args, repo_root, best_path, export_path, final_policy_path):
    model_name = args.model_name or Path(args.model).name
    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "amc_searchPPO.py"),
        "--job=compile",
        f"--model={args.model}",
        f"--model_name={model_name}",
        f"--dataset_name={args.dataset}",
        f"--preserve_ratio={1.0 - args.target_sparsity:.6f}",
        f"--final_sparsity={args.target_sparsity}",
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
    if args.use_recon:
        cmd.extend(["--recon", f"--recon_sample={args.recon_sample}"])
    return cmd


def read_metadata(path):
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    ppl = float(metadata["ppl"])
    ell = math.log(ppl) if ppl > 0 else float("inf")
    return metadata, ell, ppl


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = Path(args.selected_candidates_json)
    with selected_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)

    rules = [rule.strip() for rule in args.rules.split(",") if rule.strip()]
    if not rules:
        raise RuntimeError("No rules requested.")

    rows = []
    command_lines = []
    eval_root = output_dir / "final_eval"
    for rule in rules:
        payload = selected.get(rule)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Rule {rule} not found in {selected_path}")
        candidate_id = payload.get("candidate_id") or payload.get("probe_row", {}).get("candidate_id")
        if not candidate_id:
            raise RuntimeError(f"Rule {rule} has no candidate_id")

        run_dir = eval_root / f"{safe_id(rule)}__{safe_id(candidate_id)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best_candidate.json"
        export_path = run_dir / "checkpoint.pth.tar"
        final_policy_path = run_dir / "final_policy.json"
        metadata_path = metadata_path_for_export(export_path)

        compile_payload = dict(payload)
        compile_payload["selected_mode"] = rule
        compile_payload["compensation_aligned_eval"] = True
        compile_payload["target_sparsity"] = args.target_sparsity
        compile_payload["recon_enabled"] = bool(args.use_recon)
        write_json(best_path, compile_payload)

        cmd = build_compile_command(args, repo_root, best_path, export_path, final_policy_path)
        command_lines.append(" ".join(shlex.quote(str(part)) for part in cmd))

        eval_seconds = ""
        if metadata_path.exists() and not args.force:
            pass
        elif args.dry_run:
            continue
        else:
            print(" ".join(shlex.quote(str(part)) for part in cmd))
            start_time = time.time()
            rc = subprocess.call(cmd, cwd=repo_root)
            eval_seconds = time.time() - start_time
            if rc != 0:
                raise SystemExit(rc)

        if metadata_path.exists():
            metadata, ell, ppl = read_metadata(metadata_path)
            rows.append(
                {
                    "rule": rule,
                    "candidate_id": candidate_id,
                    "target_sparsity": args.target_sparsity,
                    "ell": ell,
                    "ppl": ppl,
                    "actual_sparsity": metadata.get("actual_sparsity", ""),
                    "budget_error": metadata.get("budget_error", ""),
                    "relative_budget_error": metadata.get("relative_budget_error", ""),
                    "recon_enabled": bool(args.use_recon),
                    "recon_sample": args.recon_sample if args.use_recon else "",
                    "num_samples": args.num_samples,
                    "batch_size": args.batch_size,
                    "checkpoint_path": str(export_path),
                    "metadata_path": str(metadata_path),
                    "final_policy_path": str(final_policy_path),
                    "eval_seconds": eval_seconds,
                }
            )

    commands_path = output_dir / "pas_compensation_aligned_commands.sh"
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\n")
        for line in command_lines:
            handle.write(line + "\n")

    if args.dry_run:
        print(f"Wrote dry-run commands to {commands_path}")
        return 0
    if not rows:
        raise RuntimeError("No final-evaluation rows were produced.")

    best = min(rows, key=lambda row: (float(row["ell"]), row["rule"]))
    for row in rows:
        row["best_rule"] = best["rule"]
        row["best_ell"] = best["ell"]
        row["regret_vs_best"] = float(row["ell"]) - float(best["ell"])

    csv_path = output_dir / "pas_compensation_aligned_eval.csv"
    write_csv(csv_path, rows)

    manifest = {
        "selected_candidates_json": str(selected_path),
        "model": args.model,
        "model_name": args.model_name or Path(args.model).name,
        "dataset": args.dataset,
        "rules": rules,
        "target_sparsity": args.target_sparsity,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "recon_enabled": bool(args.use_recon),
        "recon_sample": args.recon_sample if args.use_recon else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "total_eval_seconds": sum(float(row["eval_seconds"]) for row in rows if row.get("eval_seconds") not in ("", None)),
        "artifacts": {
            "pas_compensation_aligned_eval": str(csv_path),
            "commands": str(commands_path),
        },
    }
    write_json(output_dir / "pas_compensation_aligned_manifest.json", manifest)
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'pas_compensation_aligned_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
