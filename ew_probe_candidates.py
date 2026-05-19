#!/usr/bin/env python3
"""Standalone CLI for Early-Warning neighboring-budget candidate probes."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Probe FastForward candidates at sigma-delta/sigma/sigma+delta.")
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--target_sparsity", type=float, default=0.30)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--use_calibration", default="false", choices=["false", "true"])
    parser.add_argument("--prune", default="para")
    parser.add_argument("--lbound", type=float, default=0.1)
    parser.add_argument("--rbound", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    model_name = args.model_name or Path(args.model).name
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.candidate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "amc_searchPPO.py"),
        "--job=probe",
        f"--model={args.model}",
        f"--model_name={model_name}",
        f"--dataset_name={args.dataset}",
        f"--preserve_ratio={1.0 - args.target_sparsity:.6f}",
        "--structure",
        f"--prune={args.prune}",
        f"--lbound={args.lbound}",
        f"--rbound={args.rbound}",
        f"--n_samples={args.num_samples}",
        f"--data_bsize={args.batch_size}",
        f"--candidate_dir={args.candidate_dir}",
        f"--candidate_top_k={args.top_k}",
        f"--ew_delta={args.delta}",
        f"--probe_output={output_dir / 'probe_results.csv'}",
        f"--probe_jsonl_output={output_dir / 'probe_results.jsonl'}",
        f"--num_shards={args.num_shards}",
        f"--shard_id={args.shard_id}",
        f"--gpu_id={args.gpu_id}",
        f"--seed={args.seed}",
        "--enable_downstream=false",
    ]
    if args.use_calibration == "true":
        cmd.append("--recon")

    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())

