#!/usr/bin/env python3
"""Resource-gated minimal P0 runner for Early-Warning candidate reranking."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import py_compile
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REQUIRED_SCRIPTS = (
    "ew_probe_candidates.py",
    "ew_rerank.py",
    "analyze_curvature_correlation.py",
    "amc_searchPPO.py",
)
REQUIRED_MODULES = (
    "numpy",
    "torch",
    "datasets",
    "transformers",
    "accelerate",
    "tensorboardX",
    "sklearn",
    "scipy",
    "cupy",
    "tqdm",
)
HF_WEIGHT_PATTERNS = (
    "*.safetensors",
    "model*.safetensors",
    "pytorch_model*.bin",
    "*.bin",
)
DEFAULT_CKPT_ROOT = "/workspace/ckpts"


def env_default(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check local resources, then run the minimal Early-Warning P0 "
            "probe/rerank only when candidate pool, model, WikiText-2, scripts, "
            "and CUDA are available."
        )
    )
    parser.add_argument("--candidate_dir", default=env_default("CANDIDATE_DIR"))
    parser.add_argument("--model", default=env_default("MODEL"))
    parser.add_argument("--model_name", default=env_default("MODEL_NAME"))
    parser.add_argument(
        "--sigma",
        "--target_sparsity",
        dest="sigma",
        type=float,
        default=float(env_default("TARGET_SPARSITY", env_default("SIGMA", "0.30"))),
        help="Warning sparsity. Default: 0.30.",
    )
    parser.add_argument("--delta", type=float, default=float(env_default("DELTA", "0.05")))
    parser.add_argument("--top_k", type=int, default=int(env_default("TOP_K", "20")))
    parser.add_argument("--dataset", default=env_default("DATASET", "wikitext2"))
    parser.add_argument(
        "--wikitext2_path",
        default=env_default("WIKITEXT2_PATH", "/workspace/datasets/wikitext/wikitext-2-raw-v1"),
        help="Local WikiText-2 path consumed through WIKITEXT2_PATH.",
    )
    parser.add_argument("--num_samples", type=int, default=int(env_default("N_SAMPLES", "32")))
    parser.add_argument("--batch_size", type=int, default=int(env_default("BATCH_SIZE", "50")))
    parser.add_argument("--seed", type=int, default=int(env_default("SEED", "2025")))
    parser.add_argument("--gpu_id", type=int, default=int(env_default("GPU_ID", "0")))
    parser.add_argument("--num_shards", type=int, default=int(env_default("NUM_SHARDS", "1")))
    parser.add_argument("--shard_id", type=int, default=int(env_default("SHARD_ID", "0")))
    parser.add_argument("--lambda_ew", type=float, default=float(env_default("LAMBDA_EW", "1.0")))
    parser.add_argument("--tau", type=float, default=float(env_default("TAU", "0.0")))
    parser.add_argument("--rerank_mode", choices=("endpoint", "slope", "curvature"), default="curvature")
    parser.add_argument("--output_dir", default=env_default("OUTPUT_DIR"))
    parser.add_argument("--ckpt_root", default=env_default("CKPT_ROOT", DEFAULT_CKPT_ROOT))
    parser.add_argument("--run_id", default=env_default("RUN_ID"))
    parser.add_argument("--train_episodes", type=int, default=int(env_default("TRAIN_EPISODES", "5000")))
    parser.add_argument("--check_only", action="store_true", help="Only inventory resources and print commands.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--force", action="store_true", help="Run commands even when blockers are detected.")
    parser.add_argument("--no_write_blockers", action="store_true", help="Do not write BLOCKERS.md.")
    parser.add_argument(
        "--allow_hf_model_id",
        action="store_true",
        help="Do not block when --model is not a local path.",
    )
    parser.add_argument(
        "--allow_dataset_download",
        action="store_true",
        help="Do not block when the configured local WikiText-2 path is missing.",
    )
    parser.add_argument(
        "--exit_zero_on_blockers",
        action="store_true",
        help="Return exit code 0 after reporting blockers.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def shell_join(cmd: list[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def resolve_path(path: str | os.PathLike[str] | None, root: Path) -> Path | None:
    if not path:
        return None
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else root / expanded


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_jsonl_head(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def discover_candidate_dir(root: Path) -> Path | None:
    candidates = []
    for path in root.rglob("candidates.jsonl"):
        if ".git" in path.parts:
            continue
        candidates.append(path.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.joinpath("candidates.jsonl").stat().st_mtime, reverse=True)
    return candidates[0]


def candidate_score_exists(candidate: dict[str, Any], candidate_dir: Path, root: Path) -> bool:
    if candidate.get("score_vector") is not None:
        return True
    score_path = candidate.get("score_path")
    if score_path:
        resolved = resolve_path(str(score_path), root)
        if resolved and resolved.exists():
            return True
    candidate_id = candidate.get("candidate_id")
    if candidate_id and (candidate_dir / "scores" / f"{candidate_id}.pt").exists():
        return True
    return False


def inspect_candidate_pool(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    discovered = None
    candidate_dir = resolve_path(args.candidate_dir, root)
    if candidate_dir is None:
        discovered = discover_candidate_dir(root)
        candidate_dir = discovered

    info: dict[str, Any] = {
        "requested": args.candidate_dir,
        "resolved": str(candidate_dir) if candidate_dir else None,
        "discovered": str(discovered) if discovered else None,
        "exists": bool(candidate_dir and candidate_dir.exists()),
        "top_k": args.top_k,
    }

    if candidate_dir is None:
        blockers.append("candidate pool missing: pass --candidate_dir or create a candidates.jsonl pool.")
        return info, blockers, warnings
    if not candidate_dir.exists():
        blockers.append(f"candidate pool directory does not exist: {candidate_dir}")
        return info, blockers, warnings

    candidates_jsonl = candidate_dir / "candidates.jsonl"
    all_candidates_jsonl = candidate_dir / "all_candidates.jsonl"
    scores_dir = candidate_dir / "scores"
    policies_dir = candidate_dir / "policies"
    info.update(
        {
            "candidates_jsonl": str(candidates_jsonl),
            "candidates_jsonl_exists": candidates_jsonl.exists(),
            "all_candidates_jsonl_exists": all_candidates_jsonl.exists(),
            "scores_dir_exists": scores_dir.exists(),
            "policies_dir_exists": policies_dir.exists(),
            "scores_count": len(list(scores_dir.glob("*.pt"))) if scores_dir.exists() else 0,
            "policies_count": len(list(policies_dir.glob("*.json"))) if policies_dir.exists() else 0,
        }
    )
    if not candidates_jsonl.exists():
        blockers.append(f"candidate pool file missing: {candidates_jsonl}")
        return info, blockers, warnings

    try:
        count = count_jsonl(candidates_jsonl)
        head = read_jsonl_head(candidates_jsonl, max(1, args.top_k))
    except Exception as exc:
        blockers.append(f"candidate pool is not readable JSONL: {candidates_jsonl} ({exc})")
        return info, blockers, warnings

    missing_score_ids = [
        str(row.get("candidate_id", f"row_{idx}"))
        for idx, row in enumerate(head)
        if not candidate_score_exists(row, candidate_dir, root)
    ]
    info.update(
        {
            "candidate_count": count,
            "head_checked": len(head),
            "missing_score_refs_in_head": missing_score_ids[:10],
        }
    )
    if count <= 0:
        blockers.append(f"candidate pool is empty: {candidates_jsonl}")
    if count < args.top_k:
        blockers.append(f"candidate pool has {count} candidates, fewer than top_k={args.top_k}.")
    if missing_score_ids:
        blockers.append(
            "candidate pool has missing score tensors or score_vector values for top candidates: "
            + ", ".join(missing_score_ids[:10])
        )
    return info, blockers, warnings


def inspect_model(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {"requested": args.model, "model_name": args.model_name}
    if not args.model:
        blockers.append("model weights missing: pass --model or set MODEL.")
        return info, blockers, warnings

    model_path = resolve_path(args.model, root)
    info.update({"resolved": str(model_path), "exists": bool(model_path and model_path.exists())})
    if model_path and model_path.exists():
        if model_path.is_dir():
            weights = []
            for pattern in HF_WEIGHT_PATTERNS:
                weights.extend(model_path.glob(pattern))
            info.update(
                {
                    "is_dir": True,
                    "config_json": str(model_path / "config.json"),
                    "config_json_exists": (model_path / "config.json").exists(),
                    "weight_file_count": len(weights),
                    "sample_weight_files": [str(path) for path in sorted(weights)[:5]],
                }
            )
            if not (model_path / "config.json").exists():
                blockers.append(f"model config missing: {model_path / 'config.json'}")
            if not weights:
                blockers.append(f"model weight files missing under: {model_path}")
        else:
            info.update({"is_dir": False, "file_size": model_path.stat().st_size})
    elif args.allow_hf_model_id:
        warnings.append(f"model path is not local; treating as HF model id because --allow_hf_model_id is set: {args.model}")
    else:
        blockers.append(f"model path does not exist locally: {model_path}")
    return info, blockers, warnings


def inspect_dataset(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    dataset_path = resolve_path(args.wikitext2_path, root)
    hf_cache = env_default("HF_DATASETS_CACHE")
    default_hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    info = {
        "dataset": args.dataset,
        "wikitext2_path": str(dataset_path) if dataset_path else None,
        "wikitext2_path_exists": bool(dataset_path and dataset_path.exists()),
        "HF_DATASETS_CACHE": hf_cache,
        "HF_DATASETS_CACHE_exists": bool(hf_cache and Path(hf_cache).expanduser().exists()),
        "default_hf_datasets_cache": str(default_hf_cache),
        "default_hf_datasets_cache_exists": default_hf_cache.exists(),
    }
    if "wikitext2" in args.dataset.lower():
        if not dataset_path or not dataset_path.exists():
            message = (
                "WikiText-2 local path missing. lib/data.py reads WIKITEXT2_PATH "
                f"or the configured --wikitext2_path; got {dataset_path}."
            )
            if args.allow_dataset_download:
                warnings.append(message + " Continuing because --allow_dataset_download is set.")
            else:
                blockers.append(message)
    return info, blockers, warnings


def inspect_scripts(root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    scripts: dict[str, Any] = {}
    for script in REQUIRED_SCRIPTS:
        path = root / script
        item: dict[str, Any] = {"path": str(path), "exists": path.exists(), "py_compile": None}
        if not path.exists():
            blockers.append(f"required script missing: {path}")
        else:
            try:
                cfile = Path(tempfile.gettempdir()) / "fastforward_pycompile" / f"{script}.pyc"
                cfile.parent.mkdir(parents=True, exist_ok=True)
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
                item["py_compile"] = "ok"
            except py_compile.PyCompileError as exc:
                item["py_compile"] = "failed"
                item["error"] = str(exc)
                blockers.append(f"script parse failed: {path} ({exc})")
            except Exception as exc:
                item["py_compile"] = "failed"
                item["error"] = repr(exc)
                blockers.append(f"script parse failed: {path} ({exc})")
        scripts[script] = item

    modules: dict[str, bool] = {}
    for module in REQUIRED_MODULES:
        modules[module] = importlib.util.find_spec(module) is not None
    missing = [module for module, found in modules.items() if not found]
    if missing:
        blockers.append("Python dependencies missing: " + ", ".join(missing))

    return {"scripts": scripts, "modules": modules}, blockers, warnings


def inspect_cuda() -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    code = """
import json
try:
    import torch
    payload = {
        "torch_import": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": torch.cuda.get_device_capability(idx),
            }
            for idx in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }
except Exception as exc:
    payload = {"torch_import": False, "error": repr(exc), "cuda_available": False, "device_count": 0, "devices": []}
print(json.dumps(payload))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            text=True,
            capture_output=True,
        )
        info = json.loads(proc.stdout.strip() or "{}")
        info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        if proc.returncode != 0:
            info["stderr"] = proc.stderr
    except Exception as exc:
        info = {"torch_import": False, "error": repr(exc), "cuda_available": False, "device_count": 0, "devices": []}

    if not info.get("torch_import"):
        blockers.append(f"torch import failed: {info.get('error', 'unknown error')}")
    elif not info.get("cuda_available"):
        blockers.append("CUDA/GPU unavailable: torch.cuda.is_available() is false.")
    elif int(info.get("device_count", 0)) <= 0:
        blockers.append("CUDA/GPU unavailable: torch sees zero devices.")
    return info, blockers, warnings


def build_probe_command(args: argparse.Namespace, root: Path, candidate_dir: Path, output_dir: Path) -> list[str]:
    model_name = args.model_name or (Path(args.model).name if args.model else "model")
    return [
        sys.executable,
        "-u",
        str(root / "ew_probe_candidates.py"),
        "--candidate_dir",
        str(candidate_dir),
        "--model",
        str(args.model),
        "--model_name",
        model_name,
        "--target_sparsity",
        f"{args.sigma:.2f}",
        "--delta",
        f"{args.delta:.2f}",
        "--top_k",
        str(args.top_k),
        "--dataset",
        args.dataset,
        "--num_samples",
        str(args.num_samples),
        "--batch_size",
        str(args.batch_size),
        "--output_dir",
        str(output_dir),
        "--num_shards",
        str(args.num_shards),
        "--shard_id",
        str(args.shard_id),
        "--gpu_id",
        str(args.gpu_id),
        "--seed",
        str(args.seed),
    ]


def build_rerank_command(args: argparse.Namespace, root: Path, candidate_dir: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(root / "ew_rerank.py"),
        "--probe_results",
        str(output_dir / "probe_results.csv"),
        "--mode",
        args.rerank_mode,
        "--lambda_ew",
        str(args.lambda_ew),
        "--tau",
        str(args.tau),
        "--output_dir",
        str(output_dir),
        "--candidates_jsonl",
        str(candidate_dir / "candidates.jsonl"),
    ]


def build_analysis_command(args: argparse.Namespace, root: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(root / "analyze_curvature_correlation.py"),
        "--probe_results",
        str(output_dir / "probe_results.csv"),
        "--warning_sparsity",
        f"{args.sigma:.2f}",
        "--future_sparsity",
        f"{args.sigma + args.delta:.2f}",
        "--output_dir",
        str(output_dir),
    ]


def build_self_command(args: argparse.Namespace, root: Path, candidate_dir: Path | None) -> list[str]:
    model_value = args.model or "/path/to/model"
    candidate_value = str(candidate_dir) if candidate_dir else "/path/to/candidates"
    model_name = args.model_name or (Path(model_value).name if model_value != "/path/to/model" else "MODEL_NAME")
    return [
        sys.executable,
        str(root / "ew_p0_minimal.py"),
        "--candidate_dir",
        candidate_value,
        "--model",
        model_value,
        "--model_name",
        model_name,
        "--sigma",
        f"{args.sigma:.2f}",
        "--delta",
        f"{args.delta:.2f}",
        "--top_k",
        str(args.top_k),
        "--dataset",
        args.dataset,
        "--wikitext2_path",
        args.wikitext2_path,
        "--num_samples",
        str(args.num_samples),
    ]


def build_candidate_generation_command(args: argparse.Namespace, root: Path, candidate_dir: Path | None) -> list[str]:
    model_value = args.model or "/path/to/model"
    model_name = args.model_name or (Path(model_value).name if model_value != "/path/to/model" else "MODEL_NAME")
    candidate_value = (
        str(candidate_dir)
        if candidate_dir
        else str(Path(args.ckpt_root) / model_name / f"sparsity_{args.sigma:.2f}" / "p0_candidates" / "candidates")
    )
    preserve_ratio = 1.0 - args.sigma
    return [
        sys.executable,
        "-u",
        str(root / "amc_searchPPO.py"),
        "--job=train",
        f"--model={model_value}",
        f"--model_name={model_name}",
        f"--dataset_name={args.dataset}",
        f"--preserve_ratio={preserve_ratio:.6f}",
        "--structure",
        "--prune=para",
        "--lbound=0.1",
        "--rbound=1.0",
        f"--n_samples={args.num_samples}",
        "--num_collect=15",
        "--learning_epoch=10",
        "--reward=reward_ppl",
        f"--train_episode={args.train_episodes}",
        f"--seed={args.seed}",
        f"--output={Path(candidate_value).parent / 'logs'}",
        f"--export_path={Path(candidate_value).parent / 'endpoint_best.pth.tar'}",
        "--enable_downstream=false",
        "--state_mode=0",
        "--save_candidates",
        "--candidate_save_mode=topk",
        f"--candidate_top_k={args.top_k}",
        f"--candidate_dir={candidate_value}",
        f"--run_id={args.run_id or 'p0_candidates'}",
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_commands(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(shell_join(command) + "\n")


def write_blockers_markdown(
    path: Path,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
    next_command: list[str],
    planned_commands: list[list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Early-Warning P0 Blockers\n\n")
        handle.write(f"- timestamp: {inventory['timestamp']}\n")
        handle.write(f"- model: {args.model or 'MISSING'}\n")
        handle.write(f"- model_name: {args.model_name or 'AUTO'}\n")
        handle.write(f"- dataset: {args.dataset}\n")
        handle.write(f"- sigma: {args.sigma:.2f}\n")
        handle.write(f"- delta: {args.delta:.2f}\n")
        handle.write(f"- top_k: {args.top_k}\n")
        handle.write(f"- num_samples: {args.num_samples}\n\n")
        handle.write("## Blockers\n\n")
        for item in blockers:
            handle.write(f"- {item}\n")
        if warnings:
            handle.write("\n## Warnings\n\n")
            for item in warnings:
                handle.write(f"- {item}\n")
        handle.write("\n## Next Minimal Command\n\n")
        handle.write("```bash\n")
        handle.write(shell_join(next_command) + "\n")
        handle.write("```\n\n")
        handle.write("## Planned Probe/Rerank Commands After Blockers Clear\n\n")
        handle.write("```bash\n")
        for command in planned_commands:
            handle.write(shell_join(command) + "\n")
        handle.write("```\n")


def choose_next_command(
    args: argparse.Namespace,
    root: Path,
    candidate_dir: Path | None,
    blockers: list[str],
) -> list[str]:
    candidate_missing = any("candidate pool" in blocker or "candidates" in blocker for blocker in blockers)
    only_candidate_missing = candidate_missing and not any(
        marker in blocker
        for blocker in blockers
        for marker in ("model", "WikiText-2", "CUDA", "dependencies", "script parse", "required script")
    )
    if only_candidate_missing:
        return build_candidate_generation_command(args, root, candidate_dir)
    return build_self_command(args, root, candidate_dir)


def run_command(command: list[str], root: Path, env: dict[str, str], dry_run: bool) -> None:
    print(shell_join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=root, env=env, check=True)


def read_csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = repo_root()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = args.run_id or time.strftime("ew_p0_%Y%m%d_%H%M%S", time.gmtime())

    candidate_info, candidate_blockers, candidate_warnings = inspect_candidate_pool(args, root)
    candidate_dir = resolve_path(candidate_info.get("resolved"), root)
    output_dir = resolve_path(args.output_dir, root)
    if output_dir is None:
        output_dir = (
            candidate_dir
            if candidate_dir and candidate_dir.exists()
            else Path(args.ckpt_root) / "ew_p0_minimal" / run_id
        )

    model_info, model_blockers, model_warnings = inspect_model(args, root)
    dataset_info, dataset_blockers, dataset_warnings = inspect_dataset(args, root)
    script_info, script_blockers, script_warnings = inspect_scripts(root)
    cuda_info, cuda_blockers, cuda_warnings = inspect_cuda()

    blockers = candidate_blockers + model_blockers + dataset_blockers + script_blockers + cuda_blockers
    warnings = candidate_warnings + model_warnings + dataset_warnings + script_warnings + cuda_warnings

    planned_commands: list[list[str]] = []
    if candidate_dir:
        planned_commands = [
            build_probe_command(args, root, candidate_dir, output_dir),
            build_rerank_command(args, root, candidate_dir, output_dir),
            build_analysis_command(args, root, output_dir),
        ]

    inventory = {
        "timestamp": timestamp,
        "repo_root": str(root),
        "parameters": {
            "model": args.model,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "wikitext2_path": args.wikitext2_path,
            "sigma": args.sigma,
            "delta": args.delta,
            "top_k": args.top_k,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "gpu_id": args.gpu_id,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "ckpt_root": args.ckpt_root,
        },
        "candidate_pool": candidate_info,
        "model": model_info,
        "dataset": dataset_info,
        "scripts": script_info,
        "cuda": cuda_info,
        "warnings": warnings,
        "blockers": blockers,
        "output_dir": str(output_dir),
        "planned_commands": [shell_join(command) for command in planned_commands],
    }

    write_json(output_dir / "p0_inventory.json", inventory)
    if planned_commands:
        write_commands(output_dir / "p0_commands.sh", planned_commands)

    print(f"Inventory written to {output_dir / 'p0_inventory.json'}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if blockers and not args.force:
        print("BLOCKERS:")
        for blocker in blockers:
            print(f"  - {blocker}")
        next_command = choose_next_command(args, root, candidate_dir, blockers)
        print("Next minimal command:")
        print(shell_join(next_command))
        if not args.no_write_blockers:
            write_blockers_markdown(
                output_dir / "BLOCKERS.md",
                args,
                inventory,
                blockers,
                warnings,
                next_command,
                planned_commands,
            )
            print(f"Blockers written to {output_dir / 'BLOCKERS.md'}")
        return 0 if args.exit_zero_on_blockers else 2

    if args.check_only:
        print("Resource check passed." if not blockers else "Resource check completed with --force.")
        print("Planned commands:")
        for command in planned_commands:
            print(shell_join(command))
        return 0

    env = os.environ.copy()
    env["WIKITEXT2_PATH"] = str(resolve_path(args.wikitext2_path, root) or args.wikitext2_path)
    env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    if not candidate_dir:
        raise RuntimeError("candidate_dir is required to run probe/rerank.")

    attempted: list[str] = []
    for command in planned_commands:
        attempted.append(shell_join(command))
        run_command(command, root, env, args.dry_run)

    artifacts = {
        "probe_results_csv": str(output_dir / "probe_results.csv"),
        "probe_results_jsonl": str(output_dir / "probe_results.jsonl"),
        "rerank_results_csv": str(output_dir / "rerank_results.csv"),
        "best_candidate_json": str(output_dir / "best_candidate.json"),
        "selected_candidates_json": str(output_dir / "selected_candidates.json"),
        "correlation_table_csv": str(output_dir / "correlation_table.csv"),
        "curvature_scatter_pdf": str(output_dir / "curvature_scatter.pdf"),
    }
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": inventory["parameters"],
        "commands_attempted": attempted,
        "dry_run": args.dry_run,
        "artifacts": artifacts,
        "probe_rows": 0 if args.dry_run else read_csv_count(output_dir / "probe_results.csv"),
    }
    write_json(output_dir / "p0_run_manifest.json", manifest)
    print(f"Run manifest written to {output_dir / 'p0_run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
