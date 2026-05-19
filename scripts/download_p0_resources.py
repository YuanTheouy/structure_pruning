#!/usr/bin/env python3
"""Download the minimal model and WikiText-2 resources for P0 experiments."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


DEFAULT_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODELSCOPE_MODEL_ID = "AI-ModelScope/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TinyLlama/WikiText-2 resources for Early-Warning P0.")
    parser.add_argument("--provider", choices=["modelscope", "huggingface"], default=os.environ.get("DOWNLOAD_PROVIDER", "modelscope"))
    parser.add_argument("--dataset_provider", choices=["modelscope", "huggingface"], default=os.environ.get("DATASET_PROVIDER", "modelscope"))
    parser.add_argument("--model_id", default=os.environ.get("MODEL_ID"))
    parser.add_argument("--modelscope_model_id", default=os.environ.get("MODELSCOPE_MODEL_ID"))
    parser.add_argument("--hf_model_id", default=os.environ.get("HF_MODEL_ID"))
    parser.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", f"/workspace/Models/{DEFAULT_MODEL_NAME}"))
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--dataset_id", default=os.environ.get("DATASET_ID", "modelscope/wikitext"))
    parser.add_argument("--hf_dataset_id", default=os.environ.get("HF_DATASET_ID", "Salesforce/wikitext"))
    parser.add_argument("--modelscope_dataset_id", default=os.environ.get("MODELSCOPE_DATASET_ID"))
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_dir", default="/workspace/datasets/wikitext/wikitext-2-raw-v1")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", "/workspace/datasets/.cache/huggingface"))
    parser.add_argument("--modelscope_cache_dir", default=os.environ.get("MODELSCOPE_CACHE", "/workspace/datasets/.cache/modelscope"))
    parser.add_argument("--modelscope_backend", choices=["cli", "sdk"], default=os.environ.get("MODELSCOPE_BACKEND", "cli"))
    parser.add_argument("--manifest_path", default="/workspace/ckpts/resource_manifest.json")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--modelscope_token", default=os.environ.get("MODELSCOPE_TOKEN"))
    parser.add_argument("--fallback_to_hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_dataset", action="store_true")
    return parser.parse_args()


def effective_model_id(args: argparse.Namespace, provider: str | None = None) -> str:
    provider = provider or args.provider
    if provider == "huggingface":
        return args.hf_model_id or args.model_id or DEFAULT_HF_MODEL_ID
    return args.modelscope_model_id or args.model_id or DEFAULT_MODELSCOPE_MODEL_ID


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def download_model_huggingface(args: argparse.Namespace) -> str:
    model_dir = Path(args.model_dir).expanduser()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_id = effective_model_id(args, "huggingface")
    print(f"=> Downloading Hugging Face model {model_id} to {model_dir}")
    return snapshot_download(
        repo_id=model_id,
        revision=args.model_revision,
        local_dir=str(model_dir),
        cache_dir=args.cache_dir,
        token=args.token,
        ignore_patterns=["*.h5", "*.msgpack", "*.ot"],
    )


def download_model_modelscope(args: argparse.Namespace) -> str:
    if args.modelscope_backend == "cli":
        return download_model_modelscope_cli(args)
    return download_model_modelscope_sdk(args)


def download_model_modelscope_cli(args: argparse.Namespace) -> str:
    model_dir = Path(args.model_dir).expanduser()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_id = effective_model_id(args, "modelscope")

    cli = shutil.which("modelscope")
    if not cli:
        raise RuntimeError("modelscope CLI is not available on PATH")

    cmd = [
        cli,
        "download",
        "--model",
        model_id,
        "--local_dir",
        str(model_dir),
        "--cache_dir",
        args.modelscope_cache_dir,
        "--exclude",
        "*.h5",
        "*.msgpack",
        "*.ot",
    ]
    if args.model_revision:
        cmd.extend(["--revision", args.model_revision])
    env = os.environ.copy()
    if args.modelscope_token:
        env["MODELSCOPE_TOKEN"] = args.modelscope_token

    print(f"=> Downloading ModelScope model {model_id} to {model_dir} with CLI progress", flush=True)
    print("=> " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    try:
        return_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    if return_code != 0:
        raise RuntimeError(f"modelscope CLI exited with code {return_code}")
    return str(model_dir)


def download_model_modelscope_sdk(args: argparse.Namespace) -> str:
    from modelscope import snapshot_download as ms_snapshot_download

    model_dir = Path(args.model_dir).expanduser()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_id = effective_model_id(args, "modelscope")
    print(f"=> Downloading ModelScope model {model_id} to {model_dir} with SDK", flush=True)
    kwargs = {
        "revision": args.model_revision,
        "local_dir": str(model_dir),
        "cache_dir": args.modelscope_cache_dir,
        "token": args.modelscope_token,
        "ignore_patterns": ["*.h5", "*.msgpack", "*.ot"],
        "ignore_file_pattern": ["*.h5", "*.msgpack", "*.ot"],
    }
    signature = inspect.signature(ms_snapshot_download)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return ms_snapshot_download(model_id, **kwargs)


def download_model(args: argparse.Namespace) -> str:
    if args.provider == "huggingface":
        return download_model_huggingface(args)
    try:
        return download_model_modelscope(args)
    except Exception as exc:
        if not args.fallback_to_hf:
            print(f"=> ModelScope model download failed: {exc}")
            print("=> Not falling back to Hugging Face. Use --fallback_to_hf to allow fallback.")
            print("=> If this is a ModelScope repo-id issue, pass --modelscope_model_id or set MODELSCOPE_MODEL_ID.")
            raise
        print(f"=> ModelScope model download failed: {exc}")
        print("=> Falling back to Hugging Face model download.")
        return download_model_huggingface(args)


def to_hf_dataset(dataset: Any):
    if hasattr(dataset, "to_hf_dataset"):
        return dataset.to_hf_dataset()
    return dataset


def download_wikitext2_huggingface(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir).expanduser()
    if is_nonempty_dir(dataset_dir):
        print(f"=> Dataset directory already exists and is non-empty, skipping: {dataset_dir}")
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = args.hf_dataset_id or args.dataset_id
    print(f"=> Downloading Hugging Face dataset {dataset_id}/{args.dataset_config} to {dataset_dir}")
    dataset = DatasetDict(
        {
            split: load_dataset(
                dataset_id,
                args.dataset_config,
                split=split,
                cache_dir=args.cache_dir,
                token=args.token,
                trust_remote_code=True,
            )
            for split in ("train", "validation", "test")
        }
    )
    dataset.save_to_disk(str(dataset_dir))
    return dataset_dir


def download_wikitext2_modelscope(args: argparse.Namespace) -> Path:
    from modelscope import MsDataset

    dataset_dir = Path(args.dataset_dir).expanduser()
    if is_nonempty_dir(dataset_dir):
        print(f"=> Dataset directory already exists and is non-empty, skipping: {dataset_dir}")
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = args.modelscope_dataset_id or args.dataset_id
    print(f"=> Downloading ModelScope dataset {dataset_id}/{args.dataset_config} to {dataset_dir}")
    dataset = DatasetDict(
        {
            split: to_hf_dataset(
                MsDataset.load(
                    dataset_id,
                    subset_name=args.dataset_config,
                    split=split,
                    cache_dir=args.modelscope_cache_dir,
                )
            )
            for split in ("train", "validation", "test")
        }
    )
    dataset.save_to_disk(str(dataset_dir))
    return dataset_dir


def download_wikitext2(args: argparse.Namespace) -> Path:
    if args.dataset_provider == "huggingface":
        return download_wikitext2_huggingface(args)
    try:
        return download_wikitext2_modelscope(args)
    except Exception as exc:
        if not args.fallback_to_hf:
            print(f"=> ModelScope dataset download failed: {exc}")
            print("=> Not falling back to Hugging Face. Use --fallback_to_hf to allow fallback.")
            print("=> If this is a ModelScope repo-id issue, pass --modelscope_dataset_id or set MODELSCOPE_DATASET_ID.")
            raise
        print(f"=> ModelScope dataset download failed: {exc}")
        print("=> Falling back to Hugging Face dataset download.")
        return download_wikitext2_huggingface(args)


def main() -> int:
    args = parse_args()
    effective_dataset_id = (
        (args.modelscope_dataset_id or args.dataset_id)
        if args.dataset_provider == "modelscope"
        else (args.hf_dataset_id or args.dataset_id)
    )
    manifest = {
        "provider": args.provider,
        "dataset_provider": args.dataset_provider,
        "model_id": effective_model_id(args),
        "model_dir": args.model_dir,
        "dataset_id": effective_dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_dir": args.dataset_dir,
    }

    if not args.skip_model:
        manifest["model_snapshot"] = download_model(args)
    if not args.skip_dataset:
        manifest["dataset_path"] = str(download_wikitext2(args))

    manifest_path = Path(args.manifest_path).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"=> Wrote {manifest_path}")
    print("=> Export these before running P0:")
    print(f"export MODEL={args.model_dir}")
    print(f"export MODEL_NAME={Path(args.model_dir).name}")
    print(f"export WIKITEXT2_PATH={args.dataset_dir}")
    print(f"export WIKITEXT2_CONFIG={args.dataset_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
