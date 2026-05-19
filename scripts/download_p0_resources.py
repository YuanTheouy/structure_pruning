#!/usr/bin/env python3
"""Download the minimal model and WikiText-2 resources for P0 experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OPT/WikiText-2 resources for Early-Warning P0.")
    parser.add_argument("--model_id", default="facebook/opt-2.7b")
    parser.add_argument("--model_dir", default="/workspace/Models/opt-2.7b")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--dataset_id", default="Salesforce/wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_dir", default="/workspace/datasets/wikitext/wikitext-2-raw-v1")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", "/workspace/datasets/.cache/huggingface"))
    parser.add_argument("--manifest_path", default="/workspace/ckpts/resource_manifest.json")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_dataset", action="store_true")
    return parser.parse_args()


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def download_model(args: argparse.Namespace) -> str:
    model_dir = Path(args.model_dir).expanduser()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"=> Downloading model {args.model_id} to {model_dir}")
    return snapshot_download(
        repo_id=args.model_id,
        revision=args.model_revision,
        local_dir=str(model_dir),
        cache_dir=args.cache_dir,
        token=args.token,
        ignore_patterns=["*.h5", "*.msgpack", "*.ot"],
    )


def download_wikitext2(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir).expanduser()
    if is_nonempty_dir(dataset_dir):
        print(f"=> Dataset directory already exists and is non-empty, skipping: {dataset_dir}")
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"=> Downloading dataset {args.dataset_id}/{args.dataset_config} to {dataset_dir}")
    dataset = DatasetDict(
        {
            split: load_dataset(
                args.dataset_id,
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


def main() -> int:
    args = parse_args()
    manifest = {
        "model_id": args.model_id,
        "model_dir": args.model_dir,
        "dataset_id": args.dataset_id,
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
