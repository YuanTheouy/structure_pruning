#!/usr/bin/env python3
"""Download the minimal model and WikiText-2 resources for P0 experiments."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen2.5-1.5B"
DEFAULT_MODEL_ID = "qwen/Qwen2.5-1.5B"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen2.5/WikiText-2 resources for Early-Warning P0.")
    parser.add_argument("--provider", choices=["modelscope", "huggingface", "direct"], default=os.environ.get("DOWNLOAD_PROVIDER", "modelscope"))
    parser.add_argument("--dataset_provider", choices=["modelscope", "huggingface"], default=os.environ.get("DATASET_PROVIDER", "modelscope"))
    parser.add_argument("--model_id", default=os.environ.get("MODEL_ID"))
    parser.add_argument("--modelscope_model_id", default=os.environ.get("MODELSCOPE_MODEL_ID"))
    parser.add_argument("--hf_model_id", default=os.environ.get("HF_MODEL_ID"))
    parser.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", f"/workspace/Models/{DEFAULT_MODEL_NAME}"))
    parser.add_argument("--hf_endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--dataset_id", default=os.environ.get("DATASET_ID", "modelscope/wikitext"))
    parser.add_argument("--hf_dataset_id", default=os.environ.get("HF_DATASET_ID", "Salesforce/wikitext"))
    parser.add_argument("--modelscope_dataset_id", default=os.environ.get("MODELSCOPE_DATASET_ID"))
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_dir", default="/workspace/datasets/wikitext/wikitext-2-raw-v1")
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", "/workspace/datasets/.cache/huggingface"))
    parser.add_argument("--modelscope_cache_dir", default=os.environ.get("MODELSCOPE_CACHE", "/workspace/datasets/.cache/modelscope"))
    parser.add_argument("--modelscope_backend", choices=["git", "cli", "sdk"], default=os.environ.get("MODELSCOPE_BACKEND", "git"))
    parser.add_argument("--modelscope_git_url", default=os.environ.get("MODELSCOPE_GIT_URL"))
    parser.add_argument("--direct_base_url", default=os.environ.get("DIRECT_MODEL_BASE_URL"))
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
        return args.hf_model_id or args.model_id or DEFAULT_MODEL_ID
    return args.modelscope_model_id or args.model_id or DEFAULT_MODEL_ID


def apply_hf_endpoint(args: argparse.Namespace) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def download_model_huggingface(args: argparse.Namespace) -> str:
    apply_hf_endpoint(args)
    from huggingface_hub import snapshot_download

    model_dir = Path(args.model_dir).expanduser()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_id = effective_model_id(args, "huggingface")
    print(f"=> Downloading Hugging Face model {model_id} to {model_dir}")
    if args.hf_endpoint:
        print(f"=> HF_ENDPOINT={args.hf_endpoint}")
    return snapshot_download(
        repo_id=model_id,
        revision=args.model_revision,
        local_dir=str(model_dir),
        cache_dir=args.cache_dir,
        token=args.token,
        endpoint=args.hf_endpoint,
        ignore_patterns=["*.h5", "*.msgpack", "*.ot"],
    )


def qwen25_direct_files(model_id: str) -> list[str]:
    if model_id not in {"qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"}:
        raise ValueError(
            "direct provider currently supports qwen/Qwen2.5-1.5B only; "
            "pass --provider modelscope or --provider huggingface for other models."
        )
    return [
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]


def direct_base_url_candidates(args: argparse.Namespace, model_id: str) -> list[str]:
    if args.direct_base_url:
        return [args.direct_base_url.rstrip("/")]
    if model_id in {"qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"}:
        return [
            "https://modelscope.cn/models/qwen/Qwen2.5-1.5B/resolve/master",
            "https://www.modelscope.cn/models/qwen/Qwen2.5-1.5B/resolve/master",
            "https://hf-mirror.com/Qwen/Qwen2.5-1.5B/resolve/main",
            "https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main",
        ]
    return []


def download_one_file(urls: list[str], destination: Path, *, chunk_size: int = 1024 * 1024) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and destination.stat().st_size > 0:
        print(f"=> Exists, skipping: {destination}", flush=True)
        return

    last_error: Exception | None = None
    for url in urls:
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        mode = "ab" if resume_from else "wb"
        try:
            print(f"=> Downloading {destination.name} from {url}", flush=True)
            response = requests.get(url, headers=headers, stream=True, timeout=(10, 120), allow_redirects=True)
            if resume_from and response.status_code == 200:
                resume_from = 0
                mode = "wb"
            response.raise_for_status()

            total_header = response.headers.get("Content-Range") or response.headers.get("Content-Length")
            total_size = None
            if total_header and "/" in total_header:
                total_size = int(total_header.rsplit("/", 1)[-1])
            elif total_header:
                total_size = int(total_header) + resume_from

            downloaded = resume_from
            last_print = time.monotonic()
            with partial.open(mode + "") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 2:
                        if total_size:
                            pct = downloaded / total_size * 100
                            print(f"=> {destination.name}: {downloaded / 2**20:.1f}/{total_size / 2**20:.1f} MiB ({pct:.1f}%)", flush=True)
                        else:
                            print(f"=> {destination.name}: {downloaded / 2**20:.1f} MiB", flush=True)
                        last_print = now
            partial.rename(destination)
            return
        except Exception as exc:
            last_error = exc
            print(f"=> Direct download failed for {url}: {exc}", flush=True)
    raise RuntimeError(f"all direct download attempts failed for {destination.name}") from last_error


def download_model_direct(args: argparse.Namespace) -> str:
    model_dir = Path(args.model_dir).expanduser()
    model_id = effective_model_id(args, "modelscope")
    files = qwen25_direct_files(model_id)
    base_urls = direct_base_url_candidates(args, model_id)
    if not base_urls:
        raise RuntimeError(f"no direct download URLs configured for {model_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"=> Direct file download for {model_id} to {model_dir}", flush=True)
    print("=> URL bases: " + ", ".join(base_urls), flush=True)
    for filename in files:
        urls = [f"{base}/{filename}" for base in base_urls]
        download_one_file(urls, model_dir / filename)
    return str(model_dir)


def download_model_modelscope(args: argparse.Namespace) -> str:
    if args.modelscope_backend == "git":
        return download_model_modelscope_git(args)
    if args.modelscope_backend == "cli":
        return download_model_modelscope_cli(args)
    return download_model_modelscope_sdk(args)


def run_streaming_command(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("=> " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    try:
        return_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    if return_code != 0:
        raise RuntimeError(f"command exited with code {return_code}: {' '.join(cmd)}")


def modelscope_git_url_candidates(args: argparse.Namespace, model_id: str) -> list[str]:
    if args.modelscope_git_url:
        return [args.modelscope_git_url]
    return [
        f"https://modelscope.cn/{model_id}.git",
        f"https://www.modelscope.cn/{model_id}.git",
        f"https://modelscope.cn/models/{model_id}.git",
        f"https://www.modelscope.cn/models/{model_id}.git",
    ]


def download_model_modelscope_git(args: argparse.Namespace) -> str:
    model_dir = Path(args.model_dir).expanduser()
    model_id = effective_model_id(args, "modelscope")
    repo_urls = modelscope_git_url_candidates(args, model_id)

    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is not available on PATH")
    has_lfs = subprocess.run(
        [git, "lfs", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not has_lfs:
        raise RuntimeError("git-lfs is required for ModelScope git downloads. Install it, then run: git lfs install")

    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    print("=> ModelScope hub: https://modelscope.cn", flush=True)
    print(f"=> Downloading ModelScope git repo {model_id} to {model_dir}", flush=True)
    print("=> URL candidates: " + ", ".join(repo_urls), flush=True)

    if model_dir.exists() and (model_dir / ".git").exists():
        run_streaming_command([git, "-C", str(model_dir), "fetch", "--all", "--prune"], env=os.environ.copy())
        if args.model_revision:
            run_streaming_command([git, "-C", str(model_dir), "checkout", args.model_revision], env=os.environ.copy())
        else:
            run_streaming_command([git, "-C", str(model_dir), "pull", "--ff-only"], env=os.environ.copy())
    elif is_nonempty_dir(model_dir):
        raise RuntimeError(
            f"{model_dir} already exists but is not a git checkout. Move it aside or pass --model_dir to a clean path."
        )
    else:
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        run_streaming_command([git, "lfs", "install"], env=os.environ.copy())
        last_error: Exception | None = None
        for index, repo_url in enumerate(repo_urls):
            tmp_dir = model_dir.parent / f".{model_dir.name}.tmp-modelscope-git-{os.getpid()}-{index}"
            try:
                run_streaming_command([git, "clone", "--progress", repo_url, str(tmp_dir)], env=env)
                if model_dir.exists():
                    model_dir.rmdir()
                tmp_dir.rename(model_dir)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"=> ModelScope git clone failed for {repo_url}: {exc}", flush=True)
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
        if last_error is not None:
            raise RuntimeError(f"all ModelScope git clone attempts failed for {model_id}") from last_error
        if args.model_revision:
            run_streaming_command([git, "-C", str(model_dir), "checkout", args.model_revision], env=os.environ.copy())

    run_streaming_command([git, "-C", str(model_dir), "lfs", "pull", "--exclude=*.h5,*.msgpack,*.ot"], env=os.environ.copy())
    return str(model_dir)


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

    print("=> ModelScope hub: https://modelscope.cn", flush=True)
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
    if args.provider == "direct":
        return download_model_direct(args)
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
    apply_hf_endpoint(args)
    from datasets import DatasetDict, load_dataset

    dataset_dir = Path(args.dataset_dir).expanduser()
    if is_nonempty_dir(dataset_dir):
        print(f"=> Dataset directory already exists and is non-empty, skipping: {dataset_dir}")
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = args.hf_dataset_id or args.dataset_id
    print(f"=> Downloading Hugging Face dataset {dataset_id}/{args.dataset_config} to {dataset_dir}")
    if args.hf_endpoint:
        print(f"=> HF_ENDPOINT={args.hf_endpoint}")
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
    from datasets import DatasetDict
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
        "direct_base_url": args.direct_base_url,
        "modelscope_backend": args.modelscope_backend,
        "modelscope_git_url": args.modelscope_git_url,
        "hf_endpoint": args.hf_endpoint,
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
