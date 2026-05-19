#!/usr/bin/env python3
"""Merge per-GPU P0 candidate pools into one rerankable candidate directory."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge FastForward candidate pool directories.")
    parser.add_argument("candidate_dirs", nargs="+", help="Input candidate directories containing candidates.jsonl.")
    parser.add_argument("--output_dir", required=True, help="Merged candidate directory.")
    parser.add_argument("--top_k", type=int, default=20, help="Keep the best K endpoint candidates; <=0 keeps all.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def endpoint_key(row: dict[str, Any]) -> tuple[float, int]:
    if row.get("endpoint_logppl") not in (None, ""):
        score = float(row["endpoint_logppl"])
    elif row.get("endpoint_ppl") not in (None, ""):
        ppl = float(row["endpoint_ppl"])
        score = math.log(ppl) if ppl > 0 else float("inf")
    else:
        score = float("inf")
    return score, int(row.get("step") or 0)


def copy_optional(src_value: str | None, dst_dir: Path, dst_name: str) -> str:
    if not src_value:
        return ""
    src = Path(src_value)
    if not src.exists():
        return src_value
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    return str(dst)


def maybe_update_policy(path_value: str, candidate_id: str) -> None:
    path = Path(path_value)
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["candidate_id"] = candidate_id
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        return


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    scores_dir = output_dir / "scores"
    policies_dir = output_dir / "policies"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, Any]] = []
    for source_idx, candidate_dir_value in enumerate(args.candidate_dirs):
        candidate_dir = Path(candidate_dir_value)
        rows = read_jsonl(candidate_dir / "candidates.jsonl")
        source_label = safe_id(candidate_dir.parent.name or f"src{source_idx}")
        for row_idx, row in enumerate(rows):
            original_id = str(row.get("candidate_id") or f"candidate_{row_idx:05d}")
            candidate_id = safe_id(f"{source_label}_{original_id}")
            item = dict(row)
            item["candidate_id"] = candidate_id
            item["source_candidate_id"] = original_id
            item["source_candidate_dir"] = str(candidate_dir)

            score_name = f"{candidate_id}.pt"
            policy_name = f"{candidate_id}.json"
            score_path = copy_optional(item.get("score_path"), scores_dir, score_name)
            policy_path = copy_optional(item.get("policy_path"), policies_dir, policy_name)
            if score_path:
                item["score_path"] = score_path
            if policy_path:
                item["policy_path"] = policy_path
                maybe_update_policy(policy_path, candidate_id)
            merged.append(item)

    merged.sort(key=endpoint_key)
    selected = merged if args.top_k <= 0 else merged[: args.top_k]
    write_jsonl(output_dir / "all_candidates.jsonl", merged)
    write_jsonl(output_dir / "candidates.jsonl", selected)

    print(f"=> Merged {len(merged)} candidates from {len(args.candidate_dirs)} pools")
    print(f"=> Wrote {len(selected)} selected candidates to {output_dir / 'candidates.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
