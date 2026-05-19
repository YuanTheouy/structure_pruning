"""Candidate persistence utilities for Early-Warning reranking."""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def safe_model_name(model_name: str | None) -> str:
    name = model_name or "model"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def to_jsonable(value: Any):
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump_json(path: str | os.PathLike, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def dump_yaml(path: str | os.PathLike, payload: Mapping[str, Any]) -> None:
    """Write a small YAML-compatible config without requiring PyYAML."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CandidateRecorder:
    """Write all candidates and maintain an endpoint top-K candidate set."""

    def __init__(
        self,
        root_dir: str | os.PathLike,
        *,
        top_k: int = 50,
        save_mode: str = "topk",
        save_every: int | None = None,
        run_config: Mapping[str, Any] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.top_k = int(top_k)
        self.save_mode = save_mode
        self.save_every = int(save_every) if save_every else None
        self.run_config = to_jsonable(run_config or {})
        self.scores_dir = self.root_dir / "scores"
        self.policies_dir = self.root_dir / "policies"
        self.all_jsonl = self.root_dir / "all_candidates.jsonl"
        self.top_jsonl = self.root_dir / "candidates.jsonl"
        self.periodic_jsonl = self.root_dir / "periodic_candidates.jsonl"
        self.config_path = self.root_dir / "run_config.json"
        self.config_yaml_path = self.root_dir / "config.yaml"
        if self.save_mode not in {"topk", "periodic", "topk_and_periodic"}:
            raise ValueError(f"Unsupported candidate save mode: {self.save_mode}")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.config_path, self.run_config)
        dump_yaml(self.config_yaml_path, self.run_config)

    def record(
        self,
        *,
        candidate_id: str,
        score_vector,
        projected_policy,
        current_sparsity: float,
        endpoint_ppl: float,
        endpoint_reward: float,
        step: int,
        seed: int | None,
        model_name: str | None,
        target_sparsity: float | None = None,
        actual_sparsity: float | None = None,
        budget_error: float | None = None,
        relative_budget_error: float | None = None,
        config: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict:
        score_path = self.scores_dir / f"{candidate_id}.pt"
        policy_path = self.policies_dir / f"{candidate_id}.json"

        score_tensor = torch.as_tensor(np.asarray(score_vector, dtype=np.float32))
        torch.save(score_tensor, score_path)

        policy_payload = {
            "candidate_id": candidate_id,
            "policy": to_jsonable(projected_policy),
            "current_sparsity": float(current_sparsity),
            "target_sparsity": float(target_sparsity if target_sparsity is not None else current_sparsity),
            "actual_sparsity": None if actual_sparsity is None else float(actual_sparsity),
            "budget_error": None if budget_error is None else float(budget_error),
            "relative_budget_error": None if relative_budget_error is None else float(relative_budget_error),
            "step": int(step),
            "seed": seed,
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        dump_json(policy_path, policy_payload)

        logppl = math.log(float(endpoint_ppl)) if endpoint_ppl and endpoint_ppl > 0 else float("inf")
        record = {
            "candidate_id": candidate_id,
            "score_path": str(score_path),
            "policy_path": str(policy_path),
            "current_sparsity": float(current_sparsity),
            "target_sparsity": float(target_sparsity if target_sparsity is not None else current_sparsity),
            "actual_sparsity": None if actual_sparsity is None else float(actual_sparsity),
            "budget_error": None if budget_error is None else float(budget_error),
            "relative_budget_error": None if relative_budget_error is None else float(relative_budget_error),
            "endpoint_ppl": float(endpoint_ppl),
            "endpoint_logppl": logppl,
            "endpoint_reward": float(endpoint_reward),
            "projected_policy": to_jsonable(projected_policy),
            "step": int(step),
            "seed": seed,
            "model_name": model_name,
            "config": to_jsonable(config or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record.update(to_jsonable(extra))

        with open(self.all_jsonl, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

        self._rewrite_selected()
        return record

    def _is_periodic(self, row: Mapping[str, Any]) -> bool:
        if not self.save_every or self.save_every <= 0:
            return False
        return int(row.get("step", 0)) % self.save_every == 0

    def _rewrite_selected(self) -> None:
        rows = read_jsonl(self.all_jsonl)
        endpoint_rows = sorted(
            rows,
            key=lambda row: (float(row.get("endpoint_logppl", float("inf"))), int(row.get("step", 0))),
        )
        top_rows = endpoint_rows if self.top_k <= 0 else endpoint_rows[: self.top_k]
        periodic_rows = [row for row in rows if self._is_periodic(row)]

        selected_by_id = {}
        if self.save_mode in {"topk", "topk_and_periodic"}:
            for row in top_rows:
                item = dict(row)
                item["selection_reason"] = "topk"
                selected_by_id[item["candidate_id"]] = item
        if self.save_mode in {"periodic", "topk_and_periodic"}:
            for row in periodic_rows:
                item = dict(row)
                item["selection_reason"] = (
                    "topk_and_periodic"
                    if item["candidate_id"] in selected_by_id
                    else "periodic"
                )
                selected_by_id[item["candidate_id"]] = item

        selected = sorted(
            selected_by_id.values(),
            key=lambda row: (int(row.get("step", 0)), float(row.get("endpoint_logppl", float("inf")))),
        )

        with open(self.top_jsonl, "w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        with open(self.periodic_jsonl, "w", encoding="utf-8") as handle:
            for row in periodic_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        keep_ids = {row["candidate_id"] for row in selected}
        top_scores_dir = self.root_dir / "top_scores"
        top_policies_dir = self.root_dir / "top_policies"
        for directory in (top_scores_dir, top_policies_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        for row in selected:
            score_src = Path(row["score_path"])
            policy_src = Path(row["policy_path"])
            if row["candidate_id"] in keep_ids and score_src.exists():
                shutil.copy2(score_src, top_scores_dir / score_src.name)
            if row["candidate_id"] in keep_ids and policy_src.exists():
                shutil.copy2(policy_src, top_policies_dir / policy_src.name)
