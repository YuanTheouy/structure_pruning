#!/usr/bin/env python3
"""Audit whether same-vector PAS budget projections are structurally nested."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amc_searchPPO import (  # noqa: E402
    build_env_module_costs,
    load_candidate_score,
    project_candidate_score_with_metadata,
)
from env.channel_pruning_env_llm_global import ChannelPruningEnv  # noqa: E402
from lib.ew_candidates import read_jsonl  # noqa: E402
from lib.ew_projector import compute_policy_budget  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether projected preserve dimensions are nested across sparsity budgets."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--sparsities", required=True, help="Comma/space separated sparsities, e.g. 0.30,0.31,0.35")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-top-k", type=int, default=0, help="Optional cap for fast debugging; 0 means all.")
    parser.add_argument("--projection-mode", choices=("current", "nested_from_base"), default="current")
    parser.add_argument("--base-sigma", type=float, default=None, help="Base sparsity for nested_from_base; default min sparsity.")
    parser.add_argument("--prune", default="para", choices=("para", "flops"))
    parser.add_argument("--lbound", type=float, default=0.1)
    parser.add_argument("--rbound", type=float, default=1.0)
    parser.add_argument("--channel-round", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=1, help="Env calibration sample count; no candidate eval is run.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_sparsities(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.replace(",", " ").split():
        values.append(float(part))
    values = sorted(set(values))
    if len(values) < 2:
        raise ValueError("--sparsities must contain at least two values")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def env_args(args: argparse.Namespace, preserve_ratio: float) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        model_name=args.model_name or Path(args.model).name,
        dataset_name=args.dataset,
        dataset_config_name=args.dataset_config_name,
        cache_dir=args.cache_dir,
        preserve_ratio=preserve_ratio,
        lbound=args.lbound,
        rbound=args.rbound,
        use_real_val=False,
        n_samples=args.n_samples,
        recon_sample=1,
        channel_round=args.channel_round,
        acc_metric="acc5",
        recon=False,
        recon_ffn_only=False,
        use_dataset_growth=False,
        dataset_initial_ratio=1.0,
        reward="reward_ppl",
        prune=args.prune,
        seed=int(args.seed),
        resume_path=None,
        start=0,
        delayed_downstream_eval=True,
        enable_downstream=False,
    )


def load_candidates(candidate_pool: Path, top_k: int = 0) -> list[dict[str, Any]]:
    path = candidate_pool / "candidates.jsonl"
    candidates = read_jsonl(path)
    if top_k and top_k > 0:
        candidates = candidates[:top_k]
    if not candidates:
        raise RuntimeError(f"No candidates found in {path}")
    return candidates


def module_info(index: int, dim: int) -> dict[str, Any]:
    return {
        "module_index": index,
        "module_type": "head" if index % 2 == 0 else "ffn",
        "layer_index": index // 2,
        "dim": dim,
    }


def policy_to_dims(policy: list[float], dim_list: list[int]) -> list[int]:
    dims: list[int] = []
    for ratio, dim in zip(policy, dim_list):
        d_prime = int(np.around(float(ratio) * int(dim)))
        d_prime = max(1, min(int(dim), d_prime))
        dims.append(d_prime)
    return dims


def extract_cost_bundle(module_costs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    costs = np.asarray(module_costs["costs"], dtype=np.float64)
    norm_raw = module_costs.get("norm_costs")
    norm_costs = np.zeros_like(costs) if norm_raw is None else np.asarray(norm_raw, dtype=np.float64)
    dim_list = np.asarray(module_costs["dim_list"], dtype=np.float64)
    channel_round = int(module_costs.get("channel_round", 1))
    cost_type = str(module_costs.get("cost_type", "para"))
    return costs, norm_costs, dim_list, channel_round, cost_type


def nested_project_score_to_policy(
    score_vector: np.ndarray,
    target_sparsity: float,
    module_costs: dict[str, Any],
    *,
    base_d_primes: list[int],
    p_min: float,
    p_max: float,
) -> dict[str, Any]:
    """Project with per-module caps from a base projection.

    This is intentionally a repair/ablation projector, not the current PAS
    projector. It mirrors the current budget correction while replacing the
    scalar upper bound with per-module base caps so stricter projections cannot
    preserve dimensions outside the base structure.
    """

    actions = np.abs(np.asarray(score_vector, dtype=np.float64))
    costs, norm_costs, dim_list, channel_round, cost_type = extract_cost_bundle(module_costs)
    if actions.shape != costs.shape:
        raise ValueError(f"score length {len(actions)} does not match module count {len(costs)}")

    target_sparsity = float(np.clip(target_sparsity, 0.0, 1.0))
    p_min = float(p_min)
    p_max = float(p_max)
    base_caps = np.asarray(base_d_primes, dtype=np.float64) / dim_list
    per_module_max = np.minimum(p_max, base_caps)
    actions = np.minimum(np.clip(actions, p_min, p_max), per_module_max)

    def cost_parts(idx: int) -> tuple[float, float]:
        if cost_type == "para":
            return float(costs[idx] - norm_costs[idx]), float(norm_costs[idx])
        return float(costs[idx]), 0.0

    def get_computation(ratios: np.ndarray | list[float]) -> float:
        total = 0.0
        for i, ratio in enumerate(ratios):
            effective_cost, fixed_cost = cost_parts(i)
            total += float(ratio) * effective_cost + fixed_cost
        return total

    target_computation = (1.0 - target_sparsity) * float(np.sum(costs))
    current_computation = get_computation(actions)

    if current_computation < target_computation:
        deficit = target_computation - current_computation
        unsaturated = [i for i, ratio in enumerate(actions) if ratio < per_module_max[i]]
        headrooms = []
        for idx in unsaturated:
            effective_cost, _ = cost_parts(idx)
            headrooms.append((per_module_max[idx] - actions[idx]) * effective_cost)
        total_headroom = float(np.sum(headrooms))
        if total_headroom > 1e-6:
            for local_idx, original_idx in enumerate(unsaturated):
                effective_cost, _ = cost_parts(original_idx)
                if effective_cost > 1e-6:
                    actions[original_idx] += deficit * (headrooms[local_idx] / total_headroom) / effective_cost
            actions = np.minimum(np.clip(actions, p_min, p_max), per_module_max)
    elif current_computation > target_computation:
        for idx in range(len(actions)):
            other_comp = 0.0
            this_comp = 0.0
            for i in range(len(actions)):
                effective_cost, fixed_cost = cost_parts(i)
                if i == idx:
                    this_comp += float(costs[i]) if cost_type == "para" else effective_cost
                elif i < idx:
                    other_comp += float(actions[i]) * effective_cost + fixed_cost
                else:
                    other_comp += p_min * effective_cost + fixed_cost
            if this_comp > 1e-6:
                max_preserve_ratio = (target_computation - other_comp) / this_comp
                actions[idx] = min(actions[idx], max_preserve_ratio)
                actions[idx] = max(actions[idx], p_min)
                actions[idx] = min(actions[idx], per_module_max[idx])

    d_primes = [max(1, int(np.around(ratio * dim))) for ratio, dim in zip(actions, dim_list)]
    if channel_round > 0:
        d_primes = [
            min(int(base_cap), int(dim), int(math.ceil(d_prime / channel_round) * channel_round))
            for d_prime, dim, base_cap in zip(d_primes, dim_list, base_d_primes)
        ]
    else:
        d_primes = [min(int(base_cap), int(dim), int(d_prime)) for d_prime, dim, base_cap in zip(d_primes, dim_list, base_d_primes)]

    rounded = np.asarray([d_prime / dim if dim > 0 else 0.0 for d_prime, dim in zip(d_primes, dim_list)], dtype=np.float64)
    overshoot = get_computation(rounded) - target_computation
    if overshoot > 0:
        for idx in range(len(rounded) - 1, -1, -1):
            if overshoot <= 0:
                break
            if dim_list[idx] <= 0 or channel_round <= 0:
                continue
            cost_per_channel = float(costs[idx] / dim_list[idx])
            cost_per_round_step = cost_per_channel * channel_round
            min_d_prime = max(1, int(np.around(p_min * dim_list[idx])))
            while d_primes[idx] > min_d_prime and overshoot > 0:
                d_primes[idx] -= channel_round
                overshoot -= cost_per_round_step

    projected = [d_prime / dim if dim > 0 else 0.0 for d_prime, dim in zip(d_primes, dim_list)]
    projected = list(np.minimum(np.clip(projected, p_min, p_max), per_module_max))
    budget = compute_policy_budget(projected, target_sparsity, module_costs)
    return {
        "policy": projected,
        "target_sparsity": target_sparsity,
        "actual_sparsity": budget["actual_sparsity"],
        "budget_error": budget["budget_error"],
        "relative_budget_error": budget["relative_budget_error"],
        "module_costs": module_costs,
        "p_min": p_min,
        "p_max": p_max,
        "metadata": {
            **budget,
            "projection_mode": "nested_from_base",
            "base_caps_enforced": True,
        },
    }


def build_projections(
    env: ChannelPruningEnv,
    score_vector: np.ndarray,
    sparsities: list[float],
    projection_mode: str,
    base_sigma: float,
    module_costs: dict[str, Any],
    dim_list: list[int],
) -> dict[float, dict[str, Any]]:
    current = {sigma: project_candidate_score_with_metadata(env, score_vector, sigma) for sigma in sparsities}
    if projection_mode == "current":
        return current

    if base_sigma not in current:
        current[base_sigma] = project_candidate_score_with_metadata(env, score_vector, base_sigma)
    base_d = policy_to_dims(current[base_sigma]["policy"], dim_list)
    nested: dict[float, dict[str, Any]] = {}
    for sigma in sparsities:
        if sigma <= base_sigma:
            nested[sigma] = current[sigma]
        else:
            nested[sigma] = nested_project_score_to_policy(
                score_vector,
                sigma,
                module_costs,
                base_d_primes=base_d,
                p_min=env.lbound,
                p_max=env.rbound,
            )
    return nested


def main() -> int:
    args = parse_args()
    sparsities = parse_sparsities(args.sparsities)
    base_sigma = args.base_sigma if args.base_sigma is not None else min(sparsities)
    candidate_pool = Path(args.candidate_pool)
    output_dir = Path(args.output_dir)
    model_name = args.model_name or Path(args.model).name

    if args.dry_run:
        print(f"Would load candidates from: {candidate_pool / 'candidates.jsonl'}")
        print(f"Would check sparsities: {sparsities}")
        print(f"Projection mode: {args.projection_mode}; base sigma: {base_sigma}")
        print(f"Would write outputs under: {output_dir}")
        return 0

    candidates = load_candidates(candidate_pool, args.candidate_top_k)
    preserve_ratio = 1.0 - min(sparsities)
    cargs = env_args(args, preserve_ratio)
    env = ChannelPruningEnv(
        args.model,
        args.dataset,
        preserve_ratio=preserve_ratio,
        n_data_worker=0,
        batch_size=args.batch_size,
        args=cargs,
        export_model=False,
        use_new_input=False,
    )
    module_costs = build_env_module_costs(env)
    dim_list = [int(dim) for dim in module_costs["dim_list"]]

    by_candidate_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []
    summary_acc: dict[tuple[float, float], dict[str, Any]] = {
        (a, b): {
            "candidate_pairs": 0,
            "pairs_with_violation": 0,
            "total_violation_modules": 0,
            "max_dimension_increase": 0,
            "max_relative_dimension_increase": 0.0,
        }
        for a, b in zip(sparsities[:-1], sparsities[1:])
    }

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", "")
        score_vector = load_candidate_score(candidate)
        projections = build_projections(
            env,
            score_vector,
            sparsities,
            args.projection_mode,
            base_sigma,
            module_costs,
            dim_list,
        )
        dims_by_sigma = {sigma: policy_to_dims(projections[sigma]["policy"], dim_list) for sigma in sparsities}

        for sigma_a, sigma_b in zip(sparsities[:-1], sparsities[1:]):
            dims_a = dims_by_sigma[sigma_a]
            dims_b = dims_by_sigma[sigma_b]
            proj_a = projections[sigma_a]
            proj_b = projections[sigma_b]
            pair_violations = []
            for module_index, (d_a, d_b, dim) in enumerate(zip(dims_a, dims_b, dim_list)):
                if d_b <= d_a:
                    continue
                increase = d_b - d_a
                rel_increase = increase / float(dim) if dim else 0.0
                info = module_info(module_index, dim)
                pair_violations.append(
                    {
                        "candidate_id": candidate_id,
                        "sigma_a": sigma_a,
                        "sigma_b": sigma_b,
                        **info,
                        "d_a": d_a,
                        "d_b": d_b,
                        "dimension_increase": increase,
                        "relative_dimension_increase": rel_increase,
                        "policy_a": proj_a["policy"][module_index],
                        "policy_b": proj_b["policy"][module_index],
                        "actual_sparsity_a": proj_a["actual_sparsity"],
                        "actual_sparsity_b": proj_b["actual_sparsity"],
                        "budget_error_a": proj_a["budget_error"],
                        "budget_error_b": proj_b["budget_error"],
                        "projection_mode": args.projection_mode,
                    }
                )
            violation_rows.extend(pair_violations)

            max_inc = max((row["dimension_increase"] for row in pair_violations), default=0)
            total_inc = sum(row["dimension_increase"] for row in pair_violations)
            by_candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "sigma_a": sigma_a,
                    "sigma_b": sigma_b,
                    "has_violation": bool(pair_violations),
                    "num_violation_modules": len(pair_violations),
                    "max_dimension_increase": max_inc,
                    "total_dimension_increase": total_inc,
                    "actual_sparsity_a": proj_a["actual_sparsity"],
                    "actual_sparsity_b": proj_b["actual_sparsity"],
                    "projection_mode": args.projection_mode,
                }
            )

            acc = summary_acc[(sigma_a, sigma_b)]
            acc["candidate_pairs"] += 1
            if pair_violations:
                acc["pairs_with_violation"] += 1
            acc["total_violation_modules"] += len(pair_violations)
            acc["max_dimension_increase"] = max(acc["max_dimension_increase"], max_inc)
            max_rel = max((row["relative_dimension_increase"] for row in pair_violations), default=0.0)
            acc["max_relative_dimension_increase"] = max(acc["max_relative_dimension_increase"], max_rel)

    summary_rows: list[dict[str, Any]] = []
    for sigma_a, sigma_b in zip(sparsities[:-1], sparsities[1:]):
        acc = summary_acc[(sigma_a, sigma_b)]
        summary_rows.append(
            {
                "model": model_name,
                "dataset": args.dataset,
                "seed": args.seed,
                "candidate_pool": str(candidate_pool),
                "num_candidates": len(candidates),
                "sigma_a": sigma_a,
                "sigma_b": sigma_b,
                "num_candidate_pairs": acc["candidate_pairs"],
                "num_pairs_with_violation": acc["pairs_with_violation"],
                "total_violation_modules": acc["total_violation_modules"],
                "max_dimension_increase": acc["max_dimension_increase"],
                "max_relative_dimension_increase": acc["max_relative_dimension_increase"],
                "all_pairs_nested": acc["total_violation_modules"] == 0,
                "projection_mode": args.projection_mode,
                "base_sigma": base_sigma,
            }
        )

    summary_path = output_dir / "nestedness_summary.csv"
    violations_path = output_dir / "nestedness_violations.csv"
    by_candidate_path = output_dir / "nestedness_by_candidate.csv"
    manifest_path = output_dir / "nestedness_manifest.json"
    write_csv(summary_path, summary_rows)
    write_csv(violations_path, violation_rows, fieldnames=[
        "candidate_id",
        "sigma_a",
        "sigma_b",
        "module_index",
        "module_type",
        "layer_index",
        "dim",
        "d_a",
        "d_b",
        "dimension_increase",
        "relative_dimension_increase",
        "policy_a",
        "policy_b",
        "actual_sparsity_a",
        "actual_sparsity_b",
        "budget_error_a",
        "budget_error_b",
        "projection_mode",
    ])
    write_csv(by_candidate_path, by_candidate_rows)
    write_json(
        manifest_path,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "model": args.model,
            "model_name": model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "candidate_pool": str(candidate_pool),
            "sparsities": sparsities,
            "projection_mode": args.projection_mode,
            "base_sigma": base_sigma,
            "candidate_count": len(candidates),
            "module_count": len(dim_list),
            "dim_list": dim_list,
            "channel_round": int(module_costs.get("channel_round", 1)),
            "cost_type": module_costs.get("cost_type", args.prune),
            "artifacts": {
                "nestedness_summary": str(summary_path),
                "nestedness_violations": str(violations_path),
                "nestedness_by_candidate": str(by_candidate_path),
                "nestedness_manifest": str(manifest_path),
            },
        },
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {violations_path}")
    print(f"Wrote {by_candidate_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
