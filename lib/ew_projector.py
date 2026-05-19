"""Reusable budget projection for Early-Warning candidate diagnostics."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def _as_float_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence, got shape {array.shape}")
    return array


def _extract_costs(module_costs):
    if isinstance(module_costs, Mapping):
        costs = _as_float_array(module_costs["costs"], "module_costs['costs']")
        norm_values = module_costs.get("norm_costs")
        if norm_values is None:
            norm_values = np.zeros_like(costs)
        norm_costs = _as_float_array(norm_values, "norm_costs")
        dim_list = module_costs.get("dim_list")
        channel_round = int(module_costs.get("channel_round", 1))
        cost_type = module_costs.get("cost_type", "para")
    else:
        costs = _as_float_array(module_costs, "module_costs")
        norm_costs = np.zeros_like(costs)
        dim_list = None
        channel_round = 1
        cost_type = "flops"

    if norm_costs.shape != costs.shape:
        raise ValueError("norm_costs must have the same length as costs")

    if dim_list is None:
        dim_array = None
    else:
        dim_array = _as_float_array(dim_list, "dim_list")
        if dim_array.shape != costs.shape:
            raise ValueError("dim_list must have the same length as costs")

    return costs, norm_costs, dim_array, channel_round, cost_type


def make_module_costs(
    costs: Sequence[float],
    *,
    norm_costs: Sequence[float] | None = None,
    dim_list: Sequence[float] | None = None,
    channel_round: int = 1,
    cost_type: str = "para",
) -> dict:
    """Build the module-cost bundle accepted by project_score_to_policy."""

    return {
        "costs": list(costs),
        "norm_costs": list(norm_costs) if norm_costs is not None else None,
        "dim_list": list(dim_list) if dim_list is not None else None,
        "channel_round": int(channel_round),
        "cost_type": cost_type,
    }


def compute_policy_budget(policy, target_sparsity, module_costs):
    """Compute actual sparsity and budget error for a projected policy."""

    ratios = _as_float_array(policy, "policy")
    costs, norm_costs, _, _, cost_type = _extract_costs(module_costs)
    if ratios.shape != costs.shape:
        raise ValueError(f"policy has length {len(ratios)}, but module_costs has length {len(costs)}")

    preserved = 0.0
    for idx, ratio in enumerate(ratios):
        if cost_type == "para":
            preserved += ratio * (costs[idx] - norm_costs[idx]) + norm_costs[idx]
        else:
            preserved += ratio * costs[idx]

    original = float(np.sum(costs))
    target_preserved = (1.0 - float(target_sparsity)) * original
    actual_sparsity = 1.0 - preserved / original if original > 0 else 0.0
    budget_error = preserved - target_preserved
    relative_budget_error = budget_error / original if original > 0 else 0.0
    return {
        "target_sparsity": float(target_sparsity),
        "actual_sparsity": float(actual_sparsity),
        "budget_error": float(budget_error),
        "relative_budget_error": float(relative_budget_error),
        "preserved_budget": float(preserved),
        "target_preserved_budget": float(target_preserved),
        "original_budget": original,
    }


def project_score_to_policy(
    score_vector,
    target_sparsity=None,
    module_costs=None,
    p_min=0.0,
    p_max=1.0,
    discretize=True,
    correct_budget=True,
    return_metadata=False,
    budget_tolerance=1e-3,
    sparsity=None,
):
    """Project a module-wise score vector to a global-budget pruning policy.

    The function mirrors the global LLM channel-pruning budget mapping used by
    ``ChannelPruningEnv._action_wall``. ``A`` is treated as the raw PPO score
    vector. ``sparsity`` is the global pruning sparsity, so the target preserve
    ratio is ``1 - sparsity``.

    ``module_costs`` can be a simple sequence of module costs, or a dictionary
    containing ``costs``, ``norm_costs``, ``dim_list``, ``channel_round`` and
    ``cost_type``. Passing the dictionary preserves the exact rounding behavior
    used by the structured FastForward environment.
    """

    if target_sparsity is None:
        if sparsity is None:
            raise ValueError("target_sparsity is required")
        target_sparsity = sparsity
    if module_costs is None:
        raise ValueError("module_costs is required")

    actions = np.abs(_as_float_array(score_vector, "score_vector"))
    costs, norm_costs, dim_list, channel_round, cost_type = _extract_costs(module_costs)

    if actions.shape != costs.shape:
        raise ValueError(f"A has length {len(actions)}, but module_costs has length {len(costs)}")

    p_min = float(p_min)
    p_max = float(p_max)
    target_sparsity = float(np.clip(target_sparsity, 0.0, 1.0))
    target_preserve = 1.0 - target_sparsity

    actions = np.clip(actions, p_min, p_max)

    def cost_parts(idx):
        if cost_type == "para":
            return costs[idx] - norm_costs[idx], norm_costs[idx]
        return costs[idx], 0.0

    def get_computation(ratios):
        total = 0.0
        for i, ratio in enumerate(ratios):
            effective_cost, fixed_cost = cost_parts(i)
            total += ratio * effective_cost + fixed_cost
        return total

    target_computation = target_preserve * float(np.sum(costs))
    current_computation = get_computation(actions)

    if correct_budget and current_computation < target_computation:
        deficit = target_computation - current_computation
        unsaturated = [i for i, ratio in enumerate(actions) if ratio < p_max]
        headrooms = []
        for idx in unsaturated:
            effective_cost, _ = cost_parts(idx)
            headrooms.append((p_max - actions[idx]) * effective_cost)

        total_headroom = float(np.sum(headrooms))
        if total_headroom > 1e-6:
            for local_idx, original_idx in enumerate(unsaturated):
                effective_cost, _ = cost_parts(original_idx)
                if effective_cost > 1e-6:
                    actions[original_idx] += deficit * (headrooms[local_idx] / total_headroom) / effective_cost
            actions = np.clip(actions, p_min, p_max)

    elif correct_budget and current_computation > target_computation:
        for idx in range(len(actions)):
            other_comp = 0.0
            this_comp = 0.0
            for i in range(len(actions)):
                effective_cost, fixed_cost = cost_parts(i)
                if i == idx:
                    this_comp += costs[i] if cost_type == "para" else effective_cost
                elif i < idx:
                    other_comp += actions[i] * effective_cost + fixed_cost
                else:
                    other_comp += p_min * effective_cost + fixed_cost

            if this_comp > 1e-6:
                max_preserve_ratio = (target_computation - other_comp) / this_comp
                actions[idx] = np.minimum(actions[idx], max_preserve_ratio)
                actions[idx] = np.maximum(actions[idx], p_min)

    if dim_list is None or not discretize:
        projected = list(np.clip(actions, p_min, p_max))
        budget = compute_policy_budget(projected, target_sparsity, module_costs)
        result = {
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
                "discretize": bool(discretize),
                "correct_budget": bool(correct_budget),
                "budget_tolerance": float(budget_tolerance),
                "within_budget_tolerance": abs(budget["relative_budget_error"]) <= budget_tolerance,
            },
        }
        return result if return_metadata else projected

    d_primes = [max(1, int(np.around(ratio * dim))) for ratio, dim in zip(actions, dim_list)]
    if channel_round > 0:
        d_primes = [
            min(int(dim), int(np.ceil(d_prime / channel_round) * channel_round))
            for d_prime, dim in zip(d_primes, dim_list)
        ]

    rounded = [d_prime / dim if dim > 0 else 0.0 for d_prime, dim in zip(d_primes, dim_list)]
    overshoot = get_computation(rounded) - target_computation

    if overshoot > 0:
        for idx in range(len(rounded) - 1, -1, -1):
            if overshoot <= 0:
                break
            if dim_list[idx] <= 0 or channel_round <= 0:
                continue

            cost_per_channel = costs[idx] / dim_list[idx]
            cost_per_round_step = cost_per_channel * channel_round
            min_d_prime = max(1, int(np.around(p_min * dim_list[idx])))

            while d_primes[idx] > min_d_prime and overshoot > 0:
                d_primes[idx] -= channel_round
                overshoot -= cost_per_round_step

    projected = [d_prime / dim if dim > 0 else 0.0 for d_prime, dim in zip(d_primes, dim_list)]
    projected = list(np.clip(projected, p_min, p_max))
    budget = compute_policy_budget(projected, target_sparsity, module_costs)
    result = {
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
            "discretize": bool(discretize),
            "correct_budget": bool(correct_budget),
            "budget_tolerance": float(budget_tolerance),
            "within_budget_tolerance": abs(budget["relative_budget_error"]) <= budget_tolerance,
        },
    }
    return result if return_metadata else projected
