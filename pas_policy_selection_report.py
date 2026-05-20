#!/usr/bin/env python3
"""Build PAS policy-selection tables and figures from existing artifacts."""

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RULES = ["FF-Endpoint", "PAS-Plus", "PAS-Slope", "PAS-Curv", "Oracle-heldout"]
SELECTION_INPUTS = {
    "FF-Endpoint": "ell_sigma_only",
    "PAS-Plus": "ell_sigma_plus_delta_only",
    "PAS-Slope": "ell_sigma_and_ell_sigma_plus_delta_only",
    "PAS-Curv": "ell_sigma_minus_delta_sigma_plus_delta_only_ablation",
    "Oracle-heldout": "heldout_analysis_only_not_for_selection",
    "Random-shortlist": "random_from_endpoint_compatible_shortlist",
}


DEFAULT_POOLS = [
    {
        "pool_id": "opt27b_seed2025",
        "model": "opt-2.7b",
        "seed": "2025",
        "sigma": 0.30,
        "heldout_sigma": 0.40,
        "pas_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas",
        "recheck_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_selected_recheck64",
        "final_eval_dir": "",
    },
    {
        "pool_id": "opt27b_seed3025",
        "model": "opt-2.7b",
        "seed": "3025",
        "sigma": 0.30,
        "heldout_sigma": 0.40,
        "pas_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025",
        "recheck_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64",
        "final_eval_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon",
    },
    {
        "pool_id": "opt13b_seed2025",
        "model": "opt-1.3b",
        "seed": "2025",
        "sigma": 0.30,
        "heldout_sigma": 0.40,
        "pas_dir": "/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025",
        "recheck_dir": "",
        "final_eval_dir": "",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build policy-selection PAS reports.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pool_config",
        default=None,
        help="Optional JSON file with a list of pool configs. Defaults to known P0 artifacts.",
    )
    parser.add_argument("--top_m", default="2,3,5")
    parser.add_argument("--epsilon_logloss", default="0.02,0.05,0.10")
    return parser.parse_args()


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_table(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [str(row.get(col, "")).replace("\n", " ") for col in columns]
            handle.write("| " + " | ".join(values) + " |\n")


def as_float(row, key, default=None):
    value = row.get(key)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def by_key(rows, key):
    return {row.get(key): row for row in rows if row.get(key)}


def row_for_rule(rows, rule):
    for row in rows:
        if row.get("rule") == rule:
            return row
    return {}


def selected_rules_by_candidate(recheck_rows):
    mapping = {}
    for row in recheck_rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            continue
        rules = [rule.strip() for rule in row.get("rules", "").split(";") if rule.strip()]
        mapping[candidate_id] = {"row": row, "rules": rules}
    return mapping


def load_pool(pool):
    pas_dir = Path(pool["pas_dir"])
    recheck_dir = Path(pool.get("recheck_dir") or "/missing")
    final_eval_dir = Path(pool.get("final_eval_dir") or "/missing")
    probe_rows = read_csv_rows(pas_dir / "probe_results.csv")
    heldout_rows = read_csv_rows(pas_dir / "heldout_results.csv")
    selection_rows = read_csv_rows(pas_dir / "selection_regret.csv")
    recheck_rows = read_csv_rows(recheck_dir / "selected_heldout_recheck_regret.csv")
    final_rows = read_csv_rows(final_eval_dir / "pas_compensation_aligned_eval.csv")
    return {
        "probe": probe_rows,
        "probe_by_id": by_key(probe_rows, "candidate_id"),
        "heldout": heldout_rows,
        "heldout_by_id": by_key(heldout_rows, "candidate_id"),
        "selection": selection_rows,
        "recheck_by_id": selected_rules_by_candidate(recheck_rows),
        "final_by_candidate": by_key(final_rows, "candidate_id"),
        "pas_manifest": read_json(pas_dir / "artifact_manifest.json"),
    }


def target_values(candidate_id, data):
    final = data["final_by_candidate"].get(candidate_id, {})
    if final:
        return as_float(final, "ell"), as_float(final, "ppl"), "final_compile_eval"
    probe = data["probe_by_id"].get(candidate_id, {})
    if probe:
        return as_float(probe, "ell_0", as_float(probe, "logppl_zero")), as_float(probe, "ppl_0", as_float(probe, "ppl_zero")), "probe_ell_sigma"
    return None, None, "missing"


def heldout_values(candidate_id, data):
    recheck = data["recheck_by_id"].get(candidate_id, {}).get("row", {})
    if recheck:
        return as_float(recheck, "ell_h_recheck"), as_float(recheck, "ppl_h_recheck"), as_float(recheck, "regret_vs_best_rechecked"), "selected_recheck_64"
    heldout = data["heldout_by_id"].get(candidate_id, {})
    if heldout:
        return as_float(heldout, "ell_h"), as_float(heldout, "ppl_h"), None, "heldout_results"
    return None, None, None, "missing"


def build_tradeoff_rows(pool, data):
    rows = []
    ff_row = row_for_rule(data["selection"], "FF-Endpoint")
    ff_candidate = ff_row.get("candidate_id")
    ff_target_ell, _, ff_target_source = target_values(ff_candidate, data)
    ff_heldout_ell, _, _, ff_heldout_source = heldout_values(ff_candidate, data)

    for rule in RULES:
        selection = row_for_rule(data["selection"], rule)
        candidate_id = selection.get("candidate_id")
        if not candidate_id and rule == "Oracle-heldout":
            candidate_id = row_for_rule(data["selection"], "Oracle-heldout").get("candidate_id")

        target_ell, target_ppl, target_source = target_values(candidate_id, data)
        heldout_ell, heldout_ppl, recheck_regret, heldout_source = heldout_values(candidate_id, data)
        if recheck_regret is None:
            recheck_regret = as_float(selection, "regret")

        delta_target = target_ell - ff_target_ell if target_ell is not None and ff_target_ell is not None else ""
        delta_heldout = heldout_ell - ff_heldout_ell if heldout_ell is not None and ff_heldout_ell is not None else ""
        rows.append(
            {
                "model": pool["model"],
                "seed": pool["seed"],
                "sigma": pool["sigma"],
                "heldout_sigma": pool["heldout_sigma"],
                "rule": rule,
                "selected_candidate": candidate_id or "",
                "target_ell": target_ell if target_ell is not None else "",
                "target_ppl": target_ppl if target_ppl is not None else "",
                "target_regret": delta_target,
                "heldout_ell": heldout_ell if heldout_ell is not None else "",
                "heldout_ppl": heldout_ppl if heldout_ppl is not None else "",
                "heldout_regret": recheck_regret if recheck_regret is not None else "",
                "delta_target_ell_vs_endpoint": delta_target,
                "delta_heldout_ell_vs_endpoint": delta_heldout,
                "selection_inputs_used": SELECTION_INPUTS.get(rule, ""),
                "artifact_source": f"{pool['pas_dir']}; target={target_source}; heldout={heldout_source}",
            }
        )

    random_row = row_for_rule(data["selection"], "Random-shortlist")
    if random_row:
        rows.append(
            {
                "model": pool["model"],
                "seed": pool["seed"],
                "sigma": pool["sigma"],
                "heldout_sigma": pool["heldout_sigma"],
                "rule": "Random-shortlist",
                "selected_candidate": "random_from_shortlist_distribution",
                "target_ell": "",
                "target_ppl": "",
                "target_regret": "",
                "heldout_ell": "",
                "heldout_ppl": "",
                "heldout_regret": random_row.get("regret_mean", ""),
                "delta_target_ell_vs_endpoint": "",
                "delta_heldout_ell_vs_endpoint": "",
                "selection_inputs_used": SELECTION_INPUTS["Random-shortlist"],
                "artifact_source": str(Path(pool["pas_dir"]) / "selection_regret.csv"),
            }
        )
    return rows


def choose_min(rows, key):
    usable = [row for row in rows if as_float(row, key) is not None]
    if not usable:
        return {}
    return min(usable, key=lambda row: (as_float(row, key), row.get("candidate_id", "")))


def build_shortlist(probe_rows, shortlist_type, value):
    ordered = sorted(probe_rows, key=lambda row: (as_float(row, "ell_0", math.inf), row.get("candidate_id", "")))
    if shortlist_type == "top_m":
        return ordered[: int(value)]
    best = as_float(ordered[0], "ell_0", 0.0) if ordered else 0.0
    return [row for row in ordered if as_float(row, "ell_0", math.inf) <= best + float(value)]


def build_sensitivity_rows(pool, data, top_m_values, epsilon_values):
    rows = []
    probe_rows = data["probe"]
    heldout_by_id = data["heldout_by_id"]
    if not probe_rows or not heldout_by_id:
        return rows

    ff = choose_min(probe_rows, "ell_0")
    ff_candidate = ff.get("candidate_id")
    ff_target = as_float(ff, "ell_0")
    ff_heldout = as_float(heldout_by_id.get(ff_candidate, {}), "ell_h")
    oracle = choose_min(data["heldout"], "ell_h")
    oracle_heldout = as_float(oracle, "ell_h")

    shortlist_specs = [("top_m", value) for value in top_m_values]
    shortlist_specs.extend(("epsilon_logloss", value) for value in epsilon_values)
    for shortlist_type, value in shortlist_specs:
        shortlist = build_shortlist(probe_rows, shortlist_type, value)
        if not shortlist:
            continue
        candidates = {
            "FF-Endpoint": choose_min(probe_rows, "ell_0"),
            "PAS-Plus": choose_min(shortlist, "ell_plus"),
            "PAS-Slope": choose_min(shortlist, "slope"),
            "PAS-Curv": choose_min(shortlist, "curvature"),
            "Oracle-heldout": oracle,
        }
        for rule, selected in candidates.items():
            candidate_id = selected.get("candidate_id")
            target_ell = as_float(data["probe_by_id"].get(candidate_id, {}), "ell_0")
            heldout_ell = as_float(heldout_by_id.get(candidate_id, {}), "ell_h")
            target_regret = target_ell - ff_target if target_ell is not None and ff_target is not None else ""
            heldout_regret = heldout_ell - oracle_heldout if heldout_ell is not None and oracle_heldout is not None else ""
            rows.append(
                {
                    "model": pool["model"],
                    "seed": pool["seed"],
                    "shortlist_type": shortlist_type,
                    "shortlist_value": value,
                    "shortlist_size": len(shortlist),
                    "rule": rule,
                    "selected_candidate": candidate_id or "",
                    "target_ell": target_ell if target_ell is not None else "",
                    "heldout_ell": heldout_ell if heldout_ell is not None else "",
                    "target_regret": target_regret,
                    "heldout_regret": heldout_regret,
                    "endpoint_cost_vs_FF": target_regret,
                    "heldout_gain_vs_FF": ff_heldout - heldout_ell if ff_heldout is not None and heldout_ell is not None else "",
                }
            )
    return rows


def selected_candidate_ids(data):
    ids = {}
    for rule in ["FF-Endpoint", "PAS-Slope", "Oracle-heldout"]:
        row = row_for_rule(data["selection"], rule)
        if row.get("candidate_id"):
            ids[rule] = row["candidate_id"]
    return ids


def plot_pool_figures(pool, data, output_dir):
    pool_dir = output_dir / pool["pool_id"]
    pool_dir.mkdir(parents=True, exist_ok=True)
    probe_rows = data["probe"]
    heldout_by_id = data["heldout_by_id"]
    selected_ids = selected_candidate_ids(data)
    if not probe_rows or not heldout_by_id:
        return []

    artifacts = []
    xs = []
    ys = []
    labels = []
    for row in probe_rows:
        candidate_id = row.get("candidate_id")
        heldout = heldout_by_id.get(candidate_id, {})
        x = as_float(row, "ell_0")
        y = as_float(heldout, "ell_h")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        labels.append(candidate_id)

    fig_path = pool_dir / "endpoint_ambiguity_scatter.pdf"
    plt.figure(figsize=(5.8, 4.2))
    plt.scatter(xs, ys, alpha=0.65, label="candidate")
    for rule, candidate_id in selected_ids.items():
        if candidate_id in labels:
            idx = labels.index(candidate_id)
            plt.scatter([xs[idx]], [ys[idx]], s=80, label=rule)
            plt.annotate(rule, (xs[idx], ys[idx]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("ell(sigma)")
    plt.ylabel("ell(held-out stricter sigma)")
    plt.title(f"{pool['model']} seed {pool['seed']}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    artifacts.append(str(fig_path))

    fig_path = pool_dir / "policy_path_lines.pdf"
    plt.figure(figsize=(6.2, 4.0))
    xlabels = ["sigma-delta", "sigma", "sigma+delta", "heldout"]
    for rule, candidate_id in selected_ids.items():
        probe = data["probe_by_id"].get(candidate_id, {})
        heldout = heldout_by_id.get(candidate_id, {})
        yvals = [
            as_float(probe, "ell_minus"),
            as_float(probe, "ell_0"),
            as_float(probe, "ell_plus"),
            as_float(heldout, "ell_h"),
        ]
        if any(value is None for value in yvals):
            continue
        plt.plot(xlabels, yvals, marker="o", label=rule)
    plt.ylabel("log-PPL")
    plt.title(f"Same-vector policy paths: {pool['model']} seed {pool['seed']}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    artifacts.append(str(fig_path))

    fig_path = pool_dir / "target_future_tradeoff.pdf"
    ff_row = row_for_rule(data["selection"], "FF-Endpoint")
    ff_candidate = ff_row.get("candidate_id")
    ff_target = as_float(data["probe_by_id"].get(ff_candidate, {}), "ell_0")
    ff_heldout = as_float(heldout_by_id.get(ff_candidate, {}), "ell_h")
    plt.figure(figsize=(5.6, 4.0))
    for rule in ["FF-Endpoint", "PAS-Plus", "PAS-Slope", "PAS-Curv", "Oracle-heldout"]:
        row = row_for_rule(data["selection"], rule)
        candidate_id = row.get("candidate_id")
        target = as_float(data["probe_by_id"].get(candidate_id, {}), "ell_0")
        heldout = as_float(heldout_by_id.get(candidate_id, {}), "ell_h")
        if None in (target, heldout, ff_target, ff_heldout):
            continue
        plt.scatter([target - ff_target], [ff_heldout - heldout], s=70)
        plt.annotate(rule, (target - ff_target, ff_heldout - heldout), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("endpoint cost vs FF")
    plt.ylabel("held-out gain vs FF")
    plt.title(f"Target/future tradeoff: {pool['model']} seed {pool['seed']}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    artifacts.append(str(fig_path))
    return artifacts


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pool_config:
        pools = read_json(args.pool_config)
    else:
        pools = DEFAULT_POOLS
    top_m_values = [int(value) for value in args.top_m.split(",") if value.strip()]
    epsilon_values = [float(value) for value in args.epsilon_logloss.split(",") if value.strip()]

    tradeoff_rows = []
    sensitivity_rows = []
    figure_artifacts = []
    manifest_pools = []
    for pool in pools:
        data = load_pool(pool)
        if not data["selection"]:
            manifest_pools.append({"pool_id": pool["pool_id"], "status": "missing_selection_artifact", "pas_dir": pool["pas_dir"]})
            continue
        tradeoff_rows.extend(build_tradeoff_rows(pool, data))
        sensitivity_rows.extend(build_sensitivity_rows(pool, data, top_m_values, epsilon_values))
        figure_artifacts.extend(plot_pool_figures(pool, data, output_dir / "figures"))
        manifest_pools.append({"pool_id": pool["pool_id"], "status": "processed", "pas_dir": pool["pas_dir"]})

    tradeoff_csv = output_dir / "policy_selection_tradeoff.csv"
    tradeoff_md = output_dir / "policy_selection_tradeoff.md"
    sensitivity_csv = output_dir / "shortlist_sensitivity.csv"
    sensitivity_md = output_dir / "shortlist_sensitivity.md"
    manifest_path = output_dir / "policy_selection_manifest.json"
    write_csv(tradeoff_csv, tradeoff_rows)
    write_markdown_table(tradeoff_md, tradeoff_rows)
    write_csv(sensitivity_csv, sensitivity_rows)
    write_markdown_table(sensitivity_md, sensitivity_rows)
    write_json(
        manifest_path,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pools": manifest_pools,
            "top_m": top_m_values,
            "epsilon_logloss": epsilon_values,
            "heldout_usage": "analysis_only_not_selection_tuning_filtering_or_early_stopping",
            "artifacts": {
                "policy_selection_tradeoff_csv": str(tradeoff_csv),
                "policy_selection_tradeoff_md": str(tradeoff_md),
                "shortlist_sensitivity_csv": str(sensitivity_csv),
                "shortlist_sensitivity_md": str(sensitivity_md),
                "figures": figure_artifacts,
            },
        },
    )
    print(f"Wrote {tradeoff_csv}")
    print(f"Wrote {tradeoff_md}")
    print(f"Wrote {sensitivity_csv}")
    print(f"Wrote {sensitivity_md}")
    print(f"Wrote {manifest_path}")
    for artifact in figure_artifacts:
        print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
