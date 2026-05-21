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
        "dataset": "wikitext2",
        "seed": "2025",
        "sigma": 0.30,
        "delta": 0.05,
        "heldout_sigma": 0.40,
        "pas_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas",
        "recheck_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_selected_recheck64",
        "final_eval_dir": "",
    },
    {
        "pool_id": "opt27b_seed3025",
        "model": "opt-2.7b",
        "dataset": "wikitext2",
        "seed": "3025",
        "sigma": 0.30,
        "delta": 0.05,
        "heldout_sigma": 0.40,
        "pas_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025",
        "recheck_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64",
        "final_eval_dir": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon",
    },
    {
        "pool_id": "opt13b_seed2025",
        "model": "opt-1.3b",
        "dataset": "wikitext2",
        "seed": "2025",
        "sigma": 0.30,
        "delta": 0.05,
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
    parser.add_argument(
        "--artifact_suffix",
        default="",
        help="Optional suffix for report/figure filenames, e.g. _sigma035.",
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


def save_current_figure(paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)


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


def dataset_name(pool):
    return pool.get("dataset") or "wikitext2"


def probe_budgets(pool):
    sigma = float(pool["sigma"])
    delta = float(pool.get("delta", 0.05))
    return [round(sigma - delta, 10), round(sigma, 10), round(sigma + delta, 10)]


def probe_budgets_text(pool):
    return "/".join(f"{value:.2f}" for value in probe_budgets(pool))


def candidate_pool_path(pool, data):
    manifest = data.get("pas_manifest") or {}
    return pool.get("candidate_pool") or manifest.get("candidate_dir") or str(Path(pool["pas_dir"]) / "candidate_pool_unknown")


def uses_heldout_for_selection(rule):
    if rule == "Oracle-heldout":
        return "yes_oracle_analysis_only"
    return "no"


def protocol_note(target_source, heldout_source):
    if target_source == "final_compile_eval" and heldout_source == "selected_recheck_64":
        return "matched_no_recovery_compile_eval"
    if target_source == "probe_ell_sigma" and heldout_source == "heldout_results":
        return "pipeline_probe_and_analysis_eval; not compensation-aligned final eval"
    if target_source == "probe_ell_sigma" and heldout_source == "selected_recheck_64":
        return "protocol_mismatch_target_probe_vs_selected_recheck; use P3 matched eval before final claim"
    if "missing" in {target_source, heldout_source}:
        return "missing_source_values"
    return f"target={target_source}; heldout={heldout_source}"


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
    candidate_pool = candidate_pool_path(pool, data)
    budgets = probe_budgets_text(pool)
    ff_row = row_for_rule(data["selection"], "FF-Endpoint")
    ff_candidate = ff_row.get("candidate_id")
    ff_target_ell, _, _ = target_values(ff_candidate, data)
    ff_heldout_ell, _, _, _ = heldout_values(ff_candidate, data)

    for rule in RULES:
        selection = row_for_rule(data["selection"], rule)
        candidate_id = selection.get("candidate_id")
        if not candidate_id and rule == "Oracle-heldout":
            candidate_id = row_for_rule(data["selection"], "Oracle-heldout").get("candidate_id")

        target_ell, target_ppl, target_source = target_values(candidate_id, data)
        heldout_ell, heldout_ppl, recheck_regret, heldout_source = heldout_values(candidate_id, data)
        if recheck_regret is None:
            recheck_regret = as_float(selection, "regret")

        target_cost = target_ell - ff_target_ell if target_ell is not None and ff_target_ell is not None else ""
        heldout_gain = ff_heldout_ell - heldout_ell if heldout_ell is not None and ff_heldout_ell is not None else ""
        rows.append(
            {
                "model": pool["model"],
                "dataset": dataset_name(pool),
                "seed": pool["seed"],
                "candidate_pool": candidate_pool,
                "target_sigma": pool["sigma"],
                "probe_budgets": budgets,
                "heldout_sigma": pool["heldout_sigma"],
                "rule": rule,
                "selected_candidate": candidate_id or "",
                "selection_inputs_used": SELECTION_INPUTS.get(rule, ""),
                "uses_heldout_for_selection": uses_heldout_for_selection(rule),
                "target_ell": target_ell if target_ell is not None else "",
                "target_ppl": target_ppl if target_ppl is not None else "",
                "heldout_ell": heldout_ell if heldout_ell is not None else "",
                "heldout_ppl": heldout_ppl if heldout_ppl is not None else "",
                "PoBR_sigma": target_cost,
                "StressGain_h": heldout_gain,
                "Regret_h": recheck_regret if recheck_regret is not None else "",
                "artifact_source_target": target_source,
                "artifact_source_heldout": heldout_source,
                "notes": protocol_note(target_source, heldout_source),
            }
        )

    random_row = row_for_rule(data["selection"], "Random-shortlist")
    if random_row:
        rows.append(
            {
                "model": pool["model"],
                "dataset": dataset_name(pool),
                "seed": pool["seed"],
                "candidate_pool": candidate_pool,
                "target_sigma": pool["sigma"],
                "probe_budgets": budgets,
                "heldout_sigma": pool["heldout_sigma"],
                "rule": "Random-shortlist",
                "selected_candidate": "random_from_shortlist_distribution",
                "selection_inputs_used": SELECTION_INPUTS["Random-shortlist"],
                "uses_heldout_for_selection": "no",
                "target_ell": "",
                "target_ppl": "",
                "heldout_ell": "",
                "heldout_ppl": "",
                "PoBR_sigma": "",
                "StressGain_h": "",
                "Regret_h": random_row.get("regret_mean", ""),
                "artifact_source_target": "distribution_only_no_single_target_eval",
                "artifact_source_heldout": str(Path(pool["pas_dir"]) / "selection_regret.csv"),
                "notes": "random_shortlist_reports_distribution_mean_not_single_selected_candidate",
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

    candidate_pool = candidate_pool_path(pool, data)
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
            target_ppl = as_float(data["probe_by_id"].get(candidate_id, {}), "ppl_0", as_float(data["probe_by_id"].get(candidate_id, {}), "ppl_zero"))
            heldout_ppl = as_float(heldout_by_id.get(candidate_id, {}), "ppl_h")
            rows.append(
                {
                    "model": pool["model"],
                    "dataset": dataset_name(pool),
                    "seed": pool["seed"],
                    "candidate_pool": candidate_pool,
                    "target_sigma": pool["sigma"],
                    "heldout_sigma": pool["heldout_sigma"],
                    "shortlist_type": shortlist_type,
                    "shortlist_value": value,
                    "shortlist_size": len(shortlist),
                    "rule": rule,
                    "selected_candidate": candidate_id or "",
                    "target_ell": target_ell if target_ell is not None else "",
                    "target_ppl": target_ppl if target_ppl is not None else "",
                    "heldout_ell": heldout_ell if heldout_ell is not None else "",
                    "heldout_ppl": heldout_ppl if heldout_ppl is not None else "",
                    "PoBR_sigma": target_regret,
                    "StressGain_h": ff_heldout - heldout_ell if ff_heldout is not None and heldout_ell is not None else "",
                    "Regret_h": heldout_regret,
                    "notes": (
                        "predeclared_endpoint_compatible_shortlist; heldout_analysis_only"
                        if rule != "Oracle-heldout"
                        else "oracle_analysis_baseline_uses_heldout"
                    ),
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


def plot_pool_figures(pool, data, output_dir, artifact_suffix=""):
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

    fig_path = pool_dir / f"endpoint_ambiguity_scatter{artifact_suffix}.pdf"
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

    paper_path = pool_dir / f"path_divergence{artifact_suffix}.pdf"
    aux_path = pool_dir / f"policy_path_lines{artifact_suffix}.pdf"
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
    save_current_figure([paper_path, aux_path])
    plt.close()
    artifacts.extend([str(paper_path), str(aux_path)])

    paper_path = pool_dir / f"robustness_frontier{artifact_suffix}.pdf"
    aux_path = pool_dir / f"target_future_tradeoff{artifact_suffix}.pdf"
    ff_row = row_for_rule(data["selection"], "FF-Endpoint")
    ff_candidate = ff_row.get("candidate_id")
    ff_target = as_float(data["probe_by_id"].get(ff_candidate, {}), "ell_0")
    ff_heldout = as_float(heldout_by_id.get(ff_candidate, {}), "ell_h")
    oracle = choose_min(data["heldout"], "ell_h")
    oracle_heldout = as_float(oracle, "ell_h")
    plt.figure(figsize=(5.6, 4.0))
    all_xs = []
    all_ys = []
    if None not in (ff_target, oracle_heldout):
        for probe_row in probe_rows:
            candidate_id = probe_row.get("candidate_id")
            heldout = heldout_by_id.get(candidate_id, {})
            target = as_float(probe_row, "ell_0")
            future = as_float(heldout, "ell_h")
            if target is None or future is None:
                continue
            all_xs.append(target - ff_target)
            all_ys.append(future - oracle_heldout)
        plt.scatter(all_xs, all_ys, color="0.72", alpha=0.55, label="candidate")
    for rule in ["FF-Endpoint", "PAS-Plus", "PAS-Slope", "PAS-Curv", "Oracle-heldout"]:
        row = row_for_rule(data["selection"], rule)
        candidate_id = row.get("candidate_id")
        target = as_float(data["probe_by_id"].get(candidate_id, {}), "ell_0")
        heldout = as_float(heldout_by_id.get(candidate_id, {}), "ell_h")
        if None in (target, heldout, ff_target, oracle_heldout):
            continue
        plt.scatter([target - ff_target], [heldout - oracle_heldout], s=70)
        plt.annotate(rule, (target - ff_target, heldout - oracle_heldout), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("PoBR_sigma")
    plt.ylabel("Regret_h")
    plt.title(f"Robustness frontier: {pool['model']} seed {pool['seed']}")
    plt.grid(alpha=0.25)
    if all_xs:
        plt.legend(fontsize=8)
    plt.tight_layout()
    save_current_figure([paper_path, aux_path])
    plt.close()
    artifacts.extend([str(paper_path), str(aux_path)])

    paper_path = pool_dir / f"sensitivity_correlation{artifact_suffix}.pdf"
    aux_path = pool_dir / f"warning_correlation{artifact_suffix}.pdf"
    metrics = [
        ("ell_plus", "ell(sigma+delta)"),
        ("slope", "PAS-Slope"),
        ("curvature", "PAS-Curv"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for axis, (metric, label) in zip(axes, metrics):
        mxs = []
        dys = []
        for row in probe_rows:
            candidate_id = row.get("candidate_id")
            heldout = heldout_by_id.get(candidate_id, {})
            ell_0 = as_float(row, "ell_0")
            ell_h = as_float(heldout, "ell_h")
            value = as_float(row, metric)
            if None in (ell_0, ell_h, value):
                continue
            mxs.append(value)
            dys.append(ell_h - ell_0)
        axis.scatter(mxs, dys, alpha=0.7)
        axis.set_xlabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("ell(heldout) - ell(sigma)")
    fig.suptitle(f"Local warnings vs held-out degradation: {pool['model']} seed {pool['seed']}")
    fig.tight_layout()
    for path in [paper_path, aux_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)
    artifacts.extend([str(paper_path), str(aux_path)])
    return artifacts


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.pool_config:
        pools = read_json(args.pool_config)
    else:
        pools = DEFAULT_POOLS
    artifact_suffix = args.artifact_suffix
    if artifact_suffix and not artifact_suffix.startswith("_"):
        artifact_suffix = "_" + artifact_suffix
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
        figure_artifacts.extend(plot_pool_figures(pool, data, output_dir / "figures", artifact_suffix=artifact_suffix))
        manifest_pools.append(
            {
                "pool_id": pool["pool_id"],
                "status": "processed",
                "pas_dir": pool["pas_dir"],
                "candidate_pool": candidate_pool_path(pool, data),
                "candidate_count": len(data["probe"]),
            }
        )

    tradeoff_csv = output_dir / f"policy_selection_tradeoff{artifact_suffix}.csv"
    tradeoff_md = output_dir / f"policy_selection_tradeoff{artifact_suffix}.md"
    sensitivity_csv = output_dir / f"shortlist_sensitivity{artifact_suffix}.csv"
    sensitivity_md = output_dir / f"shortlist_sensitivity{artifact_suffix}.md"
    pobr_seed3025_csv = output_dir / f"price_of_budget_robustness_seed3025{artifact_suffix}.csv"
    manifest_path = output_dir / "policy_selection_manifest.json"
    write_csv(tradeoff_csv, tradeoff_rows)
    write_markdown_table(tradeoff_md, tradeoff_rows)
    write_csv(sensitivity_csv, sensitivity_rows)
    write_markdown_table(sensitivity_md, sensitivity_rows)
    pobr_seed3025_rows = [
        row
        for row in tradeoff_rows
        if row.get("model") == "opt-2.7b" and str(row.get("seed")) == "3025"
    ]
    if pobr_seed3025_rows:
        write_csv(pobr_seed3025_csv, pobr_seed3025_rows)
    write_json(
        manifest_path,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pools": manifest_pools,
            "top_m": top_m_values,
            "epsilon_logloss": epsilon_values,
            "artifact_suffix": artifact_suffix,
            "heldout_usage": "analysis_only_not_selection_tuning_filtering_or_early_stopping",
            "primary_rule": "PAS-Slope",
            "ablation_rule": "PAS-Curv",
            "artifacts": {
                "policy_selection_tradeoff_csv": str(tradeoff_csv),
                "policy_selection_tradeoff_md": str(tradeoff_md),
                "shortlist_sensitivity_csv": str(sensitivity_csv),
                "shortlist_sensitivity_md": str(sensitivity_md),
                "price_of_budget_robustness_seed3025_csv": str(pobr_seed3025_csv) if pobr_seed3025_rows else "",
                "figures": figure_artifacts,
            },
        },
    )
    print(f"Wrote {tradeoff_csv}")
    print(f"Wrote {tradeoff_md}")
    print(f"Wrote {sensitivity_csv}")
    print(f"Wrote {sensitivity_md}")
    if pobr_seed3025_rows:
        print(f"Wrote {pobr_seed3025_csv}")
    print(f"Wrote {manifest_path}")
    for artifact in figure_artifacts:
        print(f"Wrote {artifact}")


if __name__ == "__main__":
    main()
