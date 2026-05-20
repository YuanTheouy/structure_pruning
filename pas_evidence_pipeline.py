#!/usr/bin/env python3
"""Build PAS evidence artifacts from one FastForward candidate pool."""

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


RULE_ENDPOINT = "FF-Endpoint"
RULE_RANDOM = "Random-shortlist"
RULE_PLUS = "PAS-Plus"
RULE_SLOPE = "PAS-Slope"
RULE_CURV = "PAS-Curv"
RULE_ORACLE = "Oracle-heldout"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute PAS controlled-selection artifacts. PAS selections use only "
            "the local 0.25/0.30/0.35 probe; the 0.40 held-out point is used "
            "only for Oracle analysis, regret, and reporting."
        )
    )
    parser.add_argument("--candidate_dir", required=True, help="Directory containing candidates.jsonl")
    parser.add_argument("--probe_results", required=True, help="Probe CSV from 0.25/0.30/0.35")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--target_sparsity", type=float, default=0.30)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--future_sparsity", type=float, default=0.40)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--shortlist_size", type=int, default=5)
    parser.add_argument(
        "--endpoint_epsilon",
        type=float,
        default=None,
        help="Optional endpoint-competitive epsilon. If set, shortlist is ell_0 <= best ell_0 + epsilon.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--random_repeats", type=int, default=1000)
    parser.add_argument("--probe_num_samples", type=int, default=None)
    parser.add_argument("--heldout_num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--heldout_results_csv", default=None, help="Existing heldout_results.csv to reuse.")
    parser.add_argument("--force_heldout", action="store_true", help="Re-evaluate 0.40 even if output exists.")
    parser.add_argument("--dry_run", action="store_true", help="Print held-out compile commands without running them.")
    parser.add_argument("--prune", default="para")
    parser.add_argument("--lbound", type=float, default=0.1)
    parser.add_argument("--rbound", type=float, default=1.0)
    return parser.parse_args()


def safe_id(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"Missing numeric field, tried {keys}: {row}")


def optional_float(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return ""


def normalize_probe_rows(rows, delta):
    normalized = []
    for row in rows:
        out = dict(row)
        ell_minus = as_float(row, "ell_minus", "logppl_minus")
        ell_0 = as_float(row, "ell_0", "logppl_zero")
        ell_plus = as_float(row, "ell_plus", "logppl_plus")
        slope = optional_float(row, "slope")
        if slope == "":
            slope = (ell_plus - ell_0) / delta
        curvature = optional_float(row, "curvature")
        if curvature == "":
            curvature = (ell_plus - 2.0 * ell_0 + ell_minus) / (delta ** 2)

        out.update(
            {
                "ell_minus": ell_minus,
                "ell_0": ell_0,
                "ell_plus": ell_plus,
                "ppl_minus": optional_float(row, "ppl_minus"),
                "ppl_0": optional_float(row, "ppl_0", "ppl_zero"),
                "ppl_plus": optional_float(row, "ppl_plus"),
                "slope": slope,
                "curvature": curvature,
                "heldout_degradation": "",
            }
        )
        normalized.append(out)
    normalized.sort(key=lambda item: (float(item["ell_0"]), item.get("candidate_id", "")))
    return normalized


def build_shortlist(rows, shortlist_size, endpoint_epsilon):
    if not rows:
        raise RuntimeError("No probe rows available for PAS selection.")
    best_ell0 = float(rows[0]["ell_0"])
    if endpoint_epsilon is not None:
        shortlist = [row for row in rows if float(row["ell_0"]) <= best_ell0 + endpoint_epsilon]
        rule = f"ell_0 <= best_ell_0 + {endpoint_epsilon:g}"
    else:
        keep = min(max(1, shortlist_size), len(rows))
        shortlist = rows[:keep]
        rule = f"top-{keep}-by-ell_0"
    return shortlist, rule


def choose_min(rows, key):
    return min(rows, key=lambda row: (float(row[key]), float(row["ell_0"]), row.get("candidate_id", "")))


def select_without_heldout(probe_rows, shortlist):
    return {
        RULE_ENDPOINT: choose_min(probe_rows, "ell_0"),
        RULE_PLUS: choose_min(shortlist, "ell_plus"),
        RULE_SLOPE: choose_min(shortlist, "slope"),
        RULE_CURV: choose_min(shortlist, "curvature"),
    }


def metadata_path_for_export(export_path):
    return export_path.with_suffix(".json")


def build_compile_command(args, repo_root, best_path, export_path, final_policy_path):
    model_name = args.model_name or Path(args.model).name
    return [
        sys.executable,
        "-u",
        str(repo_root / "amc_searchPPO.py"),
        "--job=compile",
        f"--model={args.model}",
        f"--model_name={model_name}",
        f"--dataset_name={args.dataset}",
        f"--preserve_ratio={1.0 - args.future_sparsity:.6f}",
        f"--final_sparsity={args.future_sparsity}",
        f"--best_candidate_path={best_path}",
        f"--export_path={export_path}",
        f"--final_policy_path={final_policy_path}",
        "--structure",
        f"--prune={args.prune}",
        f"--lbound={args.lbound}",
        f"--rbound={args.rbound}",
        f"--n_samples={args.heldout_num_samples}",
        f"--data_bsize={args.batch_size}",
        f"--seed={args.seed}",
        "--enable_downstream=false",
    ]


def evaluate_heldout(args, probe_rows, candidates_by_id, output_dir):
    existing_path = Path(args.heldout_results_csv) if args.heldout_results_csv else output_dir / "heldout_results.csv"
    if existing_path.exists() and not args.force_heldout and not args.dry_run:
        rows = read_csv_rows(existing_path)
        required_ids = {row["candidate_id"] for row in probe_rows}
        existing_ids = {row.get("candidate_id") for row in rows}
        if required_ids.issubset(existing_ids):
            output_path = output_dir / "heldout_results.csv"
            if existing_path.resolve() != output_path.resolve():
                write_csv(output_path, rows)
                return rows, output_path
            return rows, existing_path
        missing = sorted(required_ids - existing_ids)
        print(f"Existing held-out CSV is incomplete; re-evaluating {len(missing)} missing candidates.")

    repo_root = Path(__file__).resolve().parent
    rows = []
    eval_root = output_dir / "heldout_eval"
    commands_path = output_dir / "heldout_eval_commands.sh"
    command_lines = []

    for probe in probe_rows:
        candidate_id = probe["candidate_id"]
        candidate = dict(candidates_by_id.get(candidate_id, {}))
        if not candidate:
            candidate = {"candidate_id": candidate_id}
        if not candidate.get("score_path") and probe.get("score_path"):
            candidate["score_path"] = probe["score_path"]

        run_dir = eval_root / safe_id(candidate_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best_candidate.json"
        export_path = run_dir / "checkpoint.pth.tar"
        final_policy_path = run_dir / "final_policy.json"
        metadata_path = metadata_path_for_export(export_path)

        payload = {
            "selected_mode": "pas_heldout_analysis_only",
            "selection_rule": "heldout_0.40_not_used_for_pas_selection",
            "candidate": candidate,
            "probe_row": probe,
        }
        write_json(best_path, payload)
        cmd = build_compile_command(args, repo_root, best_path, export_path, final_policy_path)
        command_lines.append(" ".join(shlex.quote(str(part)) for part in cmd))

        if metadata_path.exists() and not args.force_heldout:
            pass
        elif args.dry_run:
            continue
        else:
            print(" ".join(shlex.quote(str(part)) for part in cmd))
            rc = subprocess.call(cmd, cwd=repo_root)
            if rc != 0:
                raise SystemExit(rc)

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            ppl_h = float(metadata["ppl"])
            ell_h = math.log(ppl_h) if ppl_h > 0 else float("inf")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "future_sparsity": args.future_sparsity,
                    "ell_h": ell_h,
                    "ppl_h": ppl_h,
                    "actual_sparsity_h": metadata.get("actual_sparsity", ""),
                    "budget_error_h": metadata.get("budget_error", ""),
                    "relative_budget_error_h": metadata.get("relative_budget_error", ""),
                    "checkpoint_path": str(export_path),
                    "metadata_path": str(metadata_path),
                    "heldout_num_samples": args.heldout_num_samples,
                    "analysis_only": True,
                }
            )

    commands_path.parent.mkdir(parents=True, exist_ok=True)
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\n")
        for line in command_lines:
            handle.write(line + "\n")

    if args.dry_run:
        print(f"Wrote held-out dry-run commands to {commands_path}")
        return [], existing_path

    write_csv(output_dir / "heldout_results.csv", rows)
    return rows, output_dir / "heldout_results.csv"


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr(x, y):
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_warning_correlation(probe_rows, heldout_by_id, target_sparsity, future_sparsity):
    joined = []
    for row in probe_rows:
        heldout = heldout_by_id.get(row["candidate_id"])
        if not heldout:
            continue
        ell_h = as_float(heldout, "ell_h")
        ell_0 = float(row["ell_0"])
        item = dict(row)
        item.update({"ell_h": ell_h, "heldout_degradation": ell_h - ell_0})
        joined.append(item)

    target = np.asarray([float(row["heldout_degradation"]) for row in joined], dtype=np.float64)
    metrics = {
        "ell_plus": np.asarray([float(row["ell_plus"]) for row in joined], dtype=np.float64),
        "slope": np.asarray([float(row["slope"]) for row in joined], dtype=np.float64),
        "curvature": np.asarray([float(row["curvature"]) for row in joined], dtype=np.float64),
    }
    rows = []
    for name, values in metrics.items():
        rows.append(
            {
                "metric": name,
                "target": f"heldout_degradation_{future_sparsity:.2f}_minus_{target_sparsity:.2f}",
                "pearson": corr(values, target),
                "spearman": corr(rankdata(values), rankdata(target)),
                "n": len(joined),
            }
        )
    return rows, joined


def selection_payload(rule, probe_row, heldout_row=None, extra=None):
    payload = {
        "rule": rule,
        "candidate_id": probe_row.get("candidate_id"),
        "probe_row": probe_row,
    }
    if heldout_row:
        payload["heldout_row"] = heldout_row
    if extra:
        payload.update(extra)
    return payload


def compute_regret_rows(selections, shortlist, heldout_rows, random_repeats, seed):
    heldout_by_id = {row["candidate_id"]: row for row in heldout_rows}
    oracle = min(heldout_rows, key=lambda row: (float(row["ell_h"]), row["candidate_id"]))
    oracle_ell = float(oracle["ell_h"])

    rows = []
    for rule, probe_row in selections.items():
        candidate_id = probe_row["candidate_id"]
        heldout = heldout_by_id[candidate_id]
        ell_h = float(heldout["ell_h"])
        rows.append(
            {
                "rule": rule,
                "candidate_id": candidate_id,
                "selection_score": probe_row.get({
                    RULE_ENDPOINT: "ell_0",
                    RULE_PLUS: "ell_plus",
                    RULE_SLOPE: "slope",
                    RULE_CURV: "curvature",
                }.get(rule, "ell_0"), ""),
                "ell_0": probe_row["ell_0"],
                "ell_plus": probe_row["ell_plus"],
                "slope": probe_row["slope"],
                "curvature": probe_row["curvature"],
                "ell_h": ell_h,
                "ppl_h": heldout.get("ppl_h", ""),
                "oracle_candidate_id": oracle["candidate_id"],
                "oracle_ell_h": oracle_ell,
                "regret": ell_h - oracle_ell,
                "random_repeats": "",
                "regret_mean": "",
                "regret_std": "",
            }
        )

    rng = random.Random(seed)
    draws = []
    shortlist_ids = [row["candidate_id"] for row in shortlist if row["candidate_id"] in heldout_by_id]
    for draw_idx in range(random_repeats):
        candidate_id = rng.choice(shortlist_ids)
        heldout = heldout_by_id[candidate_id]
        regret = float(heldout["ell_h"]) - oracle_ell
        draws.append({"draw": draw_idx, "candidate_id": candidate_id, "ell_h": heldout["ell_h"], "regret": regret})
    regrets = np.asarray([float(row["regret"]) for row in draws], dtype=np.float64)
    rows.append(
        {
            "rule": RULE_RANDOM,
            "candidate_id": "",
            "selection_score": "",
            "ell_0": "",
            "ell_plus": "",
            "slope": "",
            "curvature": "",
            "ell_h": "",
            "ppl_h": "",
            "oracle_candidate_id": oracle["candidate_id"],
            "oracle_ell_h": oracle_ell,
            "regret": "",
            "random_repeats": random_repeats,
            "regret_mean": float(np.mean(regrets)) if len(regrets) else "",
            "regret_std": float(np.std(regrets)) if len(regrets) else "",
        }
    )
    rows.append(
        {
            "rule": RULE_ORACLE,
            "candidate_id": oracle["candidate_id"],
            "selection_score": oracle_ell,
            "ell_0": "",
            "ell_plus": "",
            "slope": "",
            "curvature": "",
            "ell_h": oracle_ell,
            "ppl_h": oracle.get("ppl_h", ""),
            "oracle_candidate_id": oracle["candidate_id"],
            "oracle_ell_h": oracle_ell,
            "regret": 0.0,
            "random_repeats": "",
            "regret_mean": "",
            "regret_std": "",
        }
    )
    return rows, draws, oracle


def plot_figures(output_dir, joined_rows, shortlist_ids):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path_pdf = output_dir / "path_divergence.pdf"
        xs = [0.25, 0.30, 0.35, 0.40]
        plt.figure(figsize=(6.2, 4.0))
        plotted = 0
        for row in joined_rows:
            if row["candidate_id"] not in shortlist_ids and plotted >= 12:
                continue
            ys = [float(row["ell_minus"]), float(row["ell_0"]), float(row["ell_plus"]), float(row["ell_h"])]
            alpha = 0.85 if row["candidate_id"] in shortlist_ids else 0.25
            linewidth = 1.4 if row["candidate_id"] in shortlist_ids else 0.8
            plt.plot(xs, ys, marker="o", linewidth=linewidth, alpha=alpha)
            plotted += 1
        plt.xlabel("Sparsity")
        plt.ylabel("Log PPL")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(path_pdf)
        plt.close()

        ambiguity_pdf = output_dir / "endpoint_ambiguity_scatter.pdf"
        plt.figure(figsize=(5.2, 4.2))
        colors = ["tab:orange" if row["candidate_id"] in shortlist_ids else "tab:blue" for row in joined_rows]
        plt.scatter(
            [float(row["ell_0"]) for row in joined_rows],
            [float(row["ell_h"]) for row in joined_rows],
            c=colors,
            s=36,
            alpha=0.85,
        )
        plt.xlabel("Endpoint log PPL at 0.30")
        plt.ylabel("Held-out log PPL at 0.40")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(ambiguity_pdf)
        plt.close()

        corr_pdf = output_dir / "warning_correlation.pdf"
        target = [float(row["heldout_degradation"]) for row in joined_rows]
        metrics = [("ell_plus", "ell_plus"), ("slope", "slope"), ("curvature", "curvature")]
        plt.figure(figsize=(12.0, 3.6))
        for idx, (field, label) in enumerate(metrics, start=1):
            ax = plt.subplot(1, 3, idx)
            values = [float(row[field]) for row in joined_rows]
            ax.scatter(values, target, s=32, alpha=0.85)
            if len(values) >= 2 and np.std(values) > 1e-12:
                fit = np.polyfit(values, target, deg=1)
                x_min, x_max = min(values), max(values)
                xs_fit = np.linspace(x_min, x_max, 100)
                ax.plot(xs_fit, fit[0] * xs_fit + fit[1], color="black", linewidth=1.1)
            ax.set_xlabel(label)
            if idx == 1:
                ax.set_ylabel("Held-out degradation")
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(corr_pdf)
        plt.close()
    except Exception as exc:
        print(f"Skipping PAS figures: {exc}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = Path(args.candidate_dir)

    candidates = read_jsonl(candidate_dir / "candidates.jsonl")
    if not candidates:
        raise RuntimeError(f"No candidates found in {candidate_dir / 'candidates.jsonl'}")
    candidates_by_id = {row["candidate_id"]: row for row in candidates if row.get("candidate_id")}

    probe_rows = normalize_probe_rows(read_csv_rows(Path(args.probe_results)), args.delta)
    if args.top_k > 0:
        probe_rows = probe_rows[: args.top_k]
    if not probe_rows:
        raise RuntimeError(f"No probe rows found in {args.probe_results}")

    shortlist, shortlist_rule = build_shortlist(probe_rows, args.shortlist_size, args.endpoint_epsilon)
    selections = select_without_heldout(probe_rows, shortlist)
    normalized_probe_path = output_dir / "probe_results.csv"
    write_csv(normalized_probe_path, probe_rows)

    heldout_rows, heldout_path = evaluate_heldout(args, probe_rows, candidates_by_id, output_dir)
    if args.dry_run:
        return 0
    if not heldout_rows:
        raise RuntimeError("No held-out rows available.")

    heldout_by_id = {row["candidate_id"]: row for row in heldout_rows}
    correlation_rows, joined_rows = compute_warning_correlation(
        probe_rows, heldout_by_id, args.target_sparsity, args.future_sparsity
    )
    for row in probe_rows:
        heldout = heldout_by_id.get(row["candidate_id"])
        if heldout:
            row["heldout_degradation"] = float(heldout["ell_h"]) - float(row["ell_0"])
    write_csv(normalized_probe_path, probe_rows)

    warning_csv = output_dir / "warning_correlation.csv"
    write_csv(warning_csv, correlation_rows)
    joined_csv = output_dir / "pas_joined_probe_heldout.csv"
    write_csv(joined_csv, joined_rows)

    regret_rows, random_draws, oracle = compute_regret_rows(
        selections, shortlist, heldout_rows, args.random_repeats, args.seed
    )
    regret_csv = output_dir / "selection_regret.csv"
    write_csv(regret_csv, regret_rows)
    write_csv(output_dir / "random_shortlist_draws.csv", random_draws)

    selected_json = {
        rule: selection_payload(rule, row, heldout_by_id.get(row["candidate_id"]))
        for rule, row in selections.items()
    }
    oracle_probe = next((row for row in probe_rows if row["candidate_id"] == oracle["candidate_id"]), {})
    selected_json[RULE_ORACLE] = selection_payload(RULE_ORACLE, oracle_probe, oracle, {"analysis_only": True})
    selected_json["shortlist"] = {
        "rule": shortlist_rule,
        "candidate_ids": [row["candidate_id"] for row in shortlist],
        "heldout_0.40_used_for_selection": False,
    }
    write_json(output_dir / "selected_candidates.json", selected_json)

    shortlist_ids = {row["candidate_id"] for row in shortlist}
    plot_figures(output_dir, joined_rows, shortlist_ids)

    manifest = {
        "model": args.model,
        "model_name": args.model_name or Path(args.model).name,
        "dataset": args.dataset,
        "candidate_dir": str(candidate_dir),
        "source_probe_results": str(Path(args.probe_results)),
        "seed": args.seed,
        "sigma": args.target_sparsity,
        "delta": args.delta,
        "probe_budgets": [
            round(args.target_sparsity - args.delta, 6),
            args.target_sparsity,
            round(args.target_sparsity + args.delta, 6),
        ],
        "heldout_future_budget": args.future_sparsity,
        "heldout_0.40_usage": "analysis_only_not_selection_tuning_filtering_or_early_stopping",
        "top_k": args.top_k,
        "shortlist_rule": shortlist_rule,
        "shortlist_size": len(shortlist),
        "random_repeats": args.random_repeats,
        "probe_samples": args.probe_num_samples,
        "heldout_samples": args.heldout_num_samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "artifacts": {
            "probe_results": str(normalized_probe_path),
            "heldout_results": str(heldout_path),
            "warning_correlation": str(warning_csv),
            "selection_regret": str(regret_csv),
            "selected_candidates": str(output_dir / "selected_candidates.json"),
            "path_divergence_pdf": str(output_dir / "path_divergence.pdf"),
            "endpoint_ambiguity_scatter_pdf": str(output_dir / "endpoint_ambiguity_scatter.pdf"),
            "warning_correlation_pdf": str(output_dir / "warning_correlation.pdf"),
            "joined_probe_heldout": str(joined_csv),
            "random_shortlist_draws": str(output_dir / "random_shortlist_draws.csv"),
        },
    }
    write_json(output_dir / "artifact_manifest.json", manifest)

    print(f"Wrote PAS artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
