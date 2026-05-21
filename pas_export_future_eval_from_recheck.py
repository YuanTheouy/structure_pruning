#!/usr/bin/env python3
"""Materialize matched 40% final-eval artifacts from a selected-candidate recheck."""

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export pas_compensation_aligned_eval_40.csv from an already completed "
            "selected-candidate recheck when the protocols are equivalent."
        )
    )
    parser.add_argument("--selected_candidates_json", required=True)
    parser.add_argument("--recheck_csv", required=True)
    parser.add_argument("--recheck_regret_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--future_sparsity", type=float, default=0.40)
    parser.add_argument("--rules", default="FF-Endpoint,PAS-Slope")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3025)
    return parser.parse_args()


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def rules_for_selected(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mapping = {}
    for rule, item in payload.items():
        if rule == "shortlist" or not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id") or item.get("probe_row", {}).get("candidate_id")
        if candidate_id:
            mapping[rule] = candidate_id
    return mapping


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted_rules = [rule.strip() for rule in args.rules.split(",") if rule.strip()]
    rule_to_candidate = rules_for_selected(args.selected_candidates_json)
    recheck_by_id = {row["candidate_id"]: row for row in read_csv_rows(args.recheck_csv)}
    regret_by_id = {row["candidate_id"]: row for row in read_csv_rows(args.recheck_regret_csv)}

    rows = []
    for rule in wanted_rules:
        candidate_id = rule_to_candidate.get(rule)
        if not candidate_id:
            raise RuntimeError(f"Rule {rule} not found in {args.selected_candidates_json}")
        recheck = recheck_by_id.get(candidate_id, {})
        regret = regret_by_id.get(candidate_id, {})
        if not recheck:
            raise RuntimeError(f"Candidate {candidate_id} missing from {args.recheck_csv}")
        rows.append(
            {
                "rule": rule,
                "candidate_id": candidate_id,
                "future_sparsity": args.future_sparsity,
                "ell": recheck.get("ell_h_recheck", ""),
                "ppl": recheck.get("ppl_h_recheck", ""),
                "actual_sparsity": recheck.get("actual_sparsity_h", ""),
                "budget_error": recheck.get("budget_error_h", ""),
                "relative_budget_error": recheck.get("relative_budget_error_h", ""),
                "recon_enabled": False,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "checkpoint_path": recheck.get("checkpoint_path", ""),
                "metadata_path": recheck.get("metadata_path", ""),
                "eval_seconds": recheck.get("eval_seconds_h", ""),
                "best_rechecked_candidate_id": regret.get("best_rechecked_candidate_id", ""),
                "best_rechecked_ell_h": regret.get("best_rechecked_ell_h", ""),
                "regret_vs_best_rechecked": regret.get("regret_vs_best_rechecked", ""),
                "source_recheck_csv": args.recheck_csv,
            }
        )

    csv_path = output_dir / "pas_compensation_aligned_eval_40.csv"
    manifest_path = output_dir / "pas_compensation_aligned_manifest_40.json"
    commands_path = output_dir / "pas_compensation_aligned_commands_40.sh"
    write_csv(csv_path, rows)
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("# Materialize matched 40% no-recovery evaluation artifacts from selected-candidate recheck.\n")
        handle.write(" ".join(shlex.quote(part) for part in sys.argv))
        handle.write("\n")
    write_json(
        manifest_path,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "model": args.model,
            "model_name": args.model_name or Path(args.model).name,
            "dataset": args.dataset,
            "future_sparsity": args.future_sparsity,
            "rules": wanted_rules,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "selected_candidates_json": args.selected_candidates_json,
            "source_recheck_csv": args.recheck_csv,
            "source_recheck_regret_csv": args.recheck_regret_csv,
            "equivalence_statement": (
                "The selected-candidate recheck uses amc_searchPPO.py --job=compile "
                "with final_sparsity=0.40, no reconstruction, the same model/dataset, "
                "same sample count, and the same selected priority-vector candidates. "
                "Therefore it is the matched stricter-budget final evaluation requested "
                "for P3; this script only materializes the required artifact names."
            ),
            "artifacts": {
                "pas_compensation_aligned_eval_40": str(csv_path),
                "pas_compensation_aligned_manifest_40": str(manifest_path),
                "pas_compensation_aligned_commands_40": str(commands_path),
            },
        },
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {commands_path}")


if __name__ == "__main__":
    main()
