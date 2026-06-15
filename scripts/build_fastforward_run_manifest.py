#!/usr/bin/env python3
"""Build the FastForward/PAS clean-run manifest.

The manifest is deliberately conservative. It records PDF/table/doc evidence,
but it does not upgrade a row to paper_usable unless a complete raw-artifact
mapping is available in the inspected sources.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path


COLUMNS = [
    "run_id",
    "status",
    "source",
    "model",
    "model_family",
    "sparsity",
    "method",
    "baseline",
    "seed",
    "PPL",
    "zero_shot_average",
    "calibration_protocol",
    "search_episodes_total",
    "search_episodes_per_worker",
    "num_workers",
    "gpu_count",
    "gpu_hours",
    "wall_clock_time",
    "warm_start_or_transfer",
    "dataset",
    "eval_samples",
    "artifact_path",
    "log_path",
    "commit_hash",
    "reproduces_pdf_row",
    "notes",
]

STATUSES = {
    "paper_usable",
    "diagnostic_only",
    "partial_negative",
    "debug_only",
    "missing_raw_artifact",
}

OUTPUT_DIR = Path("docs/fastforward_clean_manifest_20260615")
MISSING = "missing"


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except Exception:
        return MISSING


def model_family(model: str) -> str:
    low = model.lower()
    if "llama" in low:
        return "LLaMA"
    if "mistral" in low:
        return "Mistral"
    if "opt" in low:
        return "OPT"
    return MISSING


def clean_id(value: str) -> str:
    value = value.lower()
    value = value.replace("%", "pct")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def row(**kwargs: str) -> dict[str, str]:
    item = {key: MISSING for key in COLUMNS}
    item.update({key: str(value) for key, value in kwargs.items()})
    if item["status"] not in STATUSES:
        raise ValueError(f"bad status for {item.get('run_id')}: {item['status']}")
    if item["model_family"] == MISSING and item["model"] != MISSING:
        item["model_family"] = model_family(item["model"])
    return item


def add_pdf_rows(rows: list[dict[str, str]]) -> None:
    pdf = "../2511.18977v1.pdf"
    pdf_note = "PDF row captured, but no raw log/artifact mapping is present in inspected local sources."

    rows.append(
        row(
            run_id="pdf_table1_llama_v1_7b_20pct_ours_c4_zero_shot",
            status="missing_raw_artifact",
            source="2511.18977v1.pdf Table 1",
            model="LLaMA-V1-7B",
            sparsity="0.20",
            method="Ours(C4)",
            baseline="SliceGPT/FLAP/SVD-LLM/EAS-based",
            zero_shot_average="61.19",
            calibration_protocol="C4, 32x2048 samples",
            dataset="zero-shot seven-task average",
            artifact_path=pdf,
            reproduces_pdf_row="pdf_row_unverified",
            notes=pdf_note,
        )
    )
    rows.append(
        row(
            run_id="pdf_table1_llama_v1_7b_20pct_ours_calib_zero_shot",
            status="missing_raw_artifact",
            source="2511.18977v1.pdf Table 1",
            model="LLaMA-V1-7B",
            sparsity="0.20",
            method="Ours(Calib)",
            baseline="SliceGPT/FLAP/SVD-LLM/EAS-based",
            zero_shot_average="61.89",
            calibration_protocol="WikiText calibration, 32x2048 samples",
            dataset="zero-shot seven-task average",
            artifact_path=pdf,
            reproduces_pdf_row="pdf_row_unverified",
            notes=pdf_note,
        )
    )

    table2 = {
        "0.20": {
            "OPT-125M": ("31.44", "30.67"),
            "OPT-1.3B": ("16.75", "15.82"),
            "OPT-2.7B": ("14.55", "13.75"),
            "LLaMA-V1-7B": ("7.25", "6.64"),
            "LLaMA-V1-13B": ("6.22", "5.67"),
            "LLaMA-V2-7B": ("7.54", "6.79"),
            "LLaMA-V2-70B": ("4.32", "4.06"),
        },
        "0.30": {
            "OPT-125M": ("39.53", "36.19"),
            "OPT-1.3B": ("21.35", "18.65"),
            "OPT-2.7B": ("17.51", "16.23"),
            "LLaMA-V1-7B": ("8.65", "8.02"),
            "LLaMA-V1-13B": ("7.06", "6.43"),
            "LLaMA-V2-7B": ("8.73", "8.08"),
            "LLaMA-V2-70B": ("4.88", "4.62"),
        },
    }
    for sparsity, models in table2.items():
        for model, (ours_ppl, calib_ppl) in models.items():
            for method, ppl, calib in [
                ("Ours", ours_ppl, "none recorded in PDF row"),
                ("Ours(Calib)", calib_ppl, "calibration enabled; raw protocol missing locally"),
            ]:
                rows.append(
                    row(
                        run_id=f"pdf_table2_{clean_id(model)}_{clean_id(sparsity)}_{clean_id(method)}",
                        status="missing_raw_artifact",
                        source="2511.18977v1.pdf Table 2",
                        model=model,
                        sparsity=sparsity,
                        method=method,
                        baseline="SliceGPT/FLAP/SVD-LLM/EAS-based",
                        PPL=ppl,
                        calibration_protocol=calib,
                        dataset="WikiText-2",
                        artifact_path=pdf,
                        reproduces_pdf_row="pdf_row_unverified",
                        notes=pdf_note,
                    )
                )

    table3 = {
        "Calib.": {
            "OPT-125M": "32.25",
            "OPT-1.3B": "17.15",
            "OPT-2.7B": "14.65",
            "LLaMA-V2-7B": "7.15",
            "Mistral-7B": "7.22",
        },
        "Search": {
            "OPT-125M": "31.44",
            "OPT-1.3B": "16.75",
            "OPT-2.7B": "14.55",
            "LLaMA-V2-7B": "7.54",
            "Mistral-7B": "6.89",
        },
        "Search+Calib.": {
            "OPT-125M": "30.67",
            "OPT-1.3B": "15.82",
            "OPT-2.7B": "13.75",
            "LLaMA-V2-7B": "6.79",
            "Mistral-7B": "6.48",
        },
    }
    for method, models in table3.items():
        for model, ppl in models.items():
            rows.append(
                row(
                    run_id=f"pdf_table3_ablation_{clean_id(model)}_{clean_id(method)}",
                    status="missing_raw_artifact",
                    source="2511.18977v1.pdf Table 3",
                    model=model,
                    sparsity="0.20",
                    method=method,
                    baseline="Dense/Calib/Search/Search+Calib ablation",
                    PPL=ppl,
                    dataset="WikiText-2",
                    artifact_path=pdf,
                    reproduces_pdf_row="pdf_row_unverified",
                    notes=pdf_note,
                )
            )

    for sparsity, search_cost, total_cost, ppl, warm in [
        ("0.20", "6.13", "7.10", "6.64", "no"),
        ("0.30", "7.03", "8.12", "8.02", "yes, warm-started from 20% policy"),
    ]:
        rows.append(
            row(
                run_id=f"pdf_table4_llama_v1_7b_{clean_id(sparsity)}_ours_calib_cost",
                status="missing_raw_artifact",
                source="2511.18977v1.pdf Table 4",
                model="LLaMA-V1-7B",
                sparsity=sparsity,
                method="Ours(Calib)",
                baseline="FLAP/SliceGPT/EAS-based",
                PPL=ppl,
                gpu_hours=search_cost,
                wall_clock_time=MISSING,
                warm_start_or_transfer=warm,
                dataset="WikiText-2",
                artifact_path=pdf,
                reproduces_pdf_row="pdf_row_unverified",
                notes=f"PDF reports search GPU-hr={search_cost}, total GPU-hr={total_cost}; raw timing logs missing locally.",
            )
        )


def add_opt13_log_row(rows: list[dict[str, str]]) -> None:
    log_path = Path("../opt13-log.txt")
    ppls: list[float] = []
    episodes: list[int] = []
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"#(\d+):.*?ppl:\s*([0-9.]+)", text):
            episodes.append(int(match.group(1)))
            ppls.append(float(match.group(2)))
    notes = "Local opt13-log.txt has incomplete metadata and many very high PPL values."
    if ppls:
        notes += f" parsed_episodes={len(ppls)}, min_ppl={min(ppls):.4f}, max_ppl={max(ppls):.4f}, last_episode={max(episodes)}."
    rows.append(
        row(
            run_id="local_opt13_log_unverified",
            status="debug_only",
            source="../opt13-log.txt",
            model=MISSING,
            model_family="OPT",
            sparsity=MISSING,
            method="FastForward PPO search log",
            seed=MISSING,
            PPL=f"{min(ppls):.4f}" if ppls else MISSING,
            search_episodes_total=str(max(episodes) + 1) if episodes else MISSING,
            log_path=str(log_path),
            reproduces_pdf_row="unverified",
            notes=notes,
        )
    )


def add_pas_formal_rows(rows: list[dict[str, str]]) -> None:
    formal_dir = Path("docs/pas_formal_tables_20260525")
    selection_csv = formal_dir / "paper_selection_value_table.csv"
    if selection_csv.exists():
        with selection_csv.open("r", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                model = item.get("model", MISSING)
                seed = item.get("seed", MISSING)
                status = "diagnostic_only"
                if item.get("endpoint_is_oracle") == "True" and float(item.get("PAS_Stress_regret") or 0) > 0:
                    status = "partial_negative"
                rows.append(
                    row(
                        run_id=f"pas_formal_selection_{clean_id(model)}_seed{seed}",
                        status=status,
                        source=str(selection_csv),
                        model=model,
                        sparsity="0.30 target / 0.40 heldout",
                        method="PAS-Stress static diagnostic",
                        baseline="FF-Endpoint",
                        seed=seed,
                        dataset="WikiText-2",
                        eval_samples=MISSING,
                        artifact_path=item.get("selection_regret_csv", MISSING),
                        log_path=item.get("stress_table", MISSING),
                        reproduces_pdf_row="no",
                        notes=(
                            f"endpoint_is_oracle={item.get('endpoint_is_oracle', MISSING)}; "
                            f"FF_regret={item.get('FF_regret', MISSING)}; "
                            f"PAS_Stress_regret={item.get('PAS_Stress_regret', MISSING)}; "
                            "static PAS diagnostic, not FastForward main-table evidence."
                        ),
                    )
                )

    stress_csv = formal_dir / "paper_stress_correlation_table.csv"
    if stress_csv.exists():
        with stress_csv.open("r", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                model = item.get("model", MISSING)
                seed = item.get("seed", MISSING)
                rows.append(
                    row(
                        run_id=f"pas_formal_stress_corr_{clean_id(model)}_seed{seed}",
                        status="diagnostic_only",
                        source=str(stress_csv),
                        model=model,
                        sparsity=item.get("target_probe_heldout", "30->35->40"),
                        method="PAS-Stress correlation diagnostic",
                        baseline="FF-Endpoint",
                        seed=seed,
                        dataset="WikiText-2",
                        eval_samples=item.get("n", MISSING),
                        artifact_path=item.get("stress_table", MISSING),
                        reproduces_pdf_row="no",
                        notes=(
                            f"Spearman S35/Delta40={item.get('spearman_S35_Delta40', MISSING)}; "
                            f"partial S35/Regret40|L30={item.get('partial_S35_Regret40_given_L30', MISSING)}."
                        ),
                    )
                )


def add_curated_doc_rows(rows: list[dict[str, str]]) -> None:
    curated = [
        {
            "run_id": "pas_progressive_seed9025_ep2000_topk5_partial",
            "status": "partial_negative",
            "source": "docs/PROGRESSIVE_PAS_LOOKAHEAD_PARTIAL_2026_05_28.md",
            "model": "OPT-2.7B",
            "sparsity": "0.05->0.30 progressive",
            "method": "Progressive PAS lookahead replay",
            "baseline": "FF-stage-endpoint",
            "seed": "9025",
            "search_episodes_total": "2000",
            "gpu_count": "4 search / 4 replay",
            "dataset": "WikiText-2 fixed 5% validation subset",
            "artifact_path": "/workspace/ckpts/pas_progressive_lookahead/opt-2.7b_seed9025_ep2000_fixed0.05_staircase/replay",
            "notes": "Partial replay: 12 gate checks, PROMOTE=0, raw PAS improvements=0, completed probe batches=24/60.",
        },
        {
            "run_id": "pas_p0_smoke_opt27b_seed2025",
            "status": "debug_only",
            "source": "docs/PROJECT_PLAN.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30",
            "method": "PAS P0 smoke candidate/probe pipeline",
            "baseline": "FF-Endpoint",
            "seed": "2025-2032",
            "search_episodes_total": "80",
            "search_episodes_per_worker": "10",
            "num_workers": "8",
            "gpu_count": "8",
            "dataset": "WikiText-2",
            "eval_samples": "8 probe samples",
            "artifact_path": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew",
            "log_path": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidate_multigpu.log",
            "notes": "Smoke run only: small samples and tiny episode budget.",
        },
        {
            "run_id": "pas_policy_selection_seed3025_b407d39",
            "status": "diagnostic_only",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30 target / 0.40 stress",
            "method": "PAS policy-selection tradeoff",
            "baseline": "FF-Endpoint",
            "seed": "3025",
            "dataset": "WikiText-2",
            "artifact_path": "/workspace/ckpts/pas_policy_selection_20260521/price_of_budget_robustness_seed3025.csv",
            "commit_hash": "b407d39",
            "notes": "Protocol mismatch recorded: target from probe-side PoBR, stress from selected recheck 64.",
        },
        {
            "run_id": "pas_seed3025_final_eval30_norecon",
            "status": "diagnostic_only",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30",
            "method": "FF-Endpoint vs PAS-Slope no-reconstruction final eval",
            "baseline": "FF-Endpoint",
            "seed": "3025",
            "PPL": "FF=84.24298858642578; PAS=90.00709533691406",
            "calibration_protocol": "no reconstruction/calibration for both rules",
            "dataset": "WikiText-2",
            "eval_samples": "64",
            "wall_clock_time": "233.10833382606506 seconds",
            "artifact_path": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_eval.csv",
            "notes": "At target 30%, FF-Endpoint is better; diagnostic tradeoff evidence, not PAS main-line success.",
        },
        {
            "run_id": "pas_seed3025_final_eval40_from_recheck",
            "status": "diagnostic_only",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.40 stress",
            "method": "FF-Endpoint vs PAS-Slope matched future-budget eval",
            "baseline": "FF-Endpoint",
            "seed": "3025",
            "PPL": "FF=268.98162841796875; PAS=141.8224639892578",
            "calibration_protocol": "no reconstruction; materialized from selected-candidate recheck",
            "dataset": "WikiText-2",
            "eval_samples": "64",
            "artifact_path": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck/pas_compensation_aligned_eval_40.csv",
            "notes": "Stress-budget diagnostic: PAS-Slope better at 40%, but held-out stress budget is analysis-only.",
        },
        {
            "run_id": "pas_stress_recovery_seed3025_p1_recovery",
            "status": "partial_negative",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30 target / 0.40 heldout",
            "method": "PAS stress-recovery FFN-only ridge",
            "baseline": "raw endpoint loss",
            "seed": "3025",
            "dataset": "WikiText-2",
            "eval_samples": "64",
            "artifact_path": "/workspace/ckpts/pas_stress_recovery/recovery_analysis_opt27b_seed3025.csv",
            "notes": "Mixed/weak recovery result; do not claim PAS improves recovery.",
        },
        {
            "run_id": "pas_stress_downstream_seed3025_raw",
            "status": "partial_negative",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30",
            "method": "PAS stress raw downstream retention",
            "baseline": "L30_raw-controlled stress",
            "seed": "3025",
            "dataset": "PIQA/HellaSwag/WinoGrande/ARC/BoolQ limit100",
            "artifact_path": "/workspace/ckpts/pas_stress_recovery/downstream_analysis_opt27b_seed3025.csv",
            "notes": "Weak/mixed downstream@30 signal; exploratory only.",
        },
        {
            "run_id": "pas_local_delta_seed3025_s31_followup",
            "status": "partial_negative",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30 local probes",
            "method": "PAS local delta S31/S30.50 downstream follow-up",
            "baseline": "FF-Endpoint",
            "seed": "3025",
            "dataset": "WikiText-2 + downstream@30",
            "artifact_path": "/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_analysis/local_signal_correlation.csv",
            "notes": "Does not become a local-flatness/downstream claim; nestedness issues remain.",
        },
        {
            "run_id": "pas_overhead_seed3025_summary",
            "status": "diagnostic_only",
            "source": "docs/CLAIM_EVIDENCE.md",
            "model": "OPT-2.7B",
            "sparsity": "0.30",
            "method": "PAS overhead accounting",
            "baseline": "static checkpoint inference overhead",
            "seed": "3025",
            "gpu_hours": "candidate_search=missing; probe=0.7975969023836984; recheck=0.061435895032352875; final_eval=0.06475231495168474",
            "artifact_path": "/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead/pas_overhead_summary.csv",
            "notes": "Candidate-search and original held-out timing missing by design; no inferred GPU-hours.",
        },
    ]
    for item in curated:
        rows.append(row(**item, reproduces_pdf_row="no"))


def add_pas_radius_copied_source_rows(rows: list[dict[str, str]]) -> None:
    summary = Path("docs/pas_radius_downstream_summary_20260528.md")
    if not summary.exists():
        return
    text = summary.read_text(encoding="utf-8")
    pattern = re.compile(r"\| ([^|\n]+) \| ([^|\n]+) \| `([^`]+)` \| `([^`]+)` \|")
    for setting, label, repo_copy, source_artifact in pattern.findall(text):
        if setting == "setting":
            continue
        seed_match = re.search(r"seed(\d+)", setting)
        model = "OPT-1.3B" if "opt13b" in setting else "OPT-2.7B" if "opt27b" in setting else MISSING
        rows.append(
            row(
                run_id=f"pas_radius_copy_{clean_id(setting)}_{clean_id(label)}",
                status="diagnostic_only",
                source=str(summary),
                model=model,
                sparsity="0.30/0.35/0.40 radius/downstream",
                method=f"PAS radius/downstream copied summary: {label}",
                baseline="FF-Endpoint",
                seed=seed_match.group(1) if seed_match else MISSING,
                dataset="WikiText-2 and optional downstream@30",
                artifact_path=source_artifact,
                log_path=repo_copy,
                reproduces_pdf_row="no",
                notes="Copied server summary; useful diagnostic evidence, not a FastForward main-table raw run.",
            )
        )


def write_csv_file(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    counts = Counter(item["status"] for item in rows)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# FastForward Clean-Run Manifest 2026-06-15\n\n")
        handle.write("## Status Counts\n\n")
        handle.write("| status | count |\n| --- | --- |\n")
        for status in sorted(STATUSES):
            handle.write(f"| {status} | {counts.get(status, 0)} |\n")
        handle.write("\n## Runs And Artifacts\n\n")
        handle.write("| run_id | status | model | sparsity | method | seed | PPL | source | artifact_path | notes |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for item in rows:
            notes = item["notes"].replace("|", "/")
            handle.write(
                f"| {item['run_id']} | {item['status']} | {item['model']} | {item['sparsity']} | "
                f"{item['method']} | {item['seed']} | {item['PPL']} | {item['source']} | "
                f"{item['artifact_path']} | {notes} |\n"
            )


def write_notes(path: Path, rows: list[dict[str, str]], current_commit: str) -> None:
    counts = Counter(item["status"] for item in rows)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Manifest Generation Notes\n\n")
        handle.write(f"- generated_at_utc: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`\n")
        handle.write(f"- generator_commit: `{current_commit}`\n")
        handle.write("- generator_script: `scripts/build_fastforward_run_manifest.py`\n")
        handle.write("- output_dir: `docs/fastforward_clean_manifest_20260615/`\n\n")
        handle.write("## Status Rules\n\n")
        handle.write("- `paper_usable`: complete metadata and raw artifacts are mapped. No inspected row met this bar yet.\n")
        handle.write("- `diagnostic_only`: useful mechanism/PAS/static-stress evidence, but not final FastForward main-table evidence.\n")
        handle.write("- `partial_negative`: incomplete, weak, mixed, or negative partial PAS evidence.\n")
        handle.write("- `debug_only`: smoke tests, tiny samples/episodes, or incomplete high-PPL logs.\n")
        handle.write("- `missing_raw_artifact`: PDF or documented result without a complete raw-artifact/log mapping in inspected sources.\n\n")
        handle.write("## Important Guardrails\n\n")
        handle.write("- PDF table values are recorded as PDF claims, not clean reproductions.\n")
        handle.write("- `../opt13-log.txt` is not paper-usable because exact protocol, seed, final converged row, and model identity are incomplete.\n")
        handle.write("- PAS static/path evidence is diagnostic unless regenerated and tied to final journal tables.\n")
        handle.write("- Progressive PAS seed9025 remains partial/negative until the official replay and same-pool ablation finish on the server.\n\n")
        handle.write("## Status Counts\n\n")
        for status in sorted(STATUSES):
            handle.write(f"- {status}: `{counts.get(status, 0)}`\n")


def main() -> int:
    rows: list[dict[str, str]] = []
    current_commit = git_rev()
    add_pdf_rows(rows)
    add_opt13_log_row(rows)
    add_pas_formal_rows(rows)
    add_curated_doc_rows(rows)
    add_pas_radius_copied_source_rows(rows)

    seen = set()
    for item in rows:
        if item["run_id"] in seen:
            raise RuntimeError(f"duplicate run_id: {item['run_id']}")
        seen.add(item["run_id"])
        missing = [key for key in COLUMNS if key not in item]
        if missing:
            raise RuntimeError(f"{item['run_id']} missing columns: {missing}")
        if item["status"] not in STATUSES:
            raise RuntimeError(f"{item['run_id']} has invalid status {item['status']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_file(OUTPUT_DIR / "run_manifest.csv", rows)
    write_markdown(OUTPUT_DIR / "run_manifest.md", rows)
    write_notes(OUTPUT_DIR / "manifest_generation_notes.md", rows, current_commit)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator_script": "scripts/build_fastforward_run_manifest.py",
        "generator_commit": current_commit,
        "columns": COLUMNS,
        "status_values": sorted(STATUSES),
        "status_counts": dict(Counter(item["status"] for item in rows)),
        "runs": rows,
    }
    with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for path in [
        OUTPUT_DIR / "run_manifest.csv",
        OUTPUT_DIR / "run_manifest.md",
        OUTPUT_DIR / "run_manifest.json",
        OUTPUT_DIR / "manifest_generation_notes.md",
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
