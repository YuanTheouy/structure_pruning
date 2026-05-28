#!/usr/bin/env python3
"""Summarize Progressive PAS search efficiency from replay CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_source_csv"] = str(path)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
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


def f(value, default=float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Progressive PAS efficiency tables")
    parser.add_argument("selection_csvs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-gap", type=float, default=0.05)
    parser.add_argument("--stage-policy", choices=["last", "best_l30"], default="last")
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], stage_policy: str) -> list[dict[str, str]]:
    out = []
    prefixes = sorted({int(f(row["prefix_step"], 0)) for row in rows})
    rules = sorted({row["rule"] for row in rows})
    for prefix in prefixes:
        prefix_rows = [row for row in rows if int(f(row["prefix_step"], 0)) == prefix]
        if not prefix_rows:
            continue
        if stage_policy == "last":
            last_stage = max(f(row["stage"]) for row in prefix_rows)
            prefix_rows = [row for row in prefix_rows if abs(f(row["stage"]) - last_stage) < 1e-9]
        for rule in rules:
            rule_rows = [row for row in prefix_rows if row["rule"] == rule]
            if not rule_rows:
                continue
            if stage_policy == "best_l30":
                picked = min(rule_rows, key=lambda row: (f(row["L30"]), f(row["prefix_step"])))
            else:
                picked = rule_rows[0]
            out.append(picked)
    return out


def write_pdf(path: Path, rows: list[dict]) -> None:
    if os.environ.get("PAS_EFFICIENCY_USE_MATPLOTLIB") != "1":
        text = "Progressive PAS efficiency summary\n\n"
        for row in rows[:40]:
            text += (
                f"prefix={row['prefix_step']} rule={row['rule']} "
                f"L30={row['L30']} Regret40={row['Regret40']}\\n"
            )
        stream = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        pdf = (
            "%PDF-1.4\n"
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
            f"4 0 obj << /Length {len(stream) + 64} >> stream\n"
            "BT /F1 10 Tf 36 756 Td 12 TL "
            f"({stream}) Tj ET\n"
            "endstream endobj\n"
            "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj\n"
            "trailer << /Root 1 0 R >>\n%%EOF\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf.encode("latin-1", errors="replace"))
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_bytes(b"%PDF-1.4\ntrailer <<>>\n%%EOF\n")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rules = ["FF-stage-endpoint", "PAS-lookahead", "Oracle-next-stage"]
    for rule in rules:
        rs = sorted([row for row in rows if row["rule"] == rule], key=lambda row: f(row["prefix_step"]))
        if not rs:
            continue
        ax.plot([f(row["prefix_step"]) for row in rs], [f(row["L30"]) for row in rs], marker="o", label=rule)
    ax.set_xlabel("Prefix episodes")
    ax.set_ylabel("L30")
    ax.set_title("Progressive PAS Lookahead Efficiency")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for value in args.selection_csvs:
        all_rows.extend(read_csv(Path(value)))
    if not all_rows:
        raise RuntimeError("No selection rows found")

    selected = select_rows(all_rows, args.stage_policy)
    max_prefix = max(f(row["prefix_step"], 0) for row in selected)
    full_candidates = [
        row for row in selected
        if row["rule"] == "FF-stage-endpoint" and f(row["prefix_step"], 0) == max_prefix
    ]
    if not full_candidates:
        full_candidates = [min(selected, key=lambda row: f(row["L30"]))]
    full = min(full_candidates, key=lambda row: f(row["L30"]))
    full_l30 = f(full["L30"])

    out_rows = []
    for row in selected:
        out_rows.append(
            {
                "source_csv": row["_source_csv"],
                "model": row["model"],
                "seed": row["seed"],
                "prefix_step": row["prefix_step"],
                "stage": row["stage"],
                "rule": row["rule"],
                "candidate_id": row["candidate_id"],
                "L30": row["L30"],
                "PPL30": row["PPL30"],
                "L30_gap_to_FF_full": f(row["L30"]) - full_l30,
                "Regret40": row["Regret40"],
                "extra_probe_evals": row["extra_probe_evals"],
                "total_eval_budget": row["total_eval_budget"],
                "gate_passed": row["gate_passed"],
            }
        )

    first_reach = {}
    for rule in sorted({row["rule"] for row in selected}):
        rs = sorted([row for row in selected if row["rule"] == rule], key=lambda row: f(row["prefix_step"]))
        hit = next((row for row in rs if f(row["L30"]) <= full_l30 + args.target_gap), None)
        first_reach[rule] = hit["prefix_step"] if hit else ""

    csv_path = output_dir / "progressive_pas_efficiency.csv"
    md_path = output_dir / "progressive_pas_efficiency.md"
    pdf_path = output_dir / "progressive_pas_efficiency.pdf"
    write_csv(csv_path, out_rows)
    write_pdf(pdf_path, out_rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Progressive PAS Efficiency\n\n")
        handle.write(f"- FF-full proxy prefix: `{fmt(max_prefix)}`\n")
        handle.write(f"- FF-full L30: `{fmt(full_l30)}`\n")
        handle.write(f"- target gap: `{args.target_gap}`\n\n")
        handle.write("## Episode Reduction\n\n")
        handle.write("| rule | first_prefix_within_target_gap |\n")
        handle.write("| --- | --- |\n")
        for rule, prefix in first_reach.items():
            handle.write(f"| {rule} | {prefix} |\n")
        handle.write("\n## Efficiency Table\n\n")
        handle.write("| prefix | rule | L30 | PPL30 | gap_to_FF_full | Regret40 | extra_probe_evals | total_eval_budget | candidate |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in out_rows:
            handle.write(
                f"| {row['prefix_step']} | {row['rule']} | {row['L30']} | {row['PPL30']} | "
                f"{fmt(row['L30_gap_to_FF_full'])} | {row['Regret40']} | {row['extra_probe_evals']} | "
                f"{row['total_eval_budget']} | {row['candidate_id']} |\n"
            )

    print(f"WROTE {csv_path}")
    print(f"WROTE {md_path}")
    print(f"WROTE {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
