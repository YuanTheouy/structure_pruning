#!/usr/bin/env python3
"""Collect PAS radius/downstream markdown artifacts into repo docs.

This script is meant to run on the experiment server, where /workspace/ckpts
exists. It copies raw markdown summaries and writes one compact paper-facing
index under docs/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path


SETTINGS = [
    {
        "key": "opt13b_seed4025",
        "model": "OPT-1.3B",
        "seed": "4025",
        "expected_prefix": "opt-1.3b_seed4025",
        "sources": [
            (
                "path30_35_40",
                "/workspace/ckpts/pas_informative_radius/"
                "opt-1.3b_seed4025_ff1000_growth_path30_35_40_fixed5_no6/"
                "path30_35_40_fixed5_summary.md",
            ),
            (
                "local_radius_vs_40",
                "/workspace/ckpts/pas_informative_radius/"
                "opt-1.3b_seed4025_ff1000_growth_local_radius_fixed5_no6/"
                "local_radius_vs_40_fixed5.md",
            ),
            (
                "downstream30",
                "/workspace/ckpts/pas_informative_radius/"
                "opt-1.3b_seed4025_ff1000_growth_downstream30_local_radius_fixed5_no6/"
                "local_radius_downstream30_table.md",
            ),
        ],
    },
    {
        "key": "opt13b_seed5025",
        "model": "OPT-1.3B",
        "seed": "5025",
        "expected_prefix": "opt-1.3b_seed5025",
        "sources": [
            (
                "final",
                "/workspace/ckpts/pas_informative_radius/"
                "opt-1.3b_seed5025_ff1000_growth_rep1/FINAL_SUMMARY.md",
            ),
        ],
    },
    {
        "key": "opt27b_seed7025",
        "model": "OPT-2.7B",
        "seed": "7025",
        "expected_prefix": "opt-2.7b_seed7025",
        "sources": [
            (
                "final",
                "/workspace/ckpts/pas_informative_radius/"
                "opt-2.7b_seed7025_ff1000_growth_rep1/FINAL_SUMMARY.md",
            ),
        ],
    },
    {
        "key": "opt27b_seed8025",
        "model": "OPT-2.7B",
        "seed": "8025",
        "expected_prefix": "opt-2.7b_seed8025",
        "sources": [
            (
                "final",
                "/workspace/ckpts/pas_informative_radius/"
                "opt-2.7b_seed8025_ff1000_growth_rep1/FINAL_SUMMARY.md",
            ),
        ],
    },
]

FORMAL_TABLES = [
    (
        "formal_summary",
        "/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/summary.md",
    ),
    (
        "endpoint_ambiguity",
        "/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/"
        "paper_endpoint_ambiguity_table.md",
    ),
    (
        "stress_correlation",
        "/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/"
        "paper_stress_correlation_table.md",
    ),
    (
        "selection_value",
        "/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/"
        "paper_selection_value_table.md",
    ),
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find repo root")


def parse_markdown_table(section_text: str) -> tuple[list[str], list[list[str]]]:
    header: list[str] = []
    rows: list[list[str]] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if all(set(cell) <= {"-", " "} for cell in cells):
            continue
        if not header:
            header = cells
            continue
        rows.append(cells)
    return header, rows


def extract_section(text: str, title: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(title)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_match = re.search(r"^##\s+", text[start:], flags=re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()


def candidate_prefix_ok(text: str, expected_prefix: str) -> tuple[bool, list[str]]:
    candidates = sorted(set(re.findall(r"(opt-[0-9.]+b_seed\d+_[A-Za-z0-9_.-]+)", text)))
    bad = [cand for cand in candidates if expected_prefix not in cand]
    return not bad, bad[:5]


def copy_source(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/pas_radius_downstream_20260528")
    parser.add_argument("--summary", default="docs/pas_radius_downstream_summary_20260528.md")
    parser.add_argument(
        "--use-existing-copies",
        action="store_true",
        help="Regenerate the summary from files already copied under --output-dir if /workspace sources are unavailable.",
    )
    args = parser.parse_args()

    root = repo_root()
    out_dir = root / args.output_dir
    summary_path = root / args.summary
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    contamination: list[dict[str, str]] = []
    correlation_rows: list[list[str]] = []
    selection_rows: list[list[str]] = []

    for setting in SETTINGS:
        for label, source in setting["sources"]:
            src = Path(source)
            dest = out_dir / setting["key"] / f"{label}.md"
            if copy_source(src, dest):
                source_note = source
            elif args.use_existing_copies and dest.exists():
                source_note = f"{source} (using existing repo copy)"
            else:
                missing.append(
                    {
                        "setting": setting["key"],
                        "label": label,
                        "source": source,
                    }
                )
                continue

            text = dest.read_text(encoding="utf-8", errors="replace")
            ok, bad = candidate_prefix_ok(text, setting["expected_prefix"])
            if not ok:
                contamination.append(
                    {
                        "setting": setting["key"],
                        "label": label,
                        "bad": ", ".join(bad),
                    }
                )

            copied.append(
                {
                    "setting": setting["key"],
                    "label": label,
                    "source": source_note,
                    "dest": str(dest.relative_to(root)),
                }
            )

            corr = extract_section(text, "Correlation With Downstream@30")
            if not corr:
                corr = extract_section(text, "Correlation")
            _, corr_rows = parse_markdown_table(corr)
            for cells in corr_rows:
                if len(cells) >= 5 and cells[0].startswith("S") and cells[1].isdigit():
                    correlation_rows.append(
                        [
                            setting["model"],
                            setting["seed"],
                            cells[0],
                            cells[1],
                            cells[2],
                            cells[3],
                            cells[4],
                        ]
                    )
                elif len(cells) >= 4 and cells[0].startswith("S"):
                    correlation_rows.append(
                        [
                            setting["model"],
                            setting["seed"],
                            cells[0],
                            "",
                            cells[1],
                            cells[2],
                            cells[3],
                        ]
                    )

            selection = extract_section(text, "Selection Summary")
            selection_header, parsed_selection_rows = parse_markdown_table(selection)
            has_downstream_score = "avg_pruned_score" in selection_header
            for cells in parsed_selection_rows:
                if has_downstream_score and len(cells) >= 11 and not cells[0].startswith("Oracle-Downstream"):
                    selection_rows.append(
                        [
                            setting["model"],
                            setting["seed"],
                            *cells[:11],
                        ]
                    )

    for label, source in FORMAL_TABLES:
        src = Path(source)
        dest = out_dir / "formal_tables_20260525" / f"{label}.md"
        if copy_source(src, dest):
            source_note = source
        elif args.use_existing_copies and dest.exists():
            source_note = f"{source} (using existing repo copy)"
        else:
            missing.append({"setting": "formal_tables", "label": label, "source": source})
            continue

        if dest.exists():
            copied.append(
                {
                    "setting": "formal_tables",
                    "label": label,
                    "source": source_note,
                    "dest": str(dest.relative_to(root)),
                }
            )

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# PAS Radius/Downstream Evidence Summary 2026-05-28")
    lines.append("")
    lines.append(f"Generated on server at `{now}`.")
    lines.append("")
    lines.append("Main interpretation:")
    lines.append("")
    lines.append("- `S35` is the main compression-path stress signal for future-budget fragility.")
    lines.append("- Small local radii (`S3025`, `S3050`, `S31`) can align with downstream@30 in some settings, but the signal is not stable enough to be the main claim.")
    lines.append("- Downstream@30 and 40% path robustness can conflict, so these should be reported as separate objectives.")
    lines.append("")

    lines.append("## Downstream Correlation")
    lines.append("")
    if correlation_rows:
        lines.append("| model | seed | metric | n | pearson | spearman | partial_corr_given_L30 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in correlation_rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("No downstream correlation rows were found.")
    lines.append("")

    lines.append("## Selection Summary")
    lines.append("")
    if selection_rows:
        lines.append(
            "| model | seed | rule | step | avg_pruned_score | L30 | S3025 | S3050 | S31 | S35 | L40 | Regret40 | candidate |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in selection_rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("No selection rows were found.")
    lines.append("")

    lines.append("## Copied Source Files")
    lines.append("")
    lines.append("| setting | label | repo copy | source artifact |")
    lines.append("| --- | --- | --- | --- |")
    for row in copied:
        lines.append(f"| {row['setting']} | {row['label']} | `{row['dest']}` | `{row['source']}` |")
    lines.append("")

    if missing:
        lines.append("## Missing Sources")
        lines.append("")
        lines.append("| setting | label | source artifact |")
        lines.append("| --- | --- | --- |")
        for row in missing:
            lines.append(f"| {row['setting']} | {row['label']} | `{row['source']}` |")
        lines.append("")

    lines.append("## Candidate Prefix Check")
    lines.append("")
    if contamination:
        lines.append("Potential contamination found:")
        lines.append("")
        lines.append("| setting | label | bad candidate examples |")
        lines.append("| --- | --- | --- |")
        for row in contamination:
            lines.append(f"| {row['setting']} | {row['label']} | `{row['bad']}` |")
    else:
        lines.append("All copied setting summaries passed the candidate-prefix check.")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {summary_path}")
    print(f"COPIED_DIR {out_dir}")
    if missing:
        print(f"MISSING {len(missing)} source file(s)")
    if contamination:
        print(f"WARNING contamination candidates found: {len(contamination)} file(s)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
