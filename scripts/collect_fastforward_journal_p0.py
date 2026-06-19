#!/usr/bin/env python3
"""Collect FastForward journal P0 run metadata, tables, and lightweight plots."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MANIFEST_FIELDS = [
    "run_id",
    "status",
    "model_name",
    "model_path",
    "sparsity",
    "workstream",
    "variant",
    "seed",
    "gpu_ids",
    "gpu_count",
    "episodes_total",
    "episodes_per_gpu",
    "eval_samples",
    "dataset_path",
    "dataset_config",
    "commit_hash",
    "candidate_dir",
    "checkpoint_raw",
    "checkpoint_calib",
    "ppl_search_endpoint",
    "ppl_raw",
    "ppl_calib",
    "search_gpu_hours",
    "calibration_eval_gpu_hours",
    "wall_clock_hours",
    "calibration_recipe",
    "log_path",
    "notes",
]


EPISODE_RE = re.compile(
    r"#(?P<episode>\d+): reward=(?P<reward>[-+0-9.eEinfnaINFNA]+), "
    r"ppl=(?P<ppl>[-+0-9.eEinfnaINFNA]+), .*?para_ratio=(?P<para>[-+0-9.eEinfnaINFNA]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="P0 output root")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def num(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any) -> str:
    value = num(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def read_first_candidate(candidate_dir: str | None) -> dict[str, Any]:
    if not candidate_dir:
        return {}
    for name in ("candidates.jsonl", "all_candidates.jsonl"):
        path = Path(candidate_dir) / name
        if not path.exists():
            continue
        best: dict[str, Any] | None = None
        best_score = float("inf")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                score = num(row.get("endpoint_logppl"), float("inf"))
                if score < best_score:
                    best = row
                    best_score = score
        return best or {}
    return {}


def parse_curve(log_path: str | None, run_id: str) -> list[dict[str, Any]]:
    if not log_path:
        return []
    path = Path(log_path)
    if path.is_dir():
        matches = sorted(path.glob("**/log.txt"))
        path = matches[-1] if matches else path
    if not path.exists() or path.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPISODE_RE.search(line)
        if match:
            pending = {
                "run_id": run_id,
                "episode": int(match.group("episode")),
                "reward": num(match.group("reward")),
                "ppl": num(match.group("ppl")),
                "para_ratio": num(match.group("para")),
                "policy": "",
            }
            rows.append(pending)
            continue
        if pending is not None and line.startswith("Policy:"):
            pending["policy"] = line.removeprefix("Policy:").strip()
            pending = None
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def maybe_plot(out_dir: Path, manifest_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_status.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    finite_curves = [row for row in curve_rows if math.isfinite(num(row.get("ppl")))]
    if finite_curves:
        plt.figure(figsize=(8, 5))
        for run_id in sorted({row["run_id"] for row in finite_curves}):
            rows = [row for row in finite_curves if row["run_id"] == run_id]
            rows.sort(key=lambda row: int(row["episode"]))
            plt.plot([row["episode"] for row in rows], [row["ppl"] for row in rows], label=run_id[-28:])
        plt.xlabel("episode")
        plt.ylabel("PPL")
        plt.yscale("log")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(out_dir / "ppl_vs_episode.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for run_id in sorted({row["run_id"] for row in finite_curves}):
            rows = [row for row in curve_rows if row["run_id"] == run_id and math.isfinite(num(row.get("reward")))]
            rows.sort(key=lambda row: int(row["episode"]))
            plt.plot([row["episode"] for row in rows], [row["reward"] for row in rows], label=run_id[-28:])
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(out_dir / "reward_vs_episode.png", dpi=180)
        plt.close()

    cost_rows = [
        row for row in manifest_rows
        if row.get("ppl_calib") and row.get("search_gpu_hours")
    ]
    if cost_rows:
        plt.figure(figsize=(6, 4))
        x = [num(row["search_gpu_hours"]) for row in cost_rows]
        y = [num(row["ppl_calib"]) for row in cost_rows]
        plt.scatter(x, y)
        for row, xi, yi in zip(cost_rows, x, y):
            plt.annotate(row["run_id"][-18:], (xi, yi), fontsize=6)
        plt.xlabel("search GPU-hours")
        plt.ylabel("calibrated PPL")
        plt.tight_layout()
        plt.savefig(out_dir / "search_cost_vs_ppl.png", dpi=180)
        plt.close()

    calib_rows = [row for row in manifest_rows if row.get("ppl_raw") and row.get("ppl_calib")]
    if calib_rows:
        labels = [row["run_id"][-18:] for row in calib_rows]
        raw = [num(row["ppl_raw"]) for row in calib_rows]
        calib = [num(row["ppl_calib"]) for row in calib_rows]
        plt.figure(figsize=(max(7, len(labels) * 0.45), 4))
        xs = list(range(len(labels)))
        plt.plot(xs, raw, marker="o", label="raw")
        plt.plot(xs, calib, marker="o", label="calib")
        plt.xticks(xs, labels, rotation=60, ha="right", fontsize=6)
        plt.ylabel("PPL")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "calibration_before_after_ppl.png", dpi=180)
        plt.close()

    heatmap_rows = [row for row in curve_rows if row.get("policy")]
    if heatmap_rows:
        last_by_run: dict[str, dict[str, Any]] = {}
        for row in heatmap_rows:
            if row["run_id"] not in last_by_run or int(row["episode"]) > int(last_by_run[row["run_id"]]["episode"]):
                last_by_run[row["run_id"]] = row
        policies = []
        labels = []
        for run_id, row in sorted(last_by_run.items()):
            try:
                policy = ast.literal_eval(row["policy"])
            except Exception:
                continue
            if isinstance(policy, list) and policy:
                policies.append([float(x) for x in policy])
                labels.append(run_id[-24:])
        if policies:
            width = max(len(policy) for policy in policies)
            padded = [policy + [float("nan")] * (width - len(policy)) for policy in policies]
            plt.figure(figsize=(10, max(3, len(padded) * 0.35)))
            plt.imshow(padded, aspect="auto", interpolation="nearest")
            plt.colorbar(label="retention")
            plt.yticks(range(len(labels)), labels, fontsize=6)
            plt.xlabel("module index")
            plt.tight_layout()
            plt.savefig(out_dir / "policy_retention_heatmap.png", dpi=180)
            plt.close()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("runs/**/run_metadata.json")):
        meta = load_json(meta_path)
        raw_meta = load_json(meta.get("raw_metadata_path"))
        calib_meta = load_json(meta.get("calib_metadata_path"))
        candidate = read_first_candidate(meta.get("candidate_dir"))

        start = num(meta.get("start_epoch"), float("nan"))
        end = num(meta.get("end_epoch"), float("nan"))
        search_end = num(meta.get("search_end_epoch"), end)
        search_start = num(meta.get("search_start_epoch"), start)
        compile_start = num(meta.get("compile_start_epoch"), float("nan"))
        compile_end = num(meta.get("compile_end_epoch"), float("nan"))
        gpu_count = int(num(meta.get("gpu_count"), 1))

        manifest_rows.append({
            "run_id": meta.get("run_id", meta_path.parent.name),
            "status": meta.get("status", "unknown"),
            "model_name": meta.get("model_name", ""),
            "model_path": meta.get("model_path", ""),
            "sparsity": meta.get("target_sparsity", ""),
            "workstream": meta.get("workstream", ""),
            "variant": meta.get("variant", ""),
            "seed": meta.get("seed", ""),
            "gpu_ids": meta.get("gpu_ids", ""),
            "gpu_count": gpu_count,
            "episodes_total": meta.get("train_episodes", ""),
            "episodes_per_gpu": meta.get("train_episodes", ""),
            "eval_samples": meta.get("n_samples", ""),
            "dataset_path": meta.get("wikitext2_path", ""),
            "dataset_config": meta.get("wikitext2_config", ""),
            "commit_hash": meta.get("commit_hash", ""),
            "candidate_dir": meta.get("candidate_dir", ""),
            "checkpoint_raw": raw_meta.get("checkpoint_path", ""),
            "checkpoint_calib": calib_meta.get("checkpoint_path", ""),
            "ppl_search_endpoint": candidate.get("endpoint_ppl", ""),
            "ppl_raw": raw_meta.get("ppl", ""),
            "ppl_calib": calib_meta.get("ppl", ""),
            "search_gpu_hours": fmt((search_end - search_start) * gpu_count / 3600.0),
            "calibration_eval_gpu_hours": fmt((compile_end - compile_start) * gpu_count / 3600.0),
            "wall_clock_hours": fmt((end - start) * gpu_count / 3600.0),
            "calibration_recipe": meta.get("calibration_recipe", ""),
            "log_path": meta.get("log_path", ""),
            "notes": meta.get("notes", ""),
        })
        curve_rows.extend(parse_curve(meta.get("log_path"), meta.get("run_id", meta_path.parent.name)))

    write_csv(out_dir / "journal_p0_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    write_csv(out_dir / "journal_p0_curves.csv", curve_rows, ["run_id", "episode", "reward", "ppl", "para_ratio", "policy"])

    with (out_dir / "journal_p0_manifest.md").open("w", encoding="utf-8") as handle:
        handle.write("# FastForward Journal P0 Manifest\n\n")
        handle.write("| run_id | status | model | sparsity | variant | seed | episodes | GPUs | search_gpu_hours | raw_PPL | calib_PPL |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in manifest_rows:
            handle.write(
                f"| {row['run_id']} | {row['status']} | {row['model_name']} | {row['sparsity']} | "
                f"{row['variant']} | {row['seed']} | {row['episodes_total']} | {row['gpu_ids']} | "
                f"{row['search_gpu_hours']} | {row['ppl_raw']} | {row['ppl_calib']} |\n"
            )

    maybe_plot(out_dir, manifest_rows, curve_rows)
    print(f"Wrote {out_dir / 'journal_p0_manifest.csv'}")
    print(f"Wrote {out_dir / 'journal_p0_manifest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

