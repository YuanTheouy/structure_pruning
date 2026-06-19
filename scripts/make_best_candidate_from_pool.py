#!/usr/bin/env python3
"""Create a compile-compatible best_candidate.json from a candidate pool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="endpoint", choices=["endpoint"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def endpoint_score(row: dict[str, Any]) -> tuple[float, int, str]:
    if row.get("endpoint_logppl") not in (None, ""):
        score = float(row["endpoint_logppl"])
    elif row.get("endpoint_ppl") not in (None, ""):
        ppl = float(row["endpoint_ppl"])
        score = math.log(ppl) if ppl > 0 else float("inf")
    else:
        score = float("inf")
    return score, int(row.get("step") or row.get("episode") or 0), str(row.get("candidate_id", ""))


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    rows = read_jsonl(candidate_dir / "candidates.jsonl")
    if not rows:
        rows = read_jsonl(candidate_dir / "all_candidates.jsonl")
    if not rows:
        raise SystemExit(f"No candidates found under {candidate_dir}")

    rows = [row for row in rows if math.isfinite(endpoint_score(row)[0])]
    if not rows:
        raise SystemExit(f"No finite endpoint candidates found under {candidate_dir}")

    best = min(rows, key=endpoint_score)
    payload = {
        "selected_mode": args.mode,
        "selected_score": endpoint_score(best)[0],
        "candidate": best,
        "probe_row": {
            "candidate_id": best.get("candidate_id", ""),
            "score_path": best.get("score_path", ""),
            "endpoint_ppl": best.get("endpoint_ppl", ""),
            "endpoint_logppl": best.get("endpoint_logppl", ""),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output}: {best.get('candidate_id')} endpoint_logppl={endpoint_score(best)[0]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

