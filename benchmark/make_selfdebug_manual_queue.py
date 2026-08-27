#!/usr/bin/env python3
"""Write a manual-review CSV for repaired benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.common import RESULTS_DIR, latest_result, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="scores_selfdebug_graded_*.json (default: latest)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "manual_review_selfdebug.csv")
    args = parser.parse_args()
    rows = [row for row in read_json(args.scores or latest_result("scores_selfdebug_graded"))
            if row["selfdebug_auto_grade"] == "undetermined"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "idx", "mode", "task", "model", "run", "auto_reason", "output_dir",
            "manual_grade", "notes",
        ])
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({
                "idx": idx, "mode": row["mode"], "task": row["task"],
                "model": row["model"], "run": row["run"],
                "auto_reason": row["selfdebug_reason"], "output_dir": row["final_output_dir"],
                "manual_grade": "", "notes": "",
            })
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
