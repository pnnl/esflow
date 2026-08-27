#!/usr/bin/env python3
"""Merge manual labels with post-self-debug automatic grades."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.common import RESULTS_DIR, latest_result, read_json, timestamped_result_path, write_json


def resolve_final_grade(auto_grade: str, manual_grade: str | None) -> str:
    """Resolve final_grade: manual override > auto success/crash > undetermined."""
    automatic = {"success": "success", "crash": "crash"}.get(auto_grade)
    return manual_grade or automatic or "undetermined"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="scores_selfdebug_graded_*.json (default: latest)")
    parser.add_argument("--manual", type=Path, default=RESULTS_DIR / "manual_review_selfdebug.csv")
    args = parser.parse_args()
    manual = {}
    with args.manual.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["mode"], row["model"], row["task"], int(row["run"]))
            manual[key] = row["manual_grade"].strip().lower() or None
    final = []
    for row in read_json(args.scores or latest_result("scores_selfdebug_graded")):
        key = (row["mode"], row["model"], row["task"], row["run"])
        final.append(row | {"final_grade": resolve_final_grade(row["selfdebug_auto_grade"], manual.get(key))})
    path = timestamped_result_path("scores_selfdebug_final")
    write_json(path, final)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
