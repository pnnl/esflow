#!/usr/bin/env python3
"""Assign deterministic crash/success/undetermined grades to benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.common import REFERENCE_DIR, latest_result, read_json, timestamped_result_path, write_json
from evals.structural_grading_helpers import (
    find_key_csv, find_key_nc, has_deliverable, protocol_matches_reference,
)


def grade_row(row: dict) -> dict:
    out_dir = Path(row["output_dir"])
    task = row["task"]
    ok, reason = has_deliverable(out_dir, task)
    result = {key: row[key] for key in ("mode", "model", "task", "run")}
    result["output_dir"] = str(out_dir)
    if not ok:
        return result | {"auto_grade": "crash", "reason": reason}
    if row["mode"] == "protocol":
        ref_dir = REFERENCE_DIR / task
        match, reason = protocol_matches_reference(
            out_dir, task, find_key_csv(ref_dir, task), find_key_nc(ref_dir, task)
        )
        if match:
            return result | {"auto_grade": "success", "reason": reason}
    return result | {"auto_grade": "undetermined", "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="benchmark_*.json (default: latest)")
    args = parser.parse_args()
    source = args.results or latest_result("benchmark")
    grades = [grade_row(row) for row in read_json(source)]
    path = timestamped_result_path("scores_structural")
    write_json(path, grades)
    for grade in ("success", "undetermined", "crash"):
        print(f"{grade}: {sum(row['auto_grade'] == grade for row in grades)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
