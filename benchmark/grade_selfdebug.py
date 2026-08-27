#!/usr/bin/env python3
"""Regrade the final artifact from each traceback-driven repair attempt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.common import REFERENCE_DIR, latest_result, read_json, timestamped_result_path, write_json
from evals.structural_grading_helpers import (
    find_key_csv, find_key_nc, has_deliverable, protocol_matches_reference,
)


def grade(row: dict) -> dict:
    if row["final_source"] == "crash":
        return row | {"selfdebug_auto_grade": "crash", "selfdebug_reason": "debug exhausted"}
    out_dir = Path(row["final_output_dir"])
    ok, reason = has_deliverable(out_dir, row["task"])
    if not ok:
        return row | {"selfdebug_auto_grade": "crash", "selfdebug_reason": reason}
    if row["mode"] == "protocol":
        ref_dir = REFERENCE_DIR / row["task"]
        match, reason = protocol_matches_reference(
            out_dir, row["task"], find_key_csv(ref_dir, row["task"]), find_key_nc(ref_dir, row["task"])
        )
        if match:
            return row | {"selfdebug_auto_grade": "success", "selfdebug_reason": reason}
    return row | {"selfdebug_auto_grade": "undetermined", "selfdebug_reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        help="scores_selfdebug_resolved_*.json from resolve_selfdebug.py (default: latest)",
    )
    args = parser.parse_args()
    source = args.results or latest_result("scores_selfdebug_resolved")
    grades = [grade(row) for row in read_json(source)]
    path = timestamped_result_path("scores_selfdebug_graded")
    write_json(path, grades)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
