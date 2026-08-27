#!/usr/bin/env python3
"""Combine original structural grades with the final state of debug attempts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.common import latest_result, read_json, timestamped_result_path, write_json


def key(row: dict) -> tuple[str, str, str, int]:
    return row["mode"], row["model"], row["task"], row["run"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="scores_structural_*.json (default: latest)")
    parser.add_argument("--debug", type=Path, help="self_debug_*.json (default: latest)")
    args = parser.parse_args()
    scores = read_json(args.scores or latest_result("scores_structural"))
    debug = {key(row): row for row in read_json(args.debug or latest_result("self_debug"))}
    resolved = []
    for score in scores:
        attempt = debug.get(key(score))
        if score["auto_grade"] != "crash":
            resolved.append(score | {
                "final_source": "original", "final_output_dir": score["output_dir"],
                "debug_rounds_used": 0,
            })
        elif attempt and attempt["status"] == "fixed":
            resolved.append(score | {
                "final_source": "debug", "final_output_dir": attempt["final_output_dir"],
                "debug_rounds_used": attempt["rounds_used"],
            })
        else:
            resolved.append(score | {
                "final_source": "crash", "final_output_dir": score["output_dir"],
                "debug_rounds_used": attempt.get("rounds_used", 0) if attempt else 0,
            })
    path = timestamped_result_path("scores_selfdebug_resolved")
    write_json(path, resolved)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
