#!/usr/bin/env python3
"""
Merge auto grades (scores_structural.json) with manual review (manual_review.csv).

Produces a unified new_grade field on each score entry:
  - auto success  → "correct"
  - auto crash    → "crash"  (unless overridden by manual review)
  - manual grade  → "correct" / "silent" / "obvious" / "crash"
  - undetermined with no manual label → left as "undetermined"

Writes the updated scores back to scores_structural.json.

Usage:
    python benchmark/merge_grades.py
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_FILE = RESULTS_DIR / "scores_structural.json"
MANUAL_FILE = RESULTS_DIR / "manual_review.csv"


def main():
    with open(SCORES_FILE) as f:
        scores = json.load(f)

    # Build lookup from manual review CSV
    manual = {}
    with open(MANUAL_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            grade = row.get("manual_grade", "").strip()
            if not grade:
                continue
            key = (row["mode"], row["model"], row["task"], int(row["run"]))
            manual[key] = grade

    # Map auto_grade + manual override → new_grade
    grade_map = {
        "success": "correct",
        "crash": "crash",
    }

    n_auto = 0
    n_manual = 0
    n_undetermined = 0

    for s in scores:
        key = (s["mode"], s["model"], s["task"], s["run"])
        if key in manual:
            mg = manual[key]
            s["new_grade"] = "correct" if mg == "success" else mg
            n_manual += 1
        elif s["auto_grade"] in grade_map:
            s["new_grade"] = grade_map[s["auto_grade"]]
            n_auto += 1
        else:
            s["new_grade"] = "undetermined"
            n_undetermined += 1

    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Merged {n_manual} manual + {n_auto} auto grades "
          f"({n_undetermined} still undetermined)")
    print(f"Wrote {SCORES_FILE}")

    # Summary
    from collections import Counter
    for mode in ("protocol", "baseline"):
        c = Counter(s["new_grade"] for s in scores if s["mode"] == mode)
        total = sum(c.values())
        print(f"  {mode:10s}  correct={c.get('correct',0):3d}  "
              f"silent={c.get('silent',0):3d}  "
              f"obvious={c.get('obvious',0):3d}  "
              f"crash={c.get('crash',0):3d}  "
              f"undetermined={c.get('undetermined',0):3d}  (n={total})")


if __name__ == "__main__":
    main()
