#!/usr/bin/env python3
"""
Merge auto grades (scores_selfdebug_graded.json) with manual review
(manual_review_selfdebug.csv), producing scores_selfdebug_final.json
with a unified new_grade field for the post-self-debug condition.
"""

import csv
import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IN_FILE = RESULTS_DIR / "scores_selfdebug_graded.json"
MANUAL_FILE = RESULTS_DIR / "manual_review_selfdebug.csv"
SINGLE_SHOT_MANUAL = RESULTS_DIR / "manual_review.csv"
OUT_FILE = RESULTS_DIR / "scores_selfdebug_final.json"


def main():
    scores = json.load(open(IN_FILE))

    manual = {}
    with open(MANUAL_FILE) as f:
        for row in csv.DictReader(f):
            grade = row.get("manual_grade", "").strip()
            if not grade:
                continue
            key = (row["mode"], row["model"], row["task"], int(row["run"]))
            manual[key] = grade

    # Fallback: single-shot manual labels apply to runs whose final source
    # after self-debug is still the original output (no crash → no debug run).
    single_shot = {}
    with open(SINGLE_SHOT_MANUAL) as f:
        for row in csv.DictReader(f):
            grade = row.get("manual_grade", "").strip()
            if not grade:
                continue
            key = (row["mode"], row["model"], row["task"], int(row["run"]))
            single_shot[key] = grade

    grade_map = {"success": "correct", "crash": "crash"}
    n_auto = n_manual = n_single = n_undetermined = 0

    for s in scores:
        key = (s["mode"], s["model"], s["task"], s["run"])
        auto = s.get("selfdebug_auto_grade") or s.get("auto_grade")
        if key in manual:
            mg = manual[key]
            s["new_grade"] = "correct" if mg == "success" else mg
            n_manual += 1
        elif auto in grade_map:
            s["new_grade"] = grade_map[auto]
            n_auto += 1
        elif s.get("final_source") == "original" and key in single_shot:
            mg = single_shot[key]
            s["new_grade"] = "correct" if mg == "success" else mg
            n_single += 1
        else:
            s["new_grade"] = "undetermined"
            n_undetermined += 1

    with open(OUT_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Merged {n_manual} selfdebug-manual + {n_single} single-shot-manual "
          f"+ {n_auto} auto grades ({n_undetermined} undetermined)")
    print(f"Wrote {OUT_FILE}")

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
