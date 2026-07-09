#!/usr/bin/env python3
"""Produce results/manual_review_selfdebug.csv for the 142 undetermined runs.

Columns mirror manual_review.csv so the same audit workflow applies.
Pre-fills any existing manual labels from the original manual_review.csv for
runs whose final_output is unchanged (still original, not promoted).
"""
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
IN_SCORES = RESULTS / "scores_selfdebug_graded.json"
OLD_CSV = RESULTS / "manual_review.csv"
OUT_CSV = RESULTS / "manual_review_selfdebug.csv"


def load_old_labels():
    labels = {}
    if not OLD_CSV.exists():
        return labels
    with open(OLD_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["mode"], row["task"], row["model"], int(row["run"]))
            labels[key] = (row.get("manual_grade", ""), row.get("notes", ""))
    return labels


def main():
    rows = json.loads(IN_SCORES.read_text())
    old = load_old_labels()

    undetermined = [r for r in rows if r["selfdebug_auto_grade"] == "undetermined"]

    # Split: runs that genuinely need review vs runs that carry a prior label
    queue = []          # needs manual review (promoted debug runs, or no prior label)
    carried = []        # final_source == "original" AND prior label exists
    for r in undetermined:
        key = (r["mode"], r["task"], r["model"], r["run"])
        prior = old.get(key)
        if r["final_source"] == "original" and prior and prior[0]:
            carried.append((r, prior))
        else:
            queue.append((r, prior))

    queue.sort(key=lambda x: (x[0]["mode"], x[0]["task"], x[0]["model"], x[0]["run"]))
    carried.sort(key=lambda x: (x[0]["mode"], x[0]["task"], x[0]["model"], x[0]["run"]))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "mode", "task", "model", "run",
                    "final_source", "debug_rounds_used",
                    "auto_reason", "output_dir", "manual_grade", "notes"])
        for i, (r, prior) in enumerate(queue, 1):
            pg, pn = (prior if prior else ("", ""))
            w.writerow([
                i, r["mode"], r["task"], r["model"], r["run"],
                r["final_source"],
                r.get("debug_rounds_used", ""),
                r["selfdebug_reason"],
                r["final_output_dir"],
                pg, pn,
            ])

    # Also write the carried labels to a separate file for full traceability
    carried_path = OUT_CSV.with_name("manual_review_selfdebug_carried.csv")
    with open(carried_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "task", "model", "run",
                    "final_source", "manual_grade", "notes"])
        for r, (pg, pn) in carried:
            w.writerow([r["mode"], r["task"], r["model"], r["run"],
                        r["final_source"], pg, pn])

    from collections import Counter
    by_mode = Counter(r["mode"] for r, _ in queue)
    promoted = sum(1 for r, _ in queue if r["final_source"].startswith("debug:"))
    print(f"Manual review queue: {len(queue)} runs need fresh review")
    for mode, n in sorted(by_mode.items()):
        print(f"  {mode}: {n}")
    print(f"  Promoted from debug: {promoted}")
    print(f"  No prior label:      {len(queue) - promoted}")
    print(f"Carried (prior label, unchanged output): {len(carried)}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {carried_path}")


if __name__ == "__main__":
    main()
