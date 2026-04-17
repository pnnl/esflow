#!/usr/bin/env python3
"""Apply the original 3-step grading workflow to post-self-debug results.

Uses the final output directory from scores_selfdebug.json (which is either the
original run output, or the highest passing debug version).

Grades produced (same taxonomy as structural_grading.py):
  crash          — final deliverable still missing after debug attempts
  success        — protocol only: numerical match against reference (rtol=1%)
  undetermined   — completed but awaiting manual review (silent vs. obvious)

Writes results/scores_selfdebug_graded.json.
"""
import json
from pathlib import Path
from collections import Counter

from structural_grading import (
    DELIVERABLE, RTOL, TASKS, MODELS, REF_MODEL, REF_RUN,
    has_deliverable, find_key_csv, find_key_nc, protocol_matches_reference,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_IN = RESULTS_DIR / "scores_selfdebug.json"
SCORES_OUT = RESULTS_DIR / "scores_selfdebug_graded.json"


def main():
    # Reference files
    ref_files = {}
    for task in TASKS:
        ref_dir = RESULTS_DIR / f"{REF_MODEL}_protocol" / task / f"run{REF_RUN}_output"
        ref_files[task] = {
            "csv": find_key_csv(ref_dir, task),
            "nc":  find_key_nc(ref_dir, task),
        }

    rows = json.loads(SCORES_IN.read_text())
    out = []
    for r in rows:
        mode, task = r["mode"], r["task"]
        final = Path(r["final_output_dir"])

        if r["final_source"] == "crash":
            grade, reason = "crash", "deliverable still missing after self-debug"
        else:
            ok, why = has_deliverable(final, task)
            if not ok:
                grade, reason = "crash", f"final_output missing deliverable: {why}"
            elif mode == "protocol":
                match, why2 = protocol_matches_reference(
                    final, task, ref_files[task]["csv"], ref_files[task]["nc"]
                )
                if match:
                    grade, reason = "success", why2
                else:
                    grade, reason = "undetermined", f"protocol non-match: {why2}"
            else:
                grade, reason = "undetermined", "baseline non-crash, awaiting manual review"

        r2 = dict(r)
        r2["selfdebug_auto_grade"] = grade
        r2["selfdebug_reason"] = reason
        out.append(r2)

    SCORES_OUT.write_text(json.dumps(out, indent=2))

    # Summaries
    print(f"Post-self-debug auto-grading (3-step, rtol={RTOL})")
    print("=" * 80)
    for mode in ("protocol", "baseline"):
        c = Counter(r["selfdebug_auto_grade"] for r in out if r["mode"] == mode)
        n = sum(c.values())
        print(f"  {mode:9s}  crash={c.get('crash',0):3d}  "
              f"success={c.get('success',0):3d}  "
              f"undetermined={c.get('undetermined',0):3d}  (n={n})")

    print("\nPer-model:")
    for mode in ("protocol", "baseline"):
        print(f"  {mode}:")
        for model in MODELS:
            c = Counter(r["selfdebug_auto_grade"] for r in out
                        if r["mode"] == mode and r["model"] == model)
            print(f"    {model:32s}  crash={c.get('crash',0):2d}  "
                  f"success={c.get('success',0):2d}  "
                  f"undetermined={c.get('undetermined',0):2d}")

    n_undet = sum(1 for r in out if r["selfdebug_auto_grade"] == "undetermined")
    print(f"\nUndetermined runs awaiting manual review: {n_undet}")
    print(f"Wrote: {SCORES_OUT}")


if __name__ == "__main__":
    main()
