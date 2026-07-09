#!/usr/bin/env python3
"""Resolve the final output directory for every benchmark run after self-debug.

For each (model, mode, task, run):
  - If the original grade was NOT "crash", the final output is the original
    run{N}_output directory (untouched).
  - If the original grade was "crash", walk run{N}_debug/v{MAX}..v1 and pick
    the highest-numbered version that looks like a genuine pass (no error
    markers in the execution log AND non-empty output directory).
  - If no debug version passes, the run is still a crash.

Writes results/scores_selfdebug.json — same structure as scores_structural.json
but with extra fields:
  final_output_dir   absolute path of the dir to grade
  final_source       "original" | "debug:v{K}" | "crash"
  debug_rounds_used  integer if from debug, else null
"""
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SCORES_IN = RESULTS / "scores_structural.json"
SCORES_OUT = RESULTS / "scores_selfdebug.json"

ERROR_PATTERN = re.compile(r"\b(traceback|error|exception|failed|fatal)\b", re.IGNORECASE)


def read_log(ver_dir: Path) -> str:
    for name in ("combined.txt", "stderr.txt"):
        p = ver_dir / name
        if p.exists():
            txt = p.read_text(errors="replace").strip()
            if txt:
                return txt
    return ""


def version_passed(ver_dir: Path) -> bool:
    log = read_log(ver_dir)
    if log and ERROR_PATTERN.search(log):
        return False
    out = ver_dir / "output"
    if not out.exists():
        return False
    if not any(out.iterdir()):
        return False
    return True


def resolve_crash(debug_dir: Path, max_rounds: int = 3):
    """Return (final_output_dir, source_label, rounds_used) or (None, 'crash', N).

    Scans v{max}..v1 in descending order; the first that "passed" wins.
    """
    if not debug_dir.exists():
        return None, "crash", 0
    versions = sorted(
        (p for p in debug_dir.glob("v*") if p.is_dir() and p.name[1:].isdigit()),
        key=lambda p: int(p.name[1:]),
    )
    highest = versions[-1].name if versions else "v0"
    rounds_used = int(highest[1:]) if highest.startswith("v") else 0
    for ver in reversed(versions):
        if ver.name == "v0":
            continue
        if version_passed(ver):
            return ver / "output", f"debug:{ver.name}", int(ver.name[1:])
    return None, "crash", rounds_used


def main():
    scores = json.loads(SCORES_IN.read_text())
    resolved = []
    promoted = 0
    still_crash = 0
    for row in scores:
        r = dict(row)
        mode = r["mode"]
        model = r["model"]
        task = r["task"]
        run = r["run"]
        orig_grade = r.get("new_grade") or r.get("auto_grade")
        suffix = "_protocol" if mode == "protocol" else "_baseline"
        task_dir = RESULTS / f"{model}{suffix}" / task
        orig_out = task_dir / f"run{run}_output"
        debug_dir = task_dir / f"run{run}_debug"

        if orig_grade == "crash":
            final, src, rounds = resolve_crash(debug_dir)
            if final is not None:
                r["final_output_dir"] = str(final)
                r["final_source"] = src
                r["debug_rounds_used"] = rounds
                promoted += 1
            else:
                r["final_output_dir"] = str(orig_out)
                r["final_source"] = "crash"
                r["debug_rounds_used"] = rounds
                still_crash += 1
        else:
            r["final_output_dir"] = str(orig_out)
            r["final_source"] = "original"
            r["debug_rounds_used"] = None
        resolved.append(r)

    SCORES_OUT.write_text(json.dumps(resolved, indent=2))
    print(f"Resolved {len(resolved)} runs")
    print(f"  Crashes promoted to completion: {promoted}")
    print(f"  Still crashing: {still_crash}")
    print(f"Wrote: {SCORES_OUT}")


if __name__ == "__main__":
    main()
