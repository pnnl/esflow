#!/usr/bin/env python3
"""
Generate 4-triangle heatmap figures for benchmark results.

Each cell = one model × task combination, split into 4 triangles (4 runs).
Color scheme (danger-ranked):
  Green  — Correct: matches reference (exact or within 1%)
  Yellow — Subtly wrong: plausible output, right format/pattern, but >1% error
           (most dangerous — looks right, isn't)
  Orange — Obviously wrong: wrong output type, blank maps, empty stats
           (less dangerous — anyone would catch it)
  Red    — Crash: runtime error, no output

Reads from benchmark/results/scores.json.
Uses 'c_grade' field: "correct", "subtle", "obvious", or null (crash).

Usage:
  python benchmark/plot_results.py
  python benchmark/plot_results.py --mode protocol
  python benchmark/plot_results.py --mode baseline
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_FILE = RESULTS_DIR / "scores.json"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Colors (danger-ranked: green=safe, yellow=subtle danger, orange=obvious mistake, red=broken)
GREEN  = "#4CAF50"   # Correct (exact or within 1%)
YELLOW = "#FFEB3B"   # Subtly wrong (most dangerous — looks right, isn't)
ORANGE = "#FF9800"   # Obviously wrong (anyone would catch it)
RED    = "#F44336"   # Crash (runtime error)
GRAY   = "#E0E0E0"   # not scored yet

# Display names
MODEL_DISPLAY = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5": "GPT-5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "o4-mini": "o4-mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "phi-4": "Phi-4",
}

# Desired model order (top to bottom)
MODEL_ORDER = [
    "claude-opus-4-6",
    "gpt-5",
    "gemini-2.5-flash",
    "o4-mini",
    "claude-haiku-4-5-20251001",
    "phi-4",
]

TASK_DISPLAY = {
    "task_01_obs_summary": "T1",
    "task_02_seasonal_runoff": "T2",
    "task_03_et_benchmark": "T3",
    "task_05_basin_streamflow": "T4",
    "task_04_streamflow_fdc": "T5",
    "task_06_water_balance": "T6",
    "task_07_integrated_diagnostic": "T7",
}

TASK_ORDER = [
    "task_01_obs_summary",
    "task_02_seasonal_runoff",
    "task_03_et_benchmark",
    "task_05_basin_streamflow",
    "task_04_streamflow_fdc",
    "task_06_water_balance",
    "task_07_integrated_diagnostic",
]


def get_color(entry):
    """Determine triangle color from c_grade field.

    c_grade values (danger-ranked):
      "correct"  → GREEN  (matches reference, exact or within 1%)
      "subtle"   → YELLOW (plausible but wrong — most dangerous)
      "obvious"  → ORANGE (obviously wrong output — anyone would catch)
      None + x_pass=False → RED (crash)
      None + x_pass=True  → GRAY (not yet scored)
    """
    if entry is None:
        return GRAY

    x_pass = entry.get("x_pass")
    c_grade = entry.get("c_grade")

    # Crash — runtime error (red, least dangerous)
    if x_pass is False:
        return RED

    # Graded outcomes
    if c_grade == "correct":
        return GREEN
    if c_grade == "subtle":
        return YELLOW
    if c_grade == "obvious":
        return ORANGE

    # Not yet scored
    return GRAY


def draw_four_triangles(ax, x, y, size, colors):
    """Draw a square at (x, y) divided into 4 triangles.

    Triangle layout (matching 4 runs):
        Run 1 = top
        Run 2 = right
        Run 3 = bottom
        Run 4 = left

    (x, y) is bottom-left corner, size is side length.
    """
    cx, cy = x + size / 2, y + size / 2  # center
    corners = {
        "tl": (x, y + size),
        "tr": (x + size, y + size),
        "br": (x + size, y),
        "bl": (x, y),
    }

    # Top triangle (run 1): tl -> tr -> center
    tri_top = plt.Polygon(
        [corners["tl"], corners["tr"], (cx, cy)],
        facecolor=colors[0], edgecolor="white", linewidth=1.0
    )
    # Right triangle (run 2): tr -> br -> center
    tri_right = plt.Polygon(
        [corners["tr"], corners["br"], (cx, cy)],
        facecolor=colors[1], edgecolor="white", linewidth=1.0
    )
    # Bottom triangle (run 3): br -> bl -> center
    tri_bottom = plt.Polygon(
        [corners["br"], corners["bl"], (cx, cy)],
        facecolor=colors[2], edgecolor="white", linewidth=1.0
    )
    # Left triangle (run 4): bl -> tl -> center
    tri_left = plt.Polygon(
        [corners["bl"], corners["tl"], (cx, cy)],
        facecolor=colors[3], edgecolor="white", linewidth=1.0
    )

    ax.add_patch(tri_top)
    ax.add_patch(tri_right)
    ax.add_patch(tri_bottom)
    ax.add_patch(tri_left)


def draw_c_score(ax, x, y, size, entries):
    """Placeholder — score labels removed for clarity."""
    pass


def plot_heatmap(scores, mode, output_path):
    """Generate the 4-triangle heatmap for a given mode."""
    # Filter scores
    mode_scores = [s for s in scores if s["mode"] == mode]
    if not mode_scores:
        print(f"No scores for mode '{mode}'")
        return

    # Get available tasks and models
    available_tasks = sorted(set(s["task"] for s in mode_scores))
    tasks = [t for t in TASK_ORDER if t in available_tasks]
    available_models = sorted(set(s["model"] for s in mode_scores))
    models = [m for m in MODEL_ORDER if m in available_models]

    n_models = len(models)
    n_tasks = len(tasks)

    if n_models == 0 or n_tasks == 0:
        print(f"No data to plot for mode '{mode}'")
        return

    # Build lookup: (model, task, run) -> entry
    lookup = {}
    for s in mode_scores:
        key = (s["model"], s["task"], s["run"])
        lookup[key] = s

    # Figure setup
    cell_size = 1.0
    margin_left = 2.5
    margin_bottom = 1.2
    margin_top = 0.8
    margin_right = 0.3

    fig_w = margin_left + n_tasks * cell_size + margin_right
    fig_h = margin_bottom + n_models * cell_size + margin_top

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Draw cells
    for row_idx, model in enumerate(models):
        y = (n_models - 1 - row_idx) * cell_size
        for col_idx, task in enumerate(tasks):
            x = col_idx * cell_size

            entries = [lookup.get((model, task, run)) for run in range(1, 5)]
            colors = [get_color(e) for e in entries]

            draw_four_triangles(ax, x, y, cell_size, colors)
            draw_c_score(ax, x, y, cell_size, entries)

    # Axes setup
    ax.set_xlim(-0.05, n_tasks * cell_size + 0.05)
    ax.set_ylim(-0.05, n_models * cell_size + 0.05)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Task labels (top)
    for col_idx, task in enumerate(tasks):
        label = TASK_DISPLAY.get(task, task)
        ax.text(col_idx * cell_size + cell_size / 2, -0.15,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Model labels (left)
    for row_idx, model in enumerate(models):
        label = MODEL_DISPLAY.get(model, model)
        ax.text(-0.15, row_idx * cell_size + cell_size / 2,
                label, ha="right", va="center", fontsize=10)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    mode_label = "Protocol (YAML Workflow)" if mode == "protocol" else "Baseline (Python Code-Gen)"
    ax.set_title(mode_label, fontsize=13, fontweight="bold", pad=15)

    # Legend
    legend_elements = [
        patches.Patch(facecolor=GREEN, edgecolor="gray", label="Success"),
        patches.Patch(facecolor=YELLOW, edgecolor="gray",
                      label="Silent failure"),
        patches.Patch(facecolor=ORANGE, edgecolor="gray",
                      label="Obvious failure"),
        patches.Patch(facecolor=RED, edgecolor="gray",
                      label="Crash"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8,
              frameon=True, fancybox=True)

    # Run number annotation (small, in corner)
    # Show which triangle = which run
    inset_x = n_tasks * cell_size - 0.1
    inset_y = n_models * cell_size - 0.1
    ax.text(inset_x, inset_y,
            "△top=R1  ▷right=R2\n▽bot=R3  ◁left=R4",
            ha="right", va="bottom", fontsize=6, color="gray",
            fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined(scores, output_path):
    """Generate side-by-side protocol + baseline figure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax_idx, (ax, mode) in enumerate(zip(axes, ["protocol", "baseline"])):
        mode_scores = [s for s in scores if s["mode"] == mode]
        if not mode_scores:
            ax.text(0.5, 0.5, f"No {mode} data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        available_tasks = sorted(set(s["task"] for s in mode_scores))
        tasks = [t for t in TASK_ORDER if t in available_tasks]
        available_models = sorted(set(s["model"] for s in mode_scores))
        models = [m for m in MODEL_ORDER if m in available_models]

        n_models = len(models)
        n_tasks = len(tasks)

        lookup = {}
        for s in mode_scores:
            lookup[(s["model"], s["task"], s["run"])] = s

        cell_size = 1.0

        for row_idx, model in enumerate(models):
            y = row_idx * cell_size
            for col_idx, task in enumerate(tasks):
                x = col_idx * cell_size
                entries = [lookup.get((model, task, run)) for run in range(1, 5)]
                colors = [get_color(e) for e in entries]
                draw_four_triangles(ax, x, y, cell_size, colors)
                draw_c_score(ax, x, y, cell_size, entries)

        ax.set_xlim(-0.05, n_tasks * cell_size + 0.05)
        ax.set_ylim(-0.5, n_models * cell_size + 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # Task labels (below the grid — y-axis is inverted)
        for col_idx, task in enumerate(tasks):
            label = TASK_DISPLAY.get(task, task)
            ax.text(col_idx * cell_size + cell_size / 2,
                    n_models * cell_size + 0.15,
                    label, ha="center", va="top", fontsize=11,
                    fontweight="bold")

        # Model labels (only on left panel)
        if ax_idx == 0:
            for row_idx, model in enumerate(models):
                label = MODEL_DISPLAY.get(model, model)
                ax.text(-0.15, row_idx * cell_size + cell_size / 2,
                        label, ha="right", va="center", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        mode_label = ("(a) Protocol (YAML Workflow)" if mode == "protocol"
                      else "(b) Baseline (Python Code-Gen)")
        ax.set_title(mode_label, fontsize=12, fontweight="bold", pad=12)

    # Shared legend at bottom
    legend_elements = [
        patches.Patch(facecolor=GREEN, edgecolor="gray", label="Success"),
        patches.Patch(facecolor=YELLOW, edgecolor="gray",
                      label="Silent failure"),
        patches.Patch(facecolor=ORANGE, edgecolor="gray",
                      label="Obvious failure"),
        patches.Patch(facecolor=RED, edgecolor="gray",
                      label="Crash"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.07, 0.95, 1])  # leave room: bottom for legend+arrow, right for arrow

    # --- Gradient arrows (after tight_layout so axes positions are final) ---
    from matplotlib.patches import FancyArrowPatch

    # Per-panel arrows for both task complexity and model capability
    for ax in axes:
        bbox = ax.get_position()

        # Task complexity arrow (horizontal, below task labels)
        task_y = bbox.y0 - 0.04
        arrow_task = FancyArrowPatch(
            (bbox.x0 + 0.02, task_y), (bbox.x1 - 0.02, task_y),
            arrowstyle="->, head_width=4, head_length=4",
            color="0.5", lw=1.5, transform=fig.transFigure,
            clip_on=False)
        fig.patches.append(arrow_task)
        fig.text((bbox.x0 + bbox.x1) / 2, task_y + 0.018,
                 "Task complexity", ha="center", va="bottom",
                 fontsize=9, color="0.5", style="italic")

        # Model capability arrow (vertical, right edge)
        arrow_x = bbox.x1 + 0.01
        arrow_top = bbox.y1 - 0.04
        arrow_bot = bbox.y0 + 0.04
        arrow_model = FancyArrowPatch(
            (arrow_x, arrow_top), (arrow_x, arrow_bot),
            arrowstyle="->, head_width=4, head_length=4",
            color="0.5", lw=1.5, transform=fig.transFigure,
            clip_on=False)
        fig.patches.append(arrow_model)
        fig.text(arrow_x + 0.022, (arrow_top + arrow_bot) / 2,
                 "Model capability", ha="center", va="center",
                 fontsize=9, color="0.5", rotation=270, style="italic")

    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-triangle heatmap benchmark figures")
    parser.add_argument("--mode", choices=["protocol", "baseline", "combined"],
                        default="combined",
                        help="Which figure to generate (default: combined)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: auto)")
    args = parser.parse_args()

    if not SCORES_FILE.exists():
        print(f"ERROR: {SCORES_FILE} not found. Run benchmarks first.")
        return

    scores = json.load(open(SCORES_FILE))

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "combined":
        out = args.output or str(FIGURES_DIR / "benchmark_results.png")
        plot_combined(scores, out)
    else:
        out = args.output or str(FIGURES_DIR / f"benchmark_{args.mode}.png")
        plot_heatmap(scores, args.mode, out)


if __name__ == "__main__":
    main()
