#!/usr/bin/env python3
"""Generate the 4 non-pipeline, non-qualitative figures from CSV outputs. Matplotlib only.
Usage: python src/make_plots.py
Reads results/reliability/reliability_scores_test.csv, results/ablations/coverage_precision_curve.csv,
results/ablations/selective_risk_curve.csv, results/reliability/calibration_bins.csv.
Writes figures/plots/*.pdf. Stops and reports if a required CSV is missing.
"""
import os
import sys
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(REPO_ROOT, "figures", "plots")

# Validated colorblind-safe categorical palette (light mode), fixed slot order.
CAT_COLORS = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4",
}
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
GRID_COLOR = "#c9c8c2"

METHOD_ORDER = ["confidence_only", "calibration_only", "consistency_only", "uncertainty_only",
                 "xai_stability_only", "fixed_trustdet", "condition_calibrated_trustdet"]
METHOD_COLORS = dict(zip(METHOD_ORDER, list(CAT_COLORS.values())))
METHOD_STYLES = dict(zip(METHOD_ORDER, ["-", "--", "-.", ":", "-", "--", "-"]))
DETECTOR_COLORS = {"yolov8n": CAT_COLORS["blue"], "yolov10n": CAT_COLORS["aqua"],
                    "rtdetr": CAT_COLORS["yellow"], "fasterrcnn": CAT_COLORS["green"]}


def require(relpath):
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.isfile(path):
        print(f"[ERROR] required file missing: {relpath}")
        sys.exit(1)
    return path


def read_csv(relpath):
    with open(require(relpath), "r") as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def plot_reliability_distribution():
    rows = read_csv("results/reliability/reliability_scores_test.csv")
    correct = [float(r["condition_calibrated_score"]) for r in rows if r["is_correct"] == "True"]
    incorrect = [float(r["condition_calibrated_score"]) for r in rows if r["is_correct"] != "True"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = [i / 20 for i in range(21)]
    ax.hist(incorrect, bins=bins, alpha=0.65, color=STATUS_CRITICAL, label=f"Incorrect (n={len(incorrect)})",
            density=True)
    ax.hist(correct, bins=bins, alpha=0.65, color=STATUS_GOOD, label=f"Correct (n={len(correct)})",
            density=True)
    ax.set_xlabel("Condition-Calibrated TrustDet-AD Score")
    ax.set_ylabel("Density")
    ax.set_title("Reliability Score Distribution: Correct vs.\\ Incorrect Detections (Test)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "reliability_distribution.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] wrote {out}")


def plot_coverage_vs_precision():
    rows = read_csv("results/ablations/coverage_precision_curve.csv")
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append((float(r["coverage"]), float(r["trusted_precision"])))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHOD_ORDER:
        pts = sorted(by_method.get(method, []))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=method.replace("_", " "), color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method], linewidth=2)
    ax.set_xlabel("Coverage (fraction trusted)")
    ax.set_ylabel("Trusted Precision")
    ax.set_title("Coverage vs.\\ Trusted Precision (Test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "coverage_vs_trusted_precision.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] wrote {out}")


def plot_selective_risk():
    rows = read_csv("results/ablations/selective_risk_curve.csv")
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append((float(r["coverage"]), float(r["risk"])))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHOD_ORDER:
        pts = sorted(by_method.get(method, []))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=method.replace("_", " "), color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method], linewidth=2)
    ax.set_xlabel("Coverage (fraction accepted, ranked by score)")
    ax.set_ylabel("Selective Risk (error rate)")
    ax.set_title("Selective Risk vs.\\ Coverage (Test)")
    ax.set_xlim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "selective_risk_curve.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] wrote {out}")


def plot_calibration_reliability_diagram():
    rows = read_csv("results/reliability/calibration_bins.csv")
    # ExDark, test split, all 4 detectors -- ExDark's own class space, single-dataset diagram.
    subset = [r for r in rows if r["dataset"] == "exdark" and r["split"] == "test"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, linestyle="--", linewidth=1.5, label="Perfect calibration")
    for det, color in DETECTOR_COLORS.items():
        pts = [(float(r["avg_confidence"]), float(r["accuracy"]))
               for r in subset if r["detector"] == det and r["avg_confidence"] != "" and r["count"] != "0"]
        pts = sorted(pts)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=5, color=color, label=det, linewidth=2)
    ax.set_xlabel("Mean Predicted Confidence (bin)")
    ax.set_ylabel("Empirical Accuracy (bin)")
    ax.set_title("Calibration Reliability Diagram (ExDark, Test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "calibration_reliability_diagram.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] wrote {out}")


def main():
    load_paths()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_reliability_distribution()
    plot_coverage_vs_precision()
    plot_selective_risk()
    plot_calibration_reliability_diagram()
    print("[DONE] all 4 plots written")


if __name__ == "__main__":
    main()
