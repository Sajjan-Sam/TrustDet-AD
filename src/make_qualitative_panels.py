#!/usr/bin/env python3
"""Render trusted, uncertain and rejected detections over dataset images.

Reads figures/qualitative/selected_examples.csv and dataset images. Writes PDFs
to figures/qualitative/. Missing images are logged and the corresponding panel
cells are skipped.
Run: python src/make_qualitative_panels.py"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths
from utils_coco import get_image_root, condition_gt_paths, merge_coco_jsons, log_issue

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATUS_COLORS = {"trusted": "#0ca30c", "uncertain": "#fab219", "rejected": "#d03b3b"}


def build_image_path_lookup(paths):
    """(dataset,condition,image_id) -> absolute image path, built once from processed GT files."""
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    lookup = {}
    seen_conditions = set()
    with open(os.path.join(REPO_ROOT, "figures", "qualitative", "selected_examples.csv")) as f:
        for r in csv.DictReader(f):
            seen_conditions.add((r["dataset"], r["condition"]))
    for dataset, condition in seen_conditions:
        cond_arg = None if condition == "none" else condition
        gt_paths = condition_gt_paths(processed_dir, dataset, cond_arg)
        merged_gt, _ = merge_coco_jsons(gt_paths)
        image_root = get_image_root(paths, dataset)
        for img in merged_gt["images"]:
            lookup[(dataset, condition, img["id"])] = os.path.join(image_root, img["file_name"])
    return lookup


def render_panel(rows, gate_label, out_path, image_lookup, log_path):
    n = len(rows)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    color = STATUS_COLORS[gate_label]
    n_rendered = 0
    for i, r in enumerate(rows):
        ax = axes[i]
        key = (r["dataset"], r["condition"], int(r["image_id"]))
        img_path = image_lookup.get(key)
        img = cv2.imread(img_path) if img_path else None
        if img is None:
            log_issue(log_path, str(key), f"could not load image for {gate_label} qualitative panel")
            ax.axis("off")
            ax.text(0.5, 0.5, "image unavailable", ha="center", va="center", fontsize=9)
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        x, y, w, h = [float(v) for v in r["bbox"].split(",")]
        ax.add_patch(Rectangle((x, y), w, h, linewidth=2.5, edgecolor=color, facecolor="none"))
        ax.text(x, max(0, y - 6), f"{r['class_name']} {float(r['confidence']):.2f}",
                color="white", fontsize=8, backgroundcolor=color)
        ax.set_title(f"{r['dataset']}/{r['condition']}/{r['detector']}\nscore={float(r['condition_calibrated_score']):.3f} "
                     f"({r['correctness_label']})", fontsize=7.5)
        ax.axis("off")
        n_rendered += 1

    for j in range(len(rows), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Qualitative examples: {gate_label.upper()} (condition-calibrated TrustDet-AD, test split)",
                 fontsize=12, color=color)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] wrote {out_path} ({n_rendered}/{n} images rendered)")


def main():
    paths = load_paths()
    sel_path = os.path.join(REPO_ROOT, "figures", "qualitative", "selected_examples.csv")
    if not os.path.isfile(sel_path):
        print(f"[ERROR] {sel_path} not found. Run src/select_qualitative_examples.py first.")
        sys.exit(1)

    with open(sel_path, "r") as f:
        rows = list(csv.DictReader(f))

    log_path = os.path.join(REPO_ROOT, "logs", "qualitative_panels_issues.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    image_lookup = build_image_path_lookup(paths)

    out_dir = os.path.join(REPO_ROOT, "figures", "qualitative")
    for gate_label in ["trusted", "uncertain", "rejected"]:
        subset = [r for r in rows if r["gate_label"] == gate_label]
        if not subset:
            print(f"[WARN] no selected examples for '{gate_label}'")
            continue
        out_path = os.path.join(out_dir, f"{gate_label}_examples.pdf")
        render_panel(subset, gate_label, out_path, image_lookup, log_path)


if __name__ == "__main__":
    main()
