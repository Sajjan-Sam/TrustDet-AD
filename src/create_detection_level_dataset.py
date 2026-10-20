#!/usr/bin/env python3
"""Match raw detections to ground truth and write per-detection labels.

Reads results/raw_detections/*.json and data/processed/*.json.
Writes results/reliability/detection_level_labels.csv. Missing combinations are
logged. Completed combinations are tracked in a sidecar file for resuming.
Run: python src/create_detection_level_dataset.py"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths
from utils_coco import CONDITION_SPECS, log_issue
from match_detections import match_condition

DETECTORS = ["yolov8n", "yolov10n", "rtdetr", "fasterrcnn"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDNAMES = ["image_id", "dataset", "condition", "detector", "split", "category_id",
              "class_name", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence",
              "correctness_label", "is_correct", "matched_gt_id", "iou"]


def main():
    load_paths()

    out_dir = os.path.join(REPO_ROOT, "results", "reliability")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "detection_level_labels.csv")
    done_marker_path = os.path.join(out_dir, ".detection_level_labels_done_combos.txt")
    log_path = os.path.join(REPO_ROOT, "logs", "match_detections_issues.log")

    # Track completed combinations alongside the append-only output CSV.
    done_combos = set()
    if os.path.isfile(done_marker_path):
        with open(done_marker_path, "r") as f:
            done_combos = set(line.strip() for line in f if line.strip())
        print(f"[RESUME] {len(done_combos)} combos already done, skipping those")
    else:
        if os.path.exists(log_path):
            os.remove(log_path)

    file_is_new = not os.path.isfile(out_path)
    total_rows = 0
    with open(out_path, "a", newline="") as f, open(done_marker_path, "a") as marker_f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if file_is_new:
            writer.writeheader()

        for dataset, condition, output_name in CONDITION_SPECS:
            for detector in DETECTORS:
                combo_key = f"{output_name}_{detector}"
                if combo_key in done_combos:
                    continue
                rows = match_condition(dataset, condition, detector, output_name)
                if rows is None:
                    log_issue(log_path, f"{output_name}_{detector}.json",
                               "raw detections file missing, skipped (not invented)")
                    continue
                if not rows:
                    log_issue(log_path, output_name, "no ground-truth images for this condition")
                    continue
                writer.writerows(rows)
                f.flush()
                total_rows += len(rows)
                n_correct = sum(1 for r in rows if r["is_correct"])
                print(f"[OK] {dataset}/{condition or 'none'}/{detector}: {len(rows)} detections "
                      f"labeled ({n_correct} correct)")
                marker_f.write(combo_key + "\n")
                marker_f.flush()

    print(f"[DONE] wrote {out_path} ({total_rows} new detection rows this run)")


if __name__ == "__main__":
    main()
