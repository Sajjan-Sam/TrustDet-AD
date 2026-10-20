#!/usr/bin/env python3
"""Summarize the processed annotation files available in data/processed/.

Run: python src/dataset_stats.py"""
import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths

# (dataset, split, condition, processed_filename)
FILES = [
    ("exdark", "train", "none", "exdark_train.json"),
    ("exdark", "val", "none", "exdark_val.json"),
    ("exdark", "test", "none", "exdark_test.json"),
    ("bdd100k", "val", "daytime", "bdd100k_daytime_val.json"),
    ("bdd100k", "test", "daytime", "bdd100k_daytime_test.json"),
    ("bdd100k", "val", "night", "bdd100k_night_val.json"),
    ("bdd100k", "test", "night", "bdd100k_night_test.json"),
    ("bdd100k", "val", "rain", "bdd100k_rain_val.json"),
    ("bdd100k", "test", "rain", "bdd100k_rain_test.json"),
    ("bdd100k", "val", "fog", "bdd100k_fog_val.json"),
    ("bdd100k", "test", "fog", "bdd100k_fog_test.json"),
]


def compute_stats(coco_path):
    with open(coco_path, "r") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    num_images = len(data["images"])
    num_boxes = len(data["annotations"])

    class_counts = {}
    for ann in data["annotations"]:
        name = cat_id_to_name.get(ann["category_id"], "unknown")
        class_counts[name] = class_counts.get(name, 0) + 1

    num_classes = len(class_counts)
    avg_boxes_per_image = round(num_boxes / num_images, 4) if num_images else 0.0
    class_distribution = ";".join(f"{k}:{v}" for k, v in sorted(class_counts.items()))

    return {
        "num_images": num_images,
        "num_boxes": num_boxes,
        "num_classes": num_classes,
        "class_distribution": class_distribution,
        "avg_boxes_per_image": avg_boxes_per_image,
    }


def main():
    load_paths()  # validates configs/paths.yaml is well-formed

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(repo_root, "data", "processed")
    metrics_dir = os.path.join(repo_root, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    rows = []
    skipped = []
    for dataset, split, condition, fname in FILES:
        fpath = os.path.join(processed_dir, fname)
        if not os.path.isfile(fpath):
            skipped.append(fpath)
            continue
        stats = compute_stats(fpath)
        rows.append({
            "dataset": dataset,
            "split": split,
            "condition": condition,
            **stats,
        })

    if not rows:
        print("[ERROR] No processed dataset JSON files found. Run prepare_exdark.py / prepare_bdd100k.py first.")
        sys.exit(1)

    out_path = os.path.join(metrics_dir, "dataset_statistics.csv")
    fieldnames = ["dataset", "split", "condition", "num_images", "num_boxes",
                  "num_classes", "class_distribution", "avg_boxes_per_image"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {out_path} ({len(rows)} rows)")
    if skipped:
        print(f"[INFO] skipped {len(skipped)} not-yet-generated files:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
