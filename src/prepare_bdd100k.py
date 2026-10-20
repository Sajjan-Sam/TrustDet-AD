#!/usr/bin/env python3
"""Prepare COCO annotations for BDD100K daytime, night, rain and fog subsets.

Uses official det_val.json labels. Each condition is shuffled with the configured
seed and split into validation and test halves; these are custom evaluation
splits, not the official BDD100K test split. Conditions may overlap.
Run: python src/prepare_bdd100k.py"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2

from config_loader import load_paths, load_experiment
from utils_coco import COCOBuilder, log_issue

CONDITIONS = {
    "daytime": ("timeofday", "daytime"),
    "night": ("timeofday", "night"),
    "rain": ("weather", "rainy"),
    "fog": ("weather", "foggy"),
}


def main():
    paths = load_paths()
    exp = load_experiment()
    seed = exp["random_seed"]

    bdd_root = paths["bdd100k_root"]
    labels_path = os.path.join(bdd_root, "labels", "det_20", "det_val.json")
    images_dir = os.path.join(bdd_root, "images", "100k", "val")

    if not os.path.isfile(labels_path):
        print(f"[ERROR] BDD100K val labels not found: {labels_path}")
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"[ERROR] BDD100K val images folder not found: {images_dir}")
        sys.exit(1)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(repo_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "bdd100k_prep_issues.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    with open(labels_path, "r") as f:
        entries = json.load(f)

    all_labels = sorted({
        lbl["category"]
        for entry in entries
        for lbl in entry.get("labels", [])
        if "category" in lbl and lbl.get("box2d") is not None
    })
    if not all_labels:
        print(f"[ERROR] No category labels found in {labels_path}")
        sys.exit(1)

    rng = random.Random(seed)
    n_missing_img = 0
    n_missing_attr = 0
    n_invalid_box = 0

    for cond_name, (attr_key, attr_val) in CONDITIONS.items():
        matched = [e for e in entries if e.get("attributes", {}).get(attr_key) == attr_val]

        samples = []
        for entry in matched:
            file_name = entry.get("name")
            if not file_name:
                log_issue(log_path, str(entry), "entry missing 'name' field")
                continue

            img_path = os.path.join(images_dir, file_name)
            if not os.path.isfile(img_path):
                log_issue(log_path, img_path, "missing image file")
                n_missing_img += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                log_issue(log_path, img_path, "unreadable image file")
                n_missing_img += 1
                continue
            height, width = img.shape[:2]

            boxes = []
            for lbl in entry.get("labels", []):
                category = lbl.get("category")
                box2d = lbl.get("box2d")
                if category is None or box2d is None:
                    continue
                x1, y1, x2, y2 = box2d.get("x1"), box2d.get("y1"), box2d.get("x2"), box2d.get("y2")
                if None in (x1, y1, x2, y2):
                    log_issue(log_path, file_name, f"malformed box2d for category {category}")
                    n_invalid_box += 1
                    continue
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    log_issue(log_path, file_name, f"invalid box size for category {category}: w={w} h={h}")
                    n_invalid_box += 1
                    continue
                boxes.append((category, x1, y1, w, h))

            samples.append({"file_name": file_name, "width": width, "height": height, "boxes": boxes})

        if not samples:
            log_issue(log_path, cond_name, "no matching images found for this condition")
            print(f"[WARN] condition '{cond_name}': 0 matching images")
            continue

        rng.shuffle(samples)
        half = len(samples) // 2
        split_map = {"val": samples[:half], "test": samples[half:]}

        for split_name, split_samples in split_map.items():
            builder = COCOBuilder()
            for cat in all_labels:
                builder.add_category(cat)

            for sample in split_samples:
                img_id = builder.add_image(
                    f"100k/val/{sample['file_name']}", sample["width"], sample["height"]
                )
                for category, x, y, w, h in sample["boxes"]:
                    cat_id = builder.add_category(category)
                    builder.add_annotation(img_id, cat_id, [x, y, w, h])

            out_path = os.path.join(out_dir, f"bdd100k_{cond_name}_{split_name}.json")
            builder.save(out_path)
            print(f"[OK] {out_path}: {len(builder.images)} images, {len(builder.annotations)} annotations")

    print(f"[SUMMARY] missing_images={n_missing_img} missing_attr_entries={n_missing_attr} invalid_boxes={n_invalid_box}")
    if n_missing_img or n_missing_attr or n_invalid_box:
        print(f"[SUMMARY] see {log_path} for details")


if __name__ == "__main__":
    main()
