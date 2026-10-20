#!/usr/bin/env python3
"""Convert ExDark annotations to COCO-style train, validation and test files.

Uses a custom 70/15/15 split within each class folder, shuffled with the configured
seed. This implementation does not read a supplied dataset split file.
Run: python src/prepare_exdark.py"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2

from config_loader import load_paths, load_experiment
from utils_coco import COCOBuilder, log_issue

EXDARK_CLASSES = sorted([
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
])


def parse_annotation(ann_path):
    """Returns list of (label, x, y, w, h). Raises ValueError on corrupt content."""
    boxes = []
    with open(ann_path, "r", errors="replace") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("empty annotation file")
    for line in lines[1:]:  # skip "% bbGt version=3" header
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"malformed line: {line!r}")
        label = parts[0]
        try:
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            raise ValueError(f"non-numeric bbox in line: {line!r}")
        boxes.append((label, x, y, w, h))
    return boxes


def main():
    paths = load_paths()
    exp = load_experiment()
    seed = exp["random_seed"]

    exdark_root = paths["exdark_root"]
    images_root = os.path.join(exdark_root, "images")
    ann_root = os.path.join(exdark_root, "annotations")

    if not os.path.isdir(images_root):
        print(f"[ERROR] ExDark images folder not found: {images_root}")
        sys.exit(1)
    if not os.path.isdir(ann_root):
        print(f"[ERROR] ExDark annotations folder not found: {ann_root}")
        sys.exit(1)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(repo_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "exdark_prep_issues.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    class_folders = sorted([
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    ])
    if not class_folders:
        print(f"[ERROR] No class folders found under {images_root}")
        sys.exit(1)

    rng = random.Random(seed)
    split_samples = {"train": [], "val": [], "test": []}
    n_missing_ann = 0
    n_corrupt_ann = 0
    n_unreadable_img = 0

    for cls in class_folders:
        img_dir = os.path.join(images_root, cls)
        ann_dir = os.path.join(ann_root, cls)
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
        ])

        valid_samples = []
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            ann_path = os.path.join(ann_dir, img_file + ".txt")

            if not os.path.isfile(ann_path):
                log_issue(log_path, img_path, "missing annotation file")
                n_missing_ann += 1
                continue

            try:
                boxes = parse_annotation(ann_path)
            except ValueError as e:
                log_issue(log_path, ann_path, f"corrupted annotation: {e}")
                n_corrupt_ann += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                log_issue(log_path, img_path, "unreadable image file")
                n_unreadable_img += 1
                continue
            height, width = img.shape[:2]

            valid_samples.append({
                "file_name": f"{cls}/{img_file}",
                "width": width,
                "height": height,
                "boxes": boxes,
            })

        rng.shuffle(valid_samples)
        n = len(valid_samples)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        split_samples["train"].extend(valid_samples[:n_train])
        split_samples["val"].extend(valid_samples[n_train:n_train + n_val])
        split_samples["test"].extend(valid_samples[n_train + n_val:])

    for split_name, samples in split_samples.items():
        builder = COCOBuilder()
        for cls in EXDARK_CLASSES:
            builder.add_category(cls)

        for sample in samples:
            img_id = builder.add_image(sample["file_name"], sample["width"], sample["height"])
            for label, x, y, w, h in sample["boxes"]:
                if label not in EXDARK_CLASSES:
                    log_issue(log_path, sample["file_name"], f"unknown class label skipped: {label}")
                    continue
                if w <= 0 or h <= 0:
                    log_issue(log_path, sample["file_name"], f"invalid box size skipped: {(x, y, w, h)}")
                    continue
                cat_id = builder.add_category(label)
                builder.add_annotation(img_id, cat_id, [x, y, w, h])

        out_path = os.path.join(out_dir, f"exdark_{split_name}.json")
        builder.save(out_path)
        print(f"[OK] {out_path}: {len(builder.images)} images, {len(builder.annotations)} annotations")

    print(f"[SUMMARY] missing_annotations={n_missing_ann} corrupted_annotations={n_corrupt_ann} unreadable_images={n_unreadable_img}")
    if n_missing_ann or n_corrupt_ann or n_unreadable_img:
        print(f"[SUMMARY] see {log_path} for details")


if __name__ == "__main__":
    main()
