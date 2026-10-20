#!/usr/bin/env python3
"""Run Faster R-CNN ResNet-50-FPN (COCO-pretrained, inference only) on all dataset conditions.
Usage: python src/run_fasterrcnn.py
Reads configs/paths.yaml / configs/experiment.yaml. Writes results/raw_detections/*_fasterrcnn.json
and appends timing to results/metrics/baseline_latency_metrics.csv.
"""
import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths, load_experiment
from utils_coco import (
    get_class_map, get_image_root, condition_gt_paths, merge_coco_jsons,
    save_detections, append_csv_row, log_issue, CONDITION_SPECS,
)
from measure_latency import get_gpu_name, reset_gpu_memory_stats, get_peak_gpu_memory_mb, LatencyTimer

import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform

DETECTOR_NAME = "fasterrcnn"
WEIGHTS = "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(image_size, confidence_threshold, nms_threshold, device):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=confidence_threshold)
    model.roi_heads.nms_thresh = nms_threshold
    old_transform = model.transform
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size, max_size=image_size,
        image_mean=old_transform.image_mean, image_std=old_transform.image_std,
    )
    model.to(device)
    model.eval()
    coco_categories = weights.meta["categories"]  # index 0 = "__background__"
    return model, coco_categories


def image_to_tensor(img_path, device):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


@torch.no_grad()
def run_condition(model, coco_categories, paths, exp, dataset, condition, output_name, log_path, device):
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    gt_paths = condition_gt_paths(processed_dir, dataset, condition)
    merged_gt, _ = merge_coco_jsons(gt_paths)

    if not merged_gt["images"]:
        print(f"[WARN] {output_name}: no ground-truth images found, skipping")
        return

    class_map = get_class_map(dataset)
    image_root = get_image_root(paths, dataset)
    cat_name_to_id = {c["name"]: c["id"] for c in merged_gt["categories"]}

    images = merged_gt["images"]
    valid = []
    for img in images:
        ip = os.path.join(image_root, img["file_name"])
        if os.path.isfile(ip):
            valid.append((img, ip))
        else:
            log_issue(log_path, ip, "image file missing at inference time")

    reset_gpu_memory_stats(device)
    n_dropped_class = 0

    out_dir = os.path.join(REPO_ROOT, "results", "raw_detections")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_name}_{DETECTOR_NAME}.json")
    ckpt_path = os.path.join(out_dir, f".checkpoint_{output_name}_{DETECTOR_NAME}.json")

    detections = []
    done_image_ids = set()
    if os.path.isfile(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        detections = ckpt["detections"]
        done_image_ids = set(ckpt["done_image_ids"])
        print(f"[RESUME] {output_name}: {len(done_image_ids)} images already done, resuming")

    remaining = [(img, ip) for img, ip in valid if img["id"] not in done_image_ids]

    # One image at a time: low confidence_threshold (0.001) can blow up memory/NMS cost if batched.
    for i, (img_record, ip) in enumerate(remaining):
        t = image_to_tensor(ip, device)
        if t is None:
            log_issue(log_path, ip, "unreadable image file")
            continue

        out = model([t])[0]
        image_id = img_record["id"]
        labels = out["labels"].tolist()
        boxes = out["boxes"].tolist()
        scores = out["scores"].tolist()
        for label, box, score in zip(labels, boxes, scores):
            coco_name = coco_categories[label]
            dataset_name = class_map.get(coco_name)
            if dataset_name is None or dataset_name not in cat_name_to_id:
                n_dropped_class += 1
                continue
            category_id = cat_name_to_id[dataset_name]
            x1, y1, x2, y2 = box
            detections.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
        del out, t
        done_image_ids.add(image_id)
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 500 == 0:
            ckpt_tmp = ckpt_path + ".tmp"
            with open(ckpt_tmp, "w") as f:
                json.dump({"detections": detections, "done_image_ids": list(done_image_ids)}, f)
            os.replace(ckpt_tmp, ckpt_path)
            print(f"[PROGRESS] {output_name}: {len(done_image_ids)}/{len(valid)} images "
                  f"(checkpoint saved)", flush=True)

    save_detections(out_path, detections)
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    peak_mem = get_peak_gpu_memory_mb(device)

    timer = LatencyTimer(device)
    sample = valid[:min(50, len(valid))]
    for img_record, ip in sample[:5]:
        t = image_to_tensor(ip, device)
        if t is not None:
            model([t])
    for img_record, ip in sample:
        t = image_to_tensor(ip, device)
        if t is not None:
            timer.time_call(model, [t])

    latency_csv = os.path.join(REPO_ROOT, "results", "metrics", "baseline_latency_metrics.csv")
    fieldnames = ["dataset", "condition", "detector", "checkpoint", "input_size", "batch_size",
                  "num_images_timed", "latency_ms", "FPS", "GPU_mem_MB", "GPU_name", "timestamp"]
    row = {
        "dataset": dataset, "condition": condition or "none", "detector": DETECTOR_NAME,
        "checkpoint": WEIGHTS, "input_size": paths["image_size"], "batch_size": paths["batch_size"],
        "num_images_timed": len(sample), "latency_ms": timer.mean_latency_ms(), "FPS": timer.fps(),
        "GPU_mem_MB": peak_mem, "GPU_name": get_gpu_name(device),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    append_csv_row(latency_csv, fieldnames, row)

    print(f"[OK] {out_path}: {len(detections)} detections over {len(valid)} images "
          f"(dropped {n_dropped_class} unmapped-class boxes)")


def main():
    paths = load_paths()
    exp = load_experiment()
    device = f"cuda:{paths['gpu_id']}"

    log_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{DETECTOR_NAME}_run_issues.log")
    if len(sys.argv) <= 1 and os.path.exists(log_path):
        os.remove(log_path)

    only_condition = sys.argv[1] if len(sys.argv) > 1 else None

    model, coco_categories = load_model(
        paths["image_size"], exp["confidence_threshold"], exp["nms_threshold"], device
    )

    for dataset, condition, output_name in CONDITION_SPECS:
        if only_condition and output_name != only_condition:
            continue
        out_path = os.path.join(REPO_ROOT, "results", "raw_detections", f"{output_name}_{DETECTOR_NAME}.json")
        if only_condition is None and os.path.isfile(out_path):
            print(f"[SKIP] {out_path} already exists")
            continue
        run_condition(model, coco_categories, paths, exp, dataset, condition, output_name, log_path, device)


if __name__ == "__main__":
    main()
