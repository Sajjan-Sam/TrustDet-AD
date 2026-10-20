#!/usr/bin/env python3
"""Run YOLOv10n (COCO-pretrained, inference only) on all dataset conditions.
Usage: python src/run_yolov10.py
Reads configs/paths.yaml / configs/experiment.yaml. Writes results/raw_detections/*_yolov10n.json
and appends timing to results/metrics/baseline_latency_metrics.csv.
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths, load_experiment
from utils_coco import (
    get_class_map, get_image_root, condition_gt_paths, merge_coco_jsons,
    save_detections, append_csv_row, log_issue, CONDITION_SPECS,
)
from measure_latency import get_gpu_name, reset_gpu_memory_stats, get_peak_gpu_memory_mb, LatencyTimer

from ultralytics import YOLO

DETECTOR_NAME = "yolov10n"
WEIGHTS = "yolov10n.pt"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_condition(model, paths, exp, dataset, condition, output_name, log_path, device):
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
    image_paths = [os.path.join(image_root, img["file_name"]) for img in images]
    valid = [(iid_img, ip) for iid_img, ip in zip(images, image_paths) if os.path.isfile(ip)]
    for iid_img, ip in zip(images, image_paths):
        if not os.path.isfile(ip):
            log_issue(log_path, ip, "image file missing at inference time")

    reset_gpu_memory_stats(device)
    detections = []
    n_dropped_class = 0

    valid_paths = [ip for _, ip in valid]
    valid_imgs = [iid_img for iid_img, _ in valid]

    # Per-image predict() calls (not a single stream-over-list call): a full-list stream
    # dataloader was found to trigger a rare CUDA int32-indexing crash on some BDD100K
    # image orderings. Per-image calls are slightly slower but proven reliable here.
    import torch as _torch
    for i, (img_record, ip) in enumerate(zip(valid_imgs, valid_paths)):
        result = model.predict(
            source=ip,
            conf=exp["confidence_threshold"],
            iou=exp["nms_threshold"],
            imgsz=paths["image_size"],
            device=device,
            verbose=False,
        )[0]
        image_id = img_record["id"]
        names = result.names
        for box in result.boxes:
            cls_idx = int(box.cls.item())
            coco_name = names[cls_idx]
            dataset_name = class_map.get(coco_name)
            if dataset_name is None or dataset_name not in cat_name_to_id:
                n_dropped_class += 1
                continue
            category_id = cat_name_to_id[dataset_name]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf.item())
            detections.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })
        del result
        if (i + 1) % 20 == 0:
            _torch.cuda.empty_cache()
        if (i + 1) % 200 == 0:
            print(f"[PROGRESS] {output_name}: {i + 1}/{len(valid_paths)} images", flush=True)

    out_dir = os.path.join(REPO_ROOT, "results", "raw_detections")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{output_name}_{DETECTOR_NAME}.json")
    save_detections(out_path, detections)
    peak_mem = get_peak_gpu_memory_mb(device)

    # Single-image latency: reuse the same model (per-image predict calls already proven
    # stable), just clear the cache first so the reading isn't inflated by the bulk pass.
    _torch.cuda.empty_cache()
    timer = LatencyTimer(device)
    sample_paths = valid_paths[:min(50, len(valid_paths))]
    for ip in sample_paths[:5]:
        model.predict(source=ip, conf=exp["confidence_threshold"], iou=exp["nms_threshold"],
                             imgsz=paths["image_size"], device=device, verbose=False)
    for ip in sample_paths:
        timer.time_call(
            model.predict, source=ip, conf=exp["confidence_threshold"], iou=exp["nms_threshold"],
            imgsz=paths["image_size"], device=device, verbose=False,
        )
    _torch.cuda.empty_cache()

    latency_csv = os.path.join(REPO_ROOT, "results", "metrics", "baseline_latency_metrics.csv")
    fieldnames = ["dataset", "condition", "detector", "checkpoint", "input_size", "batch_size",
                  "num_images_timed", "latency_ms", "FPS", "GPU_mem_MB", "GPU_name", "timestamp"]
    row = {
        "dataset": dataset, "condition": condition or "none", "detector": DETECTOR_NAME,
        "checkpoint": WEIGHTS, "input_size": paths["image_size"], "batch_size": paths["batch_size"],
        "num_images_timed": len(sample_paths), "latency_ms": timer.mean_latency_ms(), "FPS": timer.fps(),
        "GPU_mem_MB": peak_mem, "GPU_name": get_gpu_name(device),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    append_csv_row(latency_csv, fieldnames, row)

    print(f"[OK] {out_path}: {len(detections)} detections over {len(valid_paths)} images "
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

    model = YOLO(WEIGHTS)

    for dataset, condition, output_name in CONDITION_SPECS:
        if only_condition and output_name != only_condition:
            continue
        out_path = os.path.join(REPO_ROOT, "results", "raw_detections", f"{output_name}_{DETECTOR_NAME}.json")
        if only_condition is None and os.path.isfile(out_path):
            print(f"[SKIP] {out_path} already exists")
            continue
        run_condition(model, paths, exp, dataset, condition, output_name, log_path, device)


if __name__ == "__main__":
    main()
