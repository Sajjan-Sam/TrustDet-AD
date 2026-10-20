#!/usr/bin/env python3
"""Evaluate detector outputs against processed COCO annotations.

Reports COCO detection metrics and precision/recall/F1 at the operating point.
Missing raw detection files are logged and skipped.
Run: python src/evaluate_detection.py"""
import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths, load_experiment
from utils_coco import (
    condition_gt_paths, merge_coco_jsons, append_csv_row, log_issue, CONDITION_SPECS,
    xywh_to_xyxy, iou_xyxy,
)
from measure_latency import get_gpu_name

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DETECTORS = ["yolov8n", "yolov10n", "rtdetr", "fasterrcnn"]
CHECKPOINTS = {
    "yolov8n": "yolov8n.pt",
    "yolov10n": "yolov10n.pt",
    "rtdetr": "rtdetr-l.pt",
    "fasterrcnn": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
}
OP_CONFIDENCE = 0.25  # fixed operating point for precision/recall/F1 (distinct from the mAP sweep)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_prf1(gt_coco, detections, iou_thresh=0.5, score_thresh=OP_CONFIDENCE):
    """Greedy highest-score-first, highest-IoU matching per image+class."""
    gt_by_img_cat = {}
    for ann in gt_coco.dataset["annotations"]:
        key = (ann["image_id"], ann["category_id"])
        gt_by_img_cat.setdefault(key, []).append({"bbox": xywh_to_xyxy(ann["bbox"]), "matched": False})

    dets = [d for d in detections if d["score"] >= score_thresh]
    dets.sort(key=lambda d: -d["score"])

    tp, fp = 0, 0
    for d in dets:
        key = (d["image_id"], d["category_id"])
        gts = gt_by_img_cat.get(key, [])
        best_iou, best_idx = 0.0, -1
        det_box = xywh_to_xyxy(d["bbox"])
        for i, g in enumerate(gts):
            if g["matched"]:
                continue
            iou = iou_xyxy(det_box, g["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thresh and best_idx >= 0:
            gts[best_idx]["matched"] = True
            tp += 1
        else:
            fp += 1

    total_gt = sum(len(v) for v in gt_by_img_cat.values())
    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_per_class_ap(coco_eval, gt_coco):
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    cat_ids = gt_coco.getCatIds()
    id_to_name = {c["id"]: c["name"] for c in gt_coco.dataset["categories"]}
    rows = []
    for k_idx, cat_id in enumerate(cat_ids):
        # AP@0.5 -> T index 0 ; AP@[.5:.95] -> mean over all T
        p50 = precision[0, :, k_idx, 0, -1]
        p50 = p50[p50 > -1]
        ap50 = float(p50.mean()) if p50.size else -1.0

        p_all = precision[:, :, k_idx, 0, -1]
        p_all = p_all[p_all > -1]
        ap_all = float(p_all.mean()) if p_all.size else -1.0

        num_gt = sum(1 for a in gt_coco.dataset["annotations"] if a["category_id"] == cat_id)
        rows.append({
            "class_name": id_to_name.get(cat_id, str(cat_id)),
            "AP50": round(ap50, 4) if ap50 >= 0 else "",
            "AP50_95": round(ap_all, 4) if ap_all >= 0 else "",
            "num_gt_instances": num_gt,
        })
    return rows


def main():
    paths = load_paths()
    exp = load_experiment()
    device = f"cuda:{paths['gpu_id']}"

    metrics_dir = os.path.join(REPO_ROOT, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    det_csv = os.path.join(metrics_dir, "baseline_detection_metrics.csv")
    per_class_csv = os.path.join(metrics_dir, "per_class_ap.csv")
    log_path = os.path.join(REPO_ROOT, "logs", "evaluate_detection_issues.log")

    # Reuse combinations already recorded in the metrics CSV on subsequent runs.
    only_combo = sys.argv[1] if len(sys.argv) > 1 else None
    done_combos = set()
    if only_combo is None and os.path.isfile(det_csv):
        import csv as _csv
        with open(det_csv, "r") as f:
            for r in _csv.DictReader(f):
                done_combos.add((r["dataset"], r["condition"], r["detector"]))
        print(f"[RESUME] {len(done_combos)} combos already in {det_csv}, skipping those")
    elif only_combo is None:
        if os.path.exists(per_class_csv):
            os.remove(per_class_csv)
        if os.path.exists(log_path):
            os.remove(log_path)

    det_fieldnames = ["dataset", "condition", "detector", "checkpoint", "input_size",
                       "confidence_threshold", "nms_threshold", "mAP@0.5", "mAP@0.5:0.95",
                       "precision", "recall", "F1", "latency_ms", "FPS", "GPU_name", "timestamp"]
    per_class_fieldnames = ["dataset", "condition", "detector", "class_name", "AP50", "AP50_95", "num_gt_instances"]

    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    raw_dir = os.path.join(REPO_ROOT, "results", "raw_detections")

    for dataset, condition, output_name in CONDITION_SPECS:
        if only_combo and output_name != only_combo:
            continue
        gt_paths = condition_gt_paths(processed_dir, dataset, condition)
        merged_gt, _ = merge_coco_jsons(gt_paths)
        if not merged_gt["images"]:
            log_issue(log_path, output_name, "no ground-truth images available")
            continue

        gt_coco = COCO()
        gt_coco.dataset = merged_gt
        gt_coco.createIndex()

        for detector in DETECTORS:
            if (dataset, condition or "none", detector) in done_combos:
                print(f"[SKIP] {dataset}/{condition or 'none'}/{detector} already in {det_csv}")
                continue
            det_path = os.path.join(raw_dir, f"{output_name}_{detector}.json")
            if not os.path.isfile(det_path):
                log_issue(log_path, det_path, "raw detections file missing, skipped (not invented)")
                continue

            with open(det_path, "r") as f:
                detections = json.load(f)

            if not detections:
                log_issue(log_path, det_path, "raw detections file is empty")
                mAP50 = mAP5095 = precision = recall = f1 = 0.0
            else:
                dt_coco = gt_coco.loadRes(det_path)
                coco_eval = COCOeval(gt_coco, dt_coco, iouType="bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                mAP5095 = round(float(coco_eval.stats[0]), 4)
                mAP50 = round(float(coco_eval.stats[1]), 4)
                precision, recall, f1 = compute_prf1(gt_coco, detections)

                for row in compute_per_class_ap(coco_eval, gt_coco):
                    row.update({"dataset": dataset, "condition": condition or "none", "detector": detector})
                    append_csv_row(per_class_csv, per_class_fieldnames, row)

            lat_row = None
            lat_csv = os.path.join(metrics_dir, "baseline_latency_metrics.csv")
            if os.path.isfile(lat_csv):
                import csv
                with open(lat_csv, "r") as f:
                    for r in csv.DictReader(f):
                        if r["dataset"] == dataset and r["condition"] == (condition or "none") and r["detector"] == detector:
                            lat_row = r

            row = {
                "dataset": dataset, "condition": condition or "none", "detector": detector,
                "checkpoint": CHECKPOINTS[detector], "input_size": paths["image_size"],
                "confidence_threshold": exp["confidence_threshold"], "nms_threshold": exp["nms_threshold"],
                "mAP@0.5": mAP50, "mAP@0.5:0.95": mAP5095,
                "precision": precision, "recall": recall, "F1": f1,
                "latency_ms": lat_row["latency_ms"] if lat_row else "",
                "FPS": lat_row["FPS"] if lat_row else "",
                "GPU_name": lat_row["GPU_name"] if lat_row else get_gpu_name(device),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            append_csv_row(det_csv, det_fieldnames, row)
            print(f"[OK] {dataset}/{condition or 'none'}/{detector}: mAP@0.5={mAP50} mAP@.5:.95={mAP5095} "
                  f"P={precision} R={recall} F1={f1}")

    print(f"[DONE] wrote {det_csv} and {per_class_csv}")


if __name__ == "__main__":
    main()
