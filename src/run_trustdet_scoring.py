#!/usr/bin/env python3
"""Compute detection reliability and fit the condition calibrator.

Scores detections with confidence >= 0.25. For each dataset/condition/detector,
a seed-42 sample of at most 150 images is used across validation and test
combined. Features use four transformed inference passes and EigenCAM saliency.
Calibration bins and logistic regression are fitted on validation rows only.

Writes per-combination checkpoints, validation/test score CSVs, and a fitted
calibrator under results/reliability/. Remove cached outputs before rerunning
with different data, models, settings or dependencies.
Run: python src/run_trustdet_scoring.py"""
import os
import sys
import csv
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths, load_experiment
from utils_coco import CONDITION_SPECS, condition_gt_paths, merge_coco_jsons, get_image_root, log_issue
from reliability_components import (
    load_val_calibration_bins, calibration_error_for, confidence_only_score, fixed_trustdet_score,
)
from prediction_consistency import apply_transform, remap_box_to_original, consistency_and_matches, TRANSFORM_NAMES
from uncertainty_score import uncertainty_for_detection
from xai_stability import EigenCAMHook, get_target_layer, crop_and_resize, saliency_similarity
from condition_calibrated_score import ConditionCalibrator, FEATURE_COLS

import cv2
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTORS = ["yolov8n", "yolov10n", "rtdetr", "fasterrcnn"]
CONF_THRESHOLD_FOR_SCORING = 0.25
MAX_IMAGES_PER_COMBO = 150
SAMPLE_SEED = 42

OUTPUT_FIELDNAMES = ["image_id", "dataset", "condition", "detector", "split", "bbox", "class_id",
                      "class_name", "confidence", "correctness_label", "is_correct",
                      "calibration_error", "prediction_consistency", "uncertainty_score",
                      "xai_stability", "runtime_overhead", "confidence_only_score",
                      "fixed_trustdet_score"]


def load_detector(name, exp, device):
    if name in ("yolov8n", "yolov10n", "rtdetr"):
        from ultralytics import YOLO, RTDETR
        weights = {"yolov8n": "yolov8n.pt", "yolov10n": "yolov10n.pt", "rtdetr": "rtdetr-l.pt"}[name]
        cls = RTDETR if name == "rtdetr" else YOLO
        model = cls(weights)
        return {"type": "ultralytics", "model": model}
    if name == "fasterrcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=exp["confidence_threshold"])
        model.roi_heads.nms_thresh = exp["nms_threshold"]
        old_t = model.transform
        model.transform = GeneralizedRCNNTransform(min_size=640, max_size=640,
                                                     image_mean=old_t.image_mean, image_std=old_t.image_std)
        model.to(device)
        model.eval()
        return {"type": "torchvision", "model": model, "categories": weights.meta["categories"]}
    raise ValueError(name)


def run_ultralytics(model, img, exp, device, image_size):
    result = model.predict(source=img, conf=exp["confidence_threshold"], iou=exp["nms_threshold"],
                            imgsz=image_size, device=device, verbose=False)[0]
    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append({"coco_name": result.names[int(box.cls.item())],
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(box.conf.item())})
    return dets


@torch.no_grad()
def run_torchvision(model, categories, img, device):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device) / 255.0
    out = model([tensor])[0]
    dets = []
    for label, box, score in zip(out["labels"].tolist(), out["boxes"].tolist(), out["scores"].tolist()):
        x1, y1, x2, y2 = box
        dets.append({"category_id": label, "coco_name": categories[label],
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return dets


def main():
    paths = load_paths()
    exp = load_experiment()
    device = f"cuda:{paths['gpu_id']}"

    from utils_coco import get_class_map
    reliability_dir = os.path.join(REPO_ROOT, "results", "reliability")
    os.makedirs(reliability_dir, exist_ok=True)
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    labels_path = os.path.join(reliability_dir, "detection_level_labels.csv")
    log_path = os.path.join(REPO_ROOT, "logs", "trustdet_scoring_issues.log")

    bins_path = os.path.join(reliability_dir, "calibration_bins.csv")
    val_bins = load_val_calibration_bins(bins_path)

    combo_out_dir = os.path.join(reliability_dir, "_combo_scores")
    os.makedirs(combo_out_dir, exist_ok=True)

    for dataset, condition, output_name in CONDITION_SPECS:
        gt_paths = condition_gt_paths(processed_dir, dataset, condition)
        merged_gt, _ = merge_coco_jsons(gt_paths)
        image_by_id = {img["id"]: img for img in merged_gt["images"]}
        image_root = get_image_root(paths, dataset)
        class_map = get_class_map(dataset)
        cat_id_to_name = {c["id"]: c["name"] for c in merged_gt["categories"]}

        for detector in DETECTORS:
            combo_key = f"{output_name}_{detector}"
            combo_path = os.path.join(combo_out_dir, f"{combo_key}.csv")
            if os.path.isfile(combo_path):
                print(f"[SKIP] {combo_key} already scored")
                continue

            # Gather this combo's detections at the scoring confidence threshold, grouped by image.
            by_image = {}
            with open(labels_path, "r") as f:
                for r in csv.DictReader(f):
                    if r["dataset"] != dataset or r["condition"] != (condition or "none") or r["detector"] != detector:
                        continue
                    if float(r["confidence"]) < CONF_THRESHOLD_FOR_SCORING:
                        continue
                    by_image.setdefault(int(r["image_id"]), []).append(r)

            if not by_image:
                print(f"[WARN] {combo_key}: no detections at confidence>={CONF_THRESHOLD_FOR_SCORING}")
                open(combo_path, "w").close()
                continue

            if len(by_image) > MAX_IMAGES_PER_COMBO:
                total_available = len(by_image)
                rng = random.Random(SAMPLE_SEED)
                keep_ids = set(rng.sample(list(by_image.keys()), MAX_IMAGES_PER_COMBO))
                by_image = {k: v for k, v in by_image.items() if k in keep_ids}
                print(f"[SAMPLE] {combo_key}: sampled {len(by_image)}/{total_available} images (seed={SAMPLE_SEED})")

            detector_bundle = load_detector(detector, exp, device)
            target_layer = get_target_layer(detector_bundle["model"], detector)
            hook = EigenCAMHook(target_layer)
            bins_for_combo = val_bins.get((dataset, condition or "none", detector))

            out_rows = []
            for image_id, det_rows in by_image.items():
                img_meta = image_by_id.get(image_id)
                if img_meta is None:
                    log_issue(log_path, f"{combo_key}:{image_id}", "image_id not found in GT (unexpected)")
                    continue
                img_path = os.path.join(image_root, img_meta["file_name"])
                img = cv2.imread(img_path)
                if img is None:
                    log_issue(log_path, img_path, "unreadable image for reliability scoring")
                    for r in det_rows:
                        out_rows.append(_zero_row(r, cat_id_to_name, dataset, condition, detector, reason="unreadable_image"))
                    continue

                t0 = time.perf_counter()

                # Original-image saliency.
                if detector_bundle["type"] == "ultralytics":
                    run_ultralytics(detector_bundle["model"], img, exp, device, paths["image_size"])
                else:
                    run_torchvision(detector_bundle["model"], detector_bundle["categories"], img, device)
                orig_saliency = hook.saliency_map()

                # 4 transforms: detections (remapped to original coords) + saliency maps.
                remapped_dets = {}
                transform_saliency = {}
                for name in TRANSFORM_NAMES:
                    t_img, param = apply_transform(img, name)
                    if detector_bundle["type"] == "ultralytics":
                        raw = run_ultralytics(detector_bundle["model"], t_img, exp, device, paths["image_size"])
                    else:
                        raw = run_torchvision(detector_bundle["model"], detector_bundle["categories"], t_img, device)
                    remapped = []
                    for d in raw:
                        dataset_name = class_map.get(d["coco_name"])
                        if dataset_name is None:
                            continue
                        cat_id = next((cid for cid, cn in cat_id_to_name.items() if cn == dataset_name), None)
                        if cat_id is None:
                            continue
                        remapped.append({
                            "category_id": cat_id,
                            "bbox": remap_box_to_original(d["bbox"], name, param),
                            "score": d["score"],
                        })
                    remapped_dets[name] = remapped
                    transform_saliency[name] = hook.saliency_map()

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                img_shape = img.shape[:2]

                claimed_matches = {name: set() for name in TRANSFORM_NAMES}
                det_rows_sorted = sorted(det_rows, key=lambda r: -float(r["confidence"]))

                for r in det_rows_sorted:
                    det = {
                        "category_id": int(r["category_id"]),
                        "bbox": [float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])],
                        "score": float(r["confidence"]),
                    }
                    consistency, matches = consistency_and_matches(det, remapped_dets)
                    uncertainty = uncertainty_for_detection(det["score"], matches)
                    calib_err = calibration_error_for(det["score"], bins_for_combo)

                    from utils_coco import xywh_to_xyxy
                    sims = []
                    if orig_saliency is not None:
                        orig_crop = crop_and_resize(orig_saliency, xywh_to_xyxy(det["bbox"]), img_shape)
                        for name, matched_det, iou in matches:
                            matched_key = id(matched_det)
                            if matched_key in claimed_matches[name]:
                                continue  # duplicate matching conflict -- skip this transform for this detection
                            claimed_matches[name].add(matched_key)
                            t_sal = transform_saliency.get(name)
                            if t_sal is None:
                                continue
                            t_crop = crop_and_resize(t_sal, xywh_to_xyxy(matched_det["bbox"]), img_shape)
                            sim = saliency_similarity(orig_crop, t_crop)
                            if sim is not None:
                                sims.append(sim)
                    if sims:
                        xai_stab = sum(sims) / len(sims)
                    else:
                        xai_stab = 0.0
                        log_issue(log_path, f"{combo_key}:{image_id}",
                                   "xai_stability failed (no valid transform saliency) -> 0")

                    conf_only = confidence_only_score(det["score"])
                    fixed_score = fixed_trustdet_score(det["score"], calib_err, consistency, uncertainty, xai_stab)

                    out_rows.append({
                        "image_id": image_id, "dataset": dataset, "condition": condition or "none",
                        "detector": detector, "split": r["split"],
                        "bbox": f"{det['bbox'][0]:.2f},{det['bbox'][1]:.2f},{det['bbox'][2]:.2f},{det['bbox'][3]:.2f}",
                        "class_id": det["category_id"], "class_name": r["class_name"],
                        "confidence": det["score"], "correctness_label": r["correctness_label"],
                        "is_correct": r["is_correct"], "calibration_error": round(calib_err, 4),
                        "prediction_consistency": round(consistency, 4),
                        "uncertainty_score": round(uncertainty, 4), "xai_stability": round(xai_stab, 4),
                        "runtime_overhead": round(elapsed_ms, 3),
                        "confidence_only_score": round(conf_only, 4),
                        "fixed_trustdet_score": round(fixed_score, 4),
                        "condition_id": f"{dataset}_{condition or 'none'}",
                    })

            hook.remove()
            del detector_bundle
            torch.cuda.empty_cache()

            with open(combo_path, "w", newline="") as f:
                fieldnames = OUTPUT_FIELDNAMES + ["condition_id"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(out_rows)
            print(f"[OK] {combo_key}: {len(out_rows)} detections scored")

    # --- Merge all combo files, fit condition-calibrated logistic regression on val, score both splits ---
    all_rows = []
    for fname in sorted(os.listdir(combo_out_dir)):
        fpath = os.path.join(combo_out_dir, fname)
        if os.path.getsize(fpath) == 0:
            continue
        with open(fpath, "r") as f:
            all_rows.extend(list(csv.DictReader(f)))

    if not all_rows:
        print("[ERROR] no scored detections at all -- nothing to fit/write")
        sys.exit(1)

    val_rows = [r for r in all_rows if r["split"] == "val"]
    test_rows = [r for r in all_rows if r["split"] == "test"]

    calibrator = ConditionCalibrator()
    calibrator.fit(val_rows)
    val_scores = calibrator.score(val_rows)
    test_scores = calibrator.score(test_rows) if test_rows else []

    for r, s in zip(val_rows, val_scores):
        r["condition_calibrated_score"] = round(float(s), 4)
    for r, s in zip(test_rows, test_scores):
        r["condition_calibrated_score"] = round(float(s), 4)

    calibrator.save(os.path.join(reliability_dir, "trustdet_calibrator.pkl"))

    final_fieldnames = OUTPUT_FIELDNAMES + ["condition_calibrated_score"]
    for split_name, rows in [("val", val_rows), ("test", test_rows)]:
        out_path = os.path.join(reliability_dir, f"reliability_scores_{split_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[DONE] wrote {out_path} ({len(rows)} rows)")


def _zero_row(r, cat_id_to_name, dataset, condition, detector, reason):
    return {
        "image_id": r["image_id"], "dataset": dataset, "condition": condition or "none",
        "detector": detector, "split": r["split"],
        "bbox": f"{r['bbox_x']},{r['bbox_y']},{r['bbox_w']},{r['bbox_h']}",
        "class_id": r["category_id"], "class_name": r["class_name"], "confidence": r["confidence"],
        "correctness_label": r["correctness_label"], "is_correct": r["is_correct"],
        "calibration_error": 1.0, "prediction_consistency": 0.0, "uncertainty_score": 1.0,
        "xai_stability": 0.0, "runtime_overhead": 0.0,
        "confidence_only_score": float(r["confidence"]), "fixed_trustdet_score": 0.0,
        "condition_id": f"{dataset}_{condition or 'none'}",
    }


if __name__ == "__main__":
    main()
