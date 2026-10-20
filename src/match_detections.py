"""Label detection correctness against ground-truth boxes.

Detections are processed in descending confidence order by match_condition().
For each detection, the highest-IoU ground-truth box across all classes determines
the label: correct, incorrect_class, incorrect_iou, background_false_positive or
duplicate. Correct matches require class agreement, IoU >= 0.5 and an unclaimed
box. Subsequent matches to that box are duplicates. Every detection gets a row.

An overlap greater than zero but below 0.5 is incorrect_iou; zero overlap is a
background false positive. See docs/REPRODUCING.md for evaluation conventions."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_coco import xywh_to_xyxy, iou_xyxy, condition_gt_paths, merge_coco_jsons

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label_image_detections(gt_boxes, detections):
    """gt_boxes: list of dicts {id, category_id, bbox(xywh)}.
    detections: list of dicts {category_id, bbox(xywh), score} sorted by score desc (caller's order kept).
    Returns list of dicts (same order as input detections) with match info added.
    """
    gts = [{"id": g["id"], "category_id": g["category_id"], "xyxy": xywh_to_xyxy(g["bbox"]), "matched": False}
           for g in gt_boxes]

    results = []
    for det in detections:
        det_xyxy = xywh_to_xyxy(det["bbox"])
        best_iou, best_gt = 0.0, None
        for g in gts:
            iou = iou_xyxy(det_xyxy, g["xyxy"])
            if iou > best_iou:
                best_iou, best_gt = iou, g

        if best_gt is None or best_iou == 0.0:
            label = "background_false_positive"
            matched_gt_id = None
        elif best_iou < 0.5:
            label = "incorrect_iou"
            matched_gt_id = None
        elif best_gt["category_id"] != det["category_id"]:
            label = "incorrect_class"
            matched_gt_id = None
        elif not best_gt["matched"]:
            label = "correct"
            best_gt["matched"] = True
            matched_gt_id = best_gt["id"]
        else:
            label = "duplicate"
            matched_gt_id = best_gt["id"]

        results.append({
            **det,
            "correctness_label": label,
            "is_correct": label == "correct",
            "matched_gt_id": matched_gt_id,
            "iou": round(best_iou, 4),
        })
    return results


def match_condition(dataset, condition, detector, output_name):
    """Returns list of per-detection label rows for one (dataset, condition, detector) combo,
    with a 'split' field (val/test) derived from which processed GT file each image came from."""
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    raw_dir = os.path.join(REPO_ROOT, "results", "raw_detections")

    gt_paths = condition_gt_paths(processed_dir, dataset, condition)
    merged_gt, source_of_image = merge_coco_jsons(gt_paths)
    if not merged_gt["images"]:
        return []

    det_path = os.path.join(raw_dir, f"{output_name}_{detector}.json")
    if not os.path.isfile(det_path):
        return None  # Missing raw detections are logged by the caller.

    with open(det_path, "r") as f:
        detections = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in merged_gt["categories"]}

    gt_by_image = {}
    for ann in merged_gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    det_by_image = {}
    for d in detections:
        det_by_image.setdefault(d["image_id"], []).append(d)

    def split_of(image_id):
        src = source_of_image.get(image_id, "")
        return "val" if "_val" in src else ("test" if "_test" in src else "unknown")

    rows = []
    for img in merged_gt["images"]:
        image_id = img["id"]
        gts = gt_by_image.get(image_id, [])
        dets = sorted(det_by_image.get(image_id, []), key=lambda d: -d["score"])
        labeled = label_image_detections(gts, dets)
        split = split_of(image_id)
        for d in labeled:
            rows.append({
                "image_id": image_id,
                "dataset": dataset,
                "condition": condition or "none",
                "detector": detector,
                "split": split,
                "category_id": d["category_id"],
                "class_name": cat_id_to_name.get(d["category_id"], str(d["category_id"])),
                "bbox_x": d["bbox"][0], "bbox_y": d["bbox"][1],
                "bbox_w": d["bbox"][2], "bbox_h": d["bbox"][3],
                "confidence": d["score"],
                "correctness_label": d["correctness_label"],
                "is_correct": d["is_correct"],
                "matched_gt_id": d["matched_gt_id"],
                "iou": d["iou"],
            })
    return rows
