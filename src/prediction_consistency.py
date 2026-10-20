"""Prediction consistency component: agreement of a detection under 4 fixed perturbations.

Transformations:
  1. brightness jitter  -- no geometric change, boxes compared directly
  2. Gaussian blur       -- no geometric change, boxes compared directly
  3. resize jitter       -- scale image, remap returned boxes back by 1/scale
  4. horizontal flip     -- mirror image, remap returned boxes by width - x

For a given original detection, consistency = (# of the 4 transforms whose output contains a
same-class detection with IoU>=0.5 against the original box) / 4.
"""
import cv2

from utils_coco import xywh_to_xyxy, iou_xyxy

TRANSFORM_NAMES = ["brightness", "blur", "resize", "hflip"]
RESIZE_SCALE = 0.85


def apply_transform(img, name):
    """Returns (transformed_image, remap_param). remap_param is transform-specific metadata
    needed to map boxes detected on the transformed image back to original-image coordinates."""
    if name == "brightness":
        return cv2.convertScaleAbs(img, alpha=1.3, beta=25), None
    if name == "blur":
        return cv2.GaussianBlur(img, (7, 7), 0), None
    if name == "resize":
        h, w = img.shape[:2]
        new_w, new_h = max(1, int(w * RESIZE_SCALE)), max(1, int(h * RESIZE_SCALE))
        return cv2.resize(img, (new_w, new_h)), RESIZE_SCALE
    if name == "hflip":
        return cv2.flip(img, 1), img.shape[1]
    raise ValueError(f"unknown transform: {name}")


def remap_box_to_original(bbox_xywh, name, param):
    """bbox_xywh: [x,y,w,h] in the transformed image's coordinate space -> original coords."""
    x, y, w, h = bbox_xywh
    if name == "resize":
        scale = param
        return [x / scale, y / scale, w / scale, h / scale]
    if name == "hflip":
        orig_w = param
        x1, x2 = x, x + w
        new_x1, new_x2 = orig_w - x2, orig_w - x1
        return [new_x1, y, new_x2 - new_x1, h]
    return [x, y, w, h]  # brightness / blur: no geometric change


def best_match(det, candidate_dets):
    """Highest-IoU same-class candidate. Returns (iou, matched_det) or (0.0, None)."""
    det_xyxy = xywh_to_xyxy(det["bbox"])
    best_iou, best = 0.0, None
    for c in candidate_dets:
        if c["category_id"] != det["category_id"]:
            continue
        iou = iou_xyxy(det_xyxy, xywh_to_xyxy(c["bbox"]))
        if iou > best_iou:
            best_iou, best = iou, c
    return best_iou, best


def consistency_and_matches(det, remapped_dets_per_transform):
    """remapped_dets_per_transform: {transform_name: [detections already remapped to original coords]}.
    Returns (consistency_fraction, list of (transform_name, matched_det, iou) for matches>=0.5)."""
    matches = []
    for name in TRANSFORM_NAMES:
        cand = remapped_dets_per_transform.get(name, [])
        iou, matched = best_match(det, cand)
        if iou >= 0.5 and matched is not None:
            matches.append((name, matched, iou))
    consistency = len(matches) / len(TRANSFORM_NAMES)
    return consistency, matches
