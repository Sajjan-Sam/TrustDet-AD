"""Shared COCO-style JSON builder used by all dataset preparation scripts."""
import os
import json

# COCO (80-class) pretrained-model output names -> dataset ground-truth class names.
# Detectors are COCO-pretrained; datasets use their own taxonomies. Only classes with a
# direct COCO equivalent are mapped; unmatched categories are logged and dropped.
EXDARK_COCO_MAP = {
    "bicycle": "Bicycle",
    "boat": "Boat",
    "bottle": "Bottle",
    "bus": "Bus",
    "car": "Car",
    "cat": "Cat",
    "chair": "Chair",
    "cup": "Cup",
    "dog": "Dog",
    "motorcycle": "Motorbike",
    "person": "People",
    "dining table": "Table",
}

# BDD100K has no COCO equivalent for "rider" or "traffic sign" -- COCO-pretrained detectors
# structurally cannot predict these classes. This is a known, documented limitation, not a bug.
BDD_COCO_MAP = {
    "bicycle": "bike",
    "bus": "bus",
    "car": "car",
    "motorcycle": "motor",
    "person": "person",
    "truck": "truck",
    "train": "train",
    "traffic light": "traffic light",
}


def get_class_map(dataset):
    if dataset == "exdark":
        return EXDARK_COCO_MAP
    if dataset == "bdd100k":
        return BDD_COCO_MAP
    raise ValueError(f"unknown dataset: {dataset}")


def get_image_root(paths_cfg, dataset):
    if dataset == "exdark":
        return os.path.join(paths_cfg["exdark_root"], "images")
    if dataset == "bdd100k":
        return os.path.join(paths_cfg["bdd100k_root"], "images")
    raise ValueError(f"unknown dataset: {dataset}")


def condition_gt_paths(processed_dir, dataset, condition):
    """Returns the val+test processed COCO json paths for a dataset/condition (train excluded)."""
    if dataset == "exdark":
        return [
            os.path.join(processed_dir, "exdark_val.json"),
            os.path.join(processed_dir, "exdark_test.json"),
        ]
    if dataset == "bdd100k":
        return [
            os.path.join(processed_dir, f"bdd100k_{condition}_val.json"),
            os.path.join(processed_dir, f"bdd100k_{condition}_test.json"),
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def merge_coco_jsons(paths):
    """Merge multiple COCO json files sharing the same category space into one in-memory dict.
    Re-indexes image/annotation ids to avoid collisions. Also returns, per merged image id,
    which source file it came from (for later split-aware evaluation)."""
    merged_images = []
    merged_annotations = []
    categories = None
    image_id_offset = 0
    ann_id_offset = 0
    source_of_image = {}

    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]

        old_to_new_img_id = {}
        for img in data["images"]:
            new_id = img["id"] + image_id_offset
            old_to_new_img_id[img["id"]] = new_id
            new_img = dict(img)
            new_img["id"] = new_id
            merged_images.append(new_img)
            source_of_image[new_id] = os.path.basename(path)

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = ann["id"] + ann_id_offset
            new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
            merged_annotations.append(new_ann)

        image_id_offset = max((img["id"] for img in merged_images), default=0) + 1
        ann_id_offset = max((a["id"] for a in merged_annotations), default=0) + 1

    return {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories or [],
    }, source_of_image


class COCOBuilder:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self._cat_ids = {}
        self._img_id = 0
        self._ann_id = 0

    def add_category(self, name):
        if name in self._cat_ids:
            return self._cat_ids[name]
        cat_id = len(self._cat_ids) + 1
        self._cat_ids[name] = cat_id
        self.categories.append({"id": cat_id, "name": name})
        return cat_id

    def add_image(self, file_name, width, height, extra=None):
        self._img_id += 1
        img_id = self._img_id
        record = {"id": img_id, "file_name": file_name, "width": width, "height": height}
        if extra:
            record.update(extra)
        self.images.append(record)
        return img_id

    def add_annotation(self, image_id, category_id, bbox):
        # bbox = [x, y, w, h] absolute pixels
        self._ann_id += 1
        ann_id = self._ann_id
        x, y, w, h = bbox
        self.annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        return ann_id

    def to_dict(self):
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


def log_issue(log_path, file_path, reason):
    with open(log_path, "a") as f:
        f.write(f"{file_path}\t{reason}\n")


# (dataset, condition, output_name) -- output_name is used in raw_detections/<output_name>_<detector>.json
CONDITION_SPECS = [
    ("exdark", None, "exdark"),
    ("bdd100k", "daytime", "bdd100k_daytime"),
    ("bdd100k", "night", "bdd100k_night"),
    ("bdd100k", "rain", "bdd100k_rain"),
    ("bdd100k", "fog", "bdd100k_fog"),
]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def save_detections(path, detections):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(detections, f)
    os.replace(tmp_path, path)  # atomic: a kill mid-write can't leave a corrupt/partial file


def append_csv_row(path, fieldnames, row):
    import csv
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
