"""Load shared dataset paths and experiment settings from YAML."""
import sys
import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_paths():
    path = os.path.join(CONFIG_DIR, "paths.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required = [
        "exdark_root", "bdd100k_root", "output_root",
        "gpu_id", "image_size", "batch_size", "num_workers",
    ]
    for key in required:
        if key not in cfg:
            print(f"[CONFIG ERROR] Missing required field '{key}' in configs/paths.yaml")
            sys.exit(1)

    for key in ["exdark_root", "bdd100k_root", "output_root"]:
        val = cfg[key]
        if not val:
            print(f"[CONFIG ERROR] '{key}' is empty in configs/paths.yaml. Set it before running.")
            sys.exit(1)
        if key != "output_root" and not os.path.isdir(val):
            print(f"[CONFIG ERROR] '{key}' path does not exist: {val}")
            sys.exit(1)

    return cfg


def load_experiment():
    path = os.path.join(CONFIG_DIR, "experiment.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required = [
        "detectors", "datasets", "iou_threshold",
        "confidence_threshold", "nms_threshold", "random_seed",
    ]
    for key in required:
        if key not in cfg:
            print(f"[CONFIG ERROR] Missing required field '{key}' in configs/experiment.yaml")
            sys.exit(1)

    return cfg
