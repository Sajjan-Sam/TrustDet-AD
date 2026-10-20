#!/usr/bin/env python3
"""Verify ExDark and BDD100K are placed correctly per configs/paths.yaml.
Usage: python src/check_dataset_paths.py
"""
import os
import sys
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "paths.yaml")


def fail(msg):
    print(f"[MISSING] {msg}")


def main():
    if not os.path.isfile(CONFIG_PATH):
        print(f"[CONFIG ERROR] configs/paths.yaml not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    exdark_root = cfg.get("exdark_root", "")
    bdd_root = cfg.get("bdd100k_root", "")

    ok = True

    if not exdark_root:
        fail("exdark_root is empty in configs/paths.yaml")
        ok = False
    elif not os.path.isdir(exdark_root):
        fail(f"exdark_root does not exist: {exdark_root}")
        ok = False
    else:
        for sub in ["images", "annotations"]:
            p = os.path.join(exdark_root, sub)
            if not os.path.isdir(p):
                fail(f"ExDark missing folder: {p}")
                ok = False
            elif not os.listdir(p):
                fail(f"ExDark folder is empty: {p}")
                ok = False

    if not bdd_root:
        fail("bdd100k_root is empty in configs/paths.yaml")
        ok = False
    elif not os.path.isdir(bdd_root):
        fail(f"bdd100k_root does not exist: {bdd_root}")
        ok = False
    else:
        for sub in ["images/100k/train", "images/100k/val"]:
            p = os.path.join(bdd_root, sub)
            if not os.path.isdir(p):
                fail(f"BDD100K missing folder: {p}")
                ok = False
            elif not os.listdir(p):
                fail(f"BDD100K folder is empty: {p}")
                ok = False

        for label_file in ["labels/det_20/det_train.json", "labels/det_20/det_val.json"]:
            p = os.path.join(bdd_root, label_file)
            if not os.path.isfile(p):
                fail(f"BDD100K missing label file: {p}")
                ok = False

    if not ok:
        print("\nDataset paths verification FAILED. Fix the missing paths above.")
        sys.exit(1)

    print("Dataset paths verified successfully")


if __name__ == "__main__":
    main()
