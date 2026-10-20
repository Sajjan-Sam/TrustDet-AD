#!/usr/bin/env python3
"""Compute detection-level ECE and confidence-bin statistics.

Reads results/reliability/detection_level_labels.csv and writes calibration_bins.csv
and detection_ece.csv in the same directory. Validation and test diagnostics are
computed separately. Threshold selection is handled by tune_thresholds.py.
Run: python src/calibration_metrics.py"""
import os
import sys
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BINS = 10


def bin_index(confidence):
    idx = int(confidence * N_BINS)
    return min(idx, N_BINS - 1)


def compute_bins(records):
    """records: list of (confidence, is_correct). Returns list of N_BINS dicts."""
    bins = [{"count": 0, "sum_conf": 0.0, "sum_correct": 0} for _ in range(N_BINS)]
    for conf, is_correct in records:
        b = bin_index(conf)
        bins[b]["count"] += 1
        bins[b]["sum_conf"] += conf
        bins[b]["sum_correct"] += int(is_correct)
    return bins


def ece_from_bins(bins, total):
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        avg_conf = b["sum_conf"] / b["count"]
        accuracy = b["sum_correct"] / b["count"]
        ece += (b["count"] / total) * abs(accuracy - avg_conf)
    return round(ece, 6)


def main():
    load_paths()

    labels_path = os.path.join(REPO_ROOT, "results", "reliability", "detection_level_labels.csv")
    if not os.path.isfile(labels_path):
        print(f"[ERROR] {labels_path} not found. Run src/create_detection_level_dataset.py first.")
        sys.exit(1)

    out_dir = os.path.join(REPO_ROOT, "results", "reliability")
    bins_csv = os.path.join(out_dir, "calibration_bins.csv")
    ece_csv = os.path.join(out_dir, "detection_ece.csv")

    # group[(dataset,condition,detector,split)] -> list of (confidence,is_correct,class_name)
    groups = defaultdict(list)
    with open(labels_path, "r") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["condition"], r["detector"], r["split"])
            groups[key].append((float(r["confidence"]), r["is_correct"] == "True", r["class_name"]))

    bins_fieldnames = ["dataset", "condition", "detector", "split", "bin_index", "bin_low", "bin_high",
                        "count", "avg_confidence", "accuracy"]
    ece_fieldnames = ["dataset", "condition", "detector", "split", "class_name", "ece", "num_detections"]

    with open(bins_csv, "w", newline="") as bf, open(ece_csv, "w", newline="") as ef:
        bin_writer = csv.DictWriter(bf, fieldnames=bins_fieldnames)
        bin_writer.writeheader()
        ece_writer = csv.DictWriter(ef, fieldnames=ece_fieldnames)
        ece_writer.writeheader()

        for (dataset, condition, detector, split), records in sorted(groups.items()):
            # Overall (class_name="ALL")
            overall_bins = compute_bins([(c, ic) for c, ic, _ in records])
            overall_ece = ece_from_bins(overall_bins, len(records))
            for i, b in enumerate(overall_bins):
                bin_writer.writerow({
                    "dataset": dataset, "condition": condition, "detector": detector, "split": split,
                    "bin_index": i, "bin_low": round(i / N_BINS, 2), "bin_high": round((i + 1) / N_BINS, 2),
                    "count": b["count"],
                    "avg_confidence": round(b["sum_conf"] / b["count"], 4) if b["count"] else "",
                    "accuracy": round(b["sum_correct"] / b["count"], 4) if b["count"] else "",
                })
            ece_writer.writerow({
                "dataset": dataset, "condition": condition, "detector": detector, "split": split,
                "class_name": "ALL", "ece": overall_ece, "num_detections": len(records),
            })

            # Class-wise ECE (bins not written per-class to keep calibration_bins.csv a manageable
            # size -- per-class bin detail is reconstructable from detection_level_labels.csv directly).
            by_class = defaultdict(list)
            for c, ic, cls in records:
                by_class[cls].append((c, ic))
            for cls, cls_records in sorted(by_class.items()):
                cls_bins = compute_bins(cls_records)
                cls_ece = ece_from_bins(cls_bins, len(cls_records))
                ece_writer.writerow({
                    "dataset": dataset, "condition": condition, "detector": detector, "split": split,
                    "class_name": cls, "ece": cls_ece, "num_detections": len(cls_records),
                })

            print(f"[OK] {dataset}/{condition}/{detector}/{split}: ECE={overall_ece} "
                  f"(n={len(records)})")

    print(f"[DONE] wrote {bins_csv} and {ece_csv}")


if __name__ == "__main__":
    main()
