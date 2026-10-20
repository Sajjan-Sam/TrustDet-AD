"""Reliability features and fixed-score baseline.

Calibration error is the absolute confidence/accuracy gap in the corresponding
validation confidence bin. Those bins are applied to both validation and test.
Prediction consistency, uncertainty and EigenCAM stability are defined in their
respective modules. Runtime overhead is measured per image in milliseconds.

The fixed baseline averages confidence, inverse calibration error, consistency,
inverse uncertainty and saliency stability. The condition-calibrated logistic
regression is defined in condition_calibrated_score.py."""
import csv
import os
from collections import defaultdict

N_BINS = 10


def load_val_calibration_bins(bins_csv_path):
    """Returns {(dataset,condition,detector): [bin dicts sorted by bin_index]} from the VAL rows
    of calibration_bins.csv."""
    table = defaultdict(lambda: [None] * N_BINS)
    with open(bins_csv_path, "r") as f:
        for r in csv.DictReader(f):
            if r["split"] != "val":
                continue
            key = (r["dataset"], r["condition"], r["detector"])
            idx = int(r["bin_index"])
            table[key][idx] = {
                "count": int(r["count"]),
                "avg_confidence": float(r["avg_confidence"]) if r["avg_confidence"] != "" else None,
                "accuracy": float(r["accuracy"]) if r["accuracy"] != "" else None,
            }
    return table


def calibration_error_for(confidence, bins_for_combo):
    """bins_for_combo: list of N_BINS bin dicts (or None) for one (dataset,condition,detector)."""
    idx = min(int(confidence * N_BINS), N_BINS - 1)
    b = bins_for_combo[idx] if bins_for_combo else None
    if b is None or b["count"] == 0 or b["accuracy"] is None:
        # No val evidence for this confidence range -- cannot claim a calibration error value.
        # Use the maximum possible miscalibration (1.0) as a conservative fallback, since an
        # empty bin means we have no basis to claim this confidence level is well-calibrated.
        return 1.0
    return abs(b["accuracy"] - b["avg_confidence"])


def confidence_only_score(normalized_confidence):
    return normalized_confidence


def fixed_trustdet_score(normalized_confidence, calibration_error, prediction_consistency,
                          uncertainty_score, xai_stability):
    goodness = [
        normalized_confidence,
        1.0 - calibration_error,
        prediction_consistency,
        1.0 - uncertainty_score,
        xai_stability,
    ]
    return sum(goodness) / len(goodness)
