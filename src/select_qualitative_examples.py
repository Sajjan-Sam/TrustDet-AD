#!/usr/bin/env python3
"""Select real detections for the trusted / uncertain / rejected qualitative figures.
Usage: python src/select_qualitative_examples.py
Reads results/reliability/reliability_scores_test.csv, results/ablations/gating_thresholds.csv
(condition_calibrated_trustdet method, frozen from val). Writes
figures/qualitative/selected_examples.csv (image_id, dataset, condition, detector, bbox,
class_name, confidence, score, gate_label, correctness_label). Samples six detections
per gate bucket using seed 42.
"""
import os
import sys
import csv
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths
from risk_aware_gate import gate_label

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER_BUCKET = 6
SEED = 42


def main():
    load_paths()
    scores_path = os.path.join(REPO_ROOT, "results", "reliability", "reliability_scores_test.csv")
    thresholds_path = os.path.join(REPO_ROOT, "results", "ablations", "gating_thresholds.csv")
    for p in (scores_path, thresholds_path):
        if not os.path.isfile(p):
            print(f"[ERROR] required file missing: {p}")
            sys.exit(1)

    with open(thresholds_path, "r") as f:
        thresholds = {r["method"]: r for r in csv.DictReader(f)}
    t = thresholds["condition_calibrated_trustdet"]
    tau_trusted, tau_rejected = float(t["tau_trusted"]), float(t["tau_rejected"])

    with open(scores_path, "r") as f:
        rows = list(csv.DictReader(f))

    buckets = {"trusted": [], "uncertain": [], "rejected": []}
    for r in rows:
        score = float(r["condition_calibrated_score"])
        label = gate_label(score, tau_trusted, tau_rejected)
        r["gate_label"] = label
        r["condition_calibrated_score_value"] = score
        buckets[label].append(r)

    rng = random.Random(SEED)
    out_dir = os.path.join(REPO_ROOT, "figures", "qualitative")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "selected_examples.csv")

    fieldnames = ["image_id", "dataset", "condition", "detector", "bbox", "class_id", "class_name",
                  "confidence", "condition_calibrated_score", "gate_label", "correctness_label", "is_correct"]
    selected = []
    for label, pool in buckets.items():
        if not pool:
            print(f"[WARN] no detections in bucket '{label}'")
            continue
        k = min(N_PER_BUCKET, len(pool))
        selected.extend(rng.sample(pool, k))
        print(f"[OK] bucket '{label}': {len(pool)} candidates, sampled {k}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in selected:
            r["condition_calibrated_score"] = r["condition_calibrated_score_value"]
            writer.writerow(r)

    print(f"[DONE] wrote {out_path} ({len(selected)} examples)")


if __name__ == "__main__":
    main()
