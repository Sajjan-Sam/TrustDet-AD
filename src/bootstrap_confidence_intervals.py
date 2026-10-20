#!/usr/bin/env python3
"""Bootstrap detection-level confidence intervals on held-out test scores.

Reads reliability_scores_test.csv and frozen validation-tuned gating_thresholds.csv.
Uses 1,000 detection resamples with replacement, seed 42, and percentile 95%
intervals. Single-class resamples are excluded from AUROC and counted.
Here FTR means 1 - trusted precision; evaluate_gating.py instead reports false
trusted detections divided by all detections. The two denominators differ.
Run: python src/bootstrap_confidence_intervals.py"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_RESAMPLES = 1000
SEED = 42
METHODS = {
    "confidence_only": "confidence_only_score",
    "condition_calibrated_trustdet": "condition_calibrated_score",
}
REQUIRED_COLUMNS = ["is_correct", "confidence_only_score", "condition_calibrated_score"]


def load_test_rows():
    path = os.path.join(REPO_ROOT, "results", "reliability", "reliability_scores_test.csv")
    if not os.path.isfile(path):
        print(f"[ERROR] required file missing: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            print(f"[ERROR] required columns missing from {path}: {missing}")
            sys.exit(1)
        return list(reader)


def load_tau_trusted():
    path = os.path.join(REPO_ROOT, "results", "ablations", "gating_thresholds.csv")
    if not os.path.isfile(path):
        print(f"[ERROR] required file missing: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        thresholds = {r["method"]: float(r["tau_trusted"]) for r in csv.DictReader(f)}
    for m in METHODS:
        if m not in thresholds:
            print(f"[ERROR] method '{m}' not found in {path}")
            sys.exit(1)
    return thresholds


def bootstrap_method(scores, correct, tau_trusted, rng):
    n = len(scores)
    precisions, ftrs, aurocs = [], [], []
    n_skipped_auroc = 0

    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        s_sample = scores[idx]
        c_sample = correct[idx]

        trusted_mask = s_sample >= tau_trusted
        n_trusted = trusted_mask.sum()
        if n_trusted > 0:
            precision = c_sample[trusted_mask].mean()
        else:
            precision = np.nan
        precisions.append(precision)
        ftrs.append(1.0 - precision if precision == precision else np.nan)

        if roc_auc_score is not None and len(np.unique(c_sample)) > 1:
            aurocs.append(roc_auc_score(c_sample, s_sample))
        else:
            n_skipped_auroc += 1

    precisions = np.array([p for p in precisions if p == p])
    ftrs = np.array([f for f in ftrs if f == f])
    aurocs = np.array(aurocs)

    return precisions, ftrs, aurocs, n_skipped_auroc


def percentile_ci(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    load_paths()
    rows = load_test_rows()
    tau_trusted = load_tau_trusted()

    is_correct = np.array([1 if r["is_correct"] == "True" else 0 for r in rows], dtype=np.float64)

    out_dir = os.path.join(REPO_ROOT, "results", "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bootstrap_confidence_intervals.csv")

    fieldnames = ["method", "metric", "point_estimate", "ci_lower_95", "ci_upper_95",
                  "n_resamples", "n_resamples_used_for_auroc", "n_resamples_skipped_auroc", "seed"]
    out_rows = []

    for method, score_col in METHODS.items():
        scores = np.array([float(r[score_col]) for r in rows], dtype=np.float64)
        rng = np.random.default_rng(SEED)
        precisions, ftrs, aurocs, n_skipped = bootstrap_method(scores, is_correct, tau_trusted[method], rng)

        trusted_mask_point = scores >= tau_trusted[method]
        point_precision = is_correct[trusted_mask_point].mean() if trusted_mask_point.sum() > 0 else float("nan")
        point_ftr = 1.0 - point_precision
        point_auroc = roc_auc_score(is_correct, scores) if roc_auc_score is not None and len(np.unique(is_correct)) > 1 else float("nan")

        p_lo, p_hi = percentile_ci(precisions)
        f_lo, f_hi = percentile_ci(ftrs)
        a_lo, a_hi = percentile_ci(aurocs)

        out_rows.append({"method": method, "metric": "trusted_precision",
                          "point_estimate": round(point_precision, 4),
                          "ci_lower_95": round(p_lo, 4), "ci_upper_95": round(p_hi, 4),
                          "n_resamples": N_RESAMPLES, "n_resamples_used_for_auroc": "",
                          "n_resamples_skipped_auroc": "", "seed": SEED})
        out_rows.append({"method": method, "metric": "ftr_selective_risk",
                          "point_estimate": round(point_ftr, 4),
                          "ci_lower_95": round(f_lo, 4), "ci_upper_95": round(f_hi, 4),
                          "n_resamples": N_RESAMPLES, "n_resamples_used_for_auroc": "",
                          "n_resamples_skipped_auroc": "", "seed": SEED})
        out_rows.append({"method": method, "metric": "reliability_auroc",
                          "point_estimate": round(point_auroc, 4) if point_auroc == point_auroc else "",
                          "ci_lower_95": round(a_lo, 4) if a_lo == a_lo else "",
                          "ci_upper_95": round(a_hi, 4) if a_hi == a_hi else "",
                          "n_resamples": N_RESAMPLES,
                          "n_resamples_used_for_auroc": N_RESAMPLES - n_skipped,
                          "n_resamples_skipped_auroc": n_skipped, "seed": SEED})

        print(f"[OK] {method}: precision={point_precision:.4f} CI=[{p_lo:.4f},{p_hi:.4f}]  "
              f"FTR={point_ftr:.4f} CI=[{f_lo:.4f},{f_hi:.4f}]  "
              f"AUROC={point_auroc:.4f} CI=[{a_lo:.4f},{a_hi:.4f}] (skipped {n_skipped}/{N_RESAMPLES} for AUROC)")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()
