#!/usr/bin/env python3
"""Apply frozen (val-tuned) thresholds to the TEST split and compute all gating metrics.
Usage: python src/evaluate_gating.py
Reads results/reliability/reliability_scores_test.csv, results/ablations/gating_thresholds.csv.
Writes results/ablations/gating_metrics_test.csv, coverage_precision_curve.csv,
selective_risk_curve.csv.

Thresholds are read from gating_thresholds.csv (fit on val by tune_thresholds.py) and applied
here unchanged -- this file never re-tunes anything on test.
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths
from risk_aware_gate import METHODS, get_score, gate_label, is_correct

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = average_precision_score = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(split):
    path = os.path.join(REPO_ROOT, "results", "reliability", f"reliability_scores_{split}.csv")
    if not os.path.isfile(path):
        print(f"[ERROR] {path} not found.")
        sys.exit(1)
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def load_thresholds():
    path = os.path.join(REPO_ROOT, "results", "ablations", "gating_thresholds.csv")
    if not os.path.isfile(path):
        print(f"[ERROR] {path} not found. Run src/tune_thresholds.py first.")
        sys.exit(1)
    with open(path, "r") as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def gating_metrics(rows, method, tau_trusted, tau_rejected):
    n_total = len(rows)
    labels = [gate_label(get_score(r, method), tau_trusted, tau_rejected) for r in rows]
    correctness = [is_correct(r) for r in rows]

    trusted = [c for l, c in zip(labels, correctness) if l == "trusted"]
    rejected = [c for l, c in zip(labels, correctness) if l == "rejected"]
    uncertain = [c for l, c in zip(labels, correctness) if l == "uncertain"]

    coverage = len(trusted) / n_total
    trusted_precision = (sum(trusted) / len(trusted)) if trusted else float("nan")
    false_trusted_rate = (len(trusted) - sum(trusted)) / n_total
    rejection_rate = len(rejected) / n_total
    uncertain_rate = len(uncertain) / n_total
    selective_risk = (1.0 - trusted_precision) if trusted else float("nan")

    scores = [get_score(r, method) for r in rows]
    y = [int(c) for c in correctness]
    if roc_auc_score is not None and len(set(y)) > 1:
        auroc = roc_auc_score(y, scores)
        auprc = average_precision_score(y, scores)
    else:
        auroc = auprc = float("nan")

    return {
        "method": method, "coverage": round(coverage, 4),
        "trusted_precision": round(trusted_precision, 4) if trusted else "",
        "false_trusted_rate": round(false_trusted_rate, 4),
        "rejection_rate": round(rejection_rate, 4), "uncertain_rate": round(uncertain_rate, 4),
        "reliability_auroc": round(auroc, 4) if auroc == auroc else "",
        "reliability_auprc": round(auprc, 4) if auprc == auprc else "",
        "selective_risk": round(selective_risk, 4) if trusted else "",
        "n_total": n_total, "n_trusted": len(trusted), "n_rejected": len(rejected),
        "n_uncertain": len(uncertain),
    }


def no_gating_metrics(rows):
    n_total = len(rows)
    correctness = [is_correct(r) for r in rows]
    precision = sum(correctness) / n_total
    return {
        "method": "no_gating", "coverage": 1.0, "trusted_precision": round(precision, 4),
        "false_trusted_rate": round(1 - precision, 4), "rejection_rate": 0.0, "uncertain_rate": 0.0,
        "reliability_auroc": "", "reliability_auprc": "",  # undefined: no continuous score to rank by
        "selective_risk": round(1 - precision, 4), "n_total": n_total, "n_trusted": n_total,
        "n_rejected": 0, "n_uncertain": 0,
    }


def coverage_precision_curve(rows, method):
    """Sweep score threshold -> (coverage, trusted_precision) for plotting.
    Vectorized (sorted-cumulative-sum), not a per-candidate O(n) recount -- with near-continuous
    scores over tens of thousands of rows the naive version does not finish in reasonable time."""
    n_total = len(rows)
    scores = np.array([get_score(r, method) for r in rows], dtype=np.float64)
    correct = np.array([1 if is_correct(r) else 0 for r in rows], dtype=np.float64)

    order = np.argsort(-scores)
    s_sorted = scores[order]
    c_sorted = correct[order]
    cum_correct = np.cumsum(c_sorted)

    change_points = np.where(np.diff(s_sorted) != 0)[0]
    group_ends = np.concatenate([change_points, [n_total - 1]])
    k_vals = group_ends + 1
    coverage = k_vals / n_total
    precision = cum_correct[group_ends] / k_vals
    tau_vals = s_sorted[group_ends]

    return [
        {"method": method, "threshold": round(float(t), 4), "coverage": round(float(c), 4),
         "trusted_precision": round(float(p), 4)}
        for t, c, p in zip(tau_vals, coverage, precision)
    ]


def selective_risk_curve(rows, method):
    """Sweep coverage in steps of 0.05 -> risk (error rate) when accepting the top-coverage
    fraction of detections ranked by score, descending."""
    n_total = len(rows)
    ranked = sorted(rows, key=lambda r: -get_score(r, method))
    curve = []
    for step in range(1, 21):
        coverage = step / 20.0
        k = max(1, int(round(coverage * n_total)))
        top_k = ranked[:k]
        n_correct = sum(1 for r in top_k if is_correct(r))
        risk = 1.0 - (n_correct / k)
        curve.append({"method": method, "coverage": round(coverage, 2), "risk": round(risk, 4)})
    return curve


def main():
    load_paths()
    test_rows = load_rows("test")
    thresholds = load_thresholds()

    out_dir = os.path.join(REPO_ROOT, "results", "ablations")
    os.makedirs(out_dir, exist_ok=True)

    metrics_fieldnames = ["method", "coverage", "trusted_precision", "false_trusted_rate",
                           "rejection_rate", "uncertain_rate", "reliability_auroc",
                           "reliability_auprc", "selective_risk", "n_total", "n_trusted",
                           "n_rejected", "n_uncertain"]
    all_metrics = [no_gating_metrics(test_rows)]

    cov_prec_rows = []
    risk_rows = []
    for method in METHODS:
        t = thresholds[method]
        tau_trusted, tau_rejected = float(t["tau_trusted"]), float(t["tau_rejected"])
        m = gating_metrics(test_rows, method, tau_trusted, tau_rejected)
        all_metrics.append(m)
        cov_prec_rows.extend(coverage_precision_curve(test_rows, method))
        risk_rows.extend(selective_risk_curve(test_rows, method))
        print(f"[OK] {method}: coverage={m['coverage']} trusted_precision={m['trusted_precision']} "
              f"AUROC={m['reliability_auroc']} selective_risk={m['selective_risk']}")

    with open(os.path.join(out_dir, "gating_metrics_test.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    with open(os.path.join(out_dir, "coverage_precision_curve.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "threshold", "coverage", "trusted_precision"])
        writer.writeheader()
        writer.writerows(cov_prec_rows)

    with open(os.path.join(out_dir, "selective_risk_curve.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "coverage", "risk"])
        writer.writeheader()
        writer.writerows(risk_rows)

    print(f"[DONE] wrote gating_metrics_test.csv, coverage_precision_curve.csv, selective_risk_curve.csv")


if __name__ == "__main__":
    main()
