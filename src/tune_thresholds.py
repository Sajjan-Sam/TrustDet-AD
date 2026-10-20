#!/usr/bin/env python3
"""Select gating thresholds using validation detections only.

The trusted threshold maximizes precision subject to coverage >= 0.50, with
false-trusted detections / all detections as the tie-breaker. The rejected
threshold maximizes rejected-set purity subject to rejection rate >= 0.10 and
must not exceed the trusted threshold. Sorted cumulative sums give O(n log n)
threshold searches. Test scores are not loaded.
Run: python src/tune_thresholds.py"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths
from risk_aware_gate import METHODS, get_score, is_correct

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COVERAGE_FLOOR = 0.5
REJECTION_FLOOR = 0.10


def load_val_rows():
    path = os.path.join(REPO_ROOT, "results", "reliability", "reliability_scores_val.csv")
    if not os.path.isfile(path):
        print(f"[ERROR] {path} not found. Run src/run_trustdet_scoring.py first.")
        sys.exit(1)
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _group_boundaries(scores_sorted_desc):
    """Indices (0-indexed, inclusive) where each contiguous equal-score group ends."""
    n = len(scores_sorted_desc)
    change_points = np.where(np.diff(scores_sorted_desc) != 0)[0]
    return np.concatenate([change_points, [n - 1]])


def tune_tau_trusted(scores, correct, n_total):
    order = np.argsort(-scores)
    s_sorted = scores[order]
    c_sorted = correct[order]
    cum_correct = np.cumsum(c_sorted)

    group_ends = _group_boundaries(s_sorted)
    k_vals = group_ends + 1  # trusted-set size if tau = s_sorted[end]
    coverage = k_vals / n_total
    precision = cum_correct[group_ends] / k_vals
    false_trusted_rate = (k_vals - cum_correct[group_ends]) / n_total
    tau_vals = s_sorted[group_ends]

    valid = coverage >= COVERAGE_FLOOR
    if not np.any(valid):
        # Degenerate case (e.g. constant score): trust everything.
        tau = float(np.min(scores)) if len(scores) else 0.0
        n_correct = int(np.sum(correct))
        return tau, 1.0, n_correct / n_total if n_total else 0.0, (n_total - n_correct) / n_total if n_total else 0.0

    v_prec = precision[valid]
    v_ftr = false_trusted_rate[valid]
    v_tau = tau_vals[valid]
    v_cov = coverage[valid]
    best_prec = v_prec.max()
    tie = np.isclose(v_prec, best_prec)
    best_idx_within_tie = np.argmin(np.where(tie, v_ftr, np.inf))
    return float(v_tau[best_idx_within_tie]), float(v_cov[best_idx_within_tie]), \
        float(v_prec[best_idx_within_tie]), float(v_ftr[best_idx_within_tie])


def tune_tau_rejected(scores, correct, n_total, tau_trusted):
    mask = scores < tau_trusted
    if not np.any(mask):
        return tau_trusted, 0.0, 0.0
    sub_scores = scores[mask]
    sub_correct = correct[mask]

    order = np.argsort(sub_scores)  # ascending: rejected set = scores < tau, grow tau upward
    s_sorted = sub_scores[order]
    c_sorted = sub_correct[order]
    cum_incorrect = np.cumsum(1 - c_sorted)

    n = len(s_sorted)
    change_points = np.where(np.diff(s_sorted) != 0)[0]
    group_starts = np.concatenate([[0], change_points + 1])
    group_ends = np.concatenate([change_points, [n - 1]])  # end of each ascending equal-score group
    k_vals = group_ends + 1  # rejected-set size if tau = first score AFTER this group (strict "<")
    rejection_rate = k_vals / n_total
    purity = cum_incorrect[group_ends] / k_vals
    # tau_rejected = the score value strictly greater than this group (rule is score < tau), i.e.
    # the first value of the next group; the last group's candidate tau is tau_trusted itself
    # (the upper bound of this masked subset).
    tau_candidates = np.empty(len(group_ends))
    if len(group_ends) > 1:
        tau_candidates[:-1] = s_sorted[group_starts[1:]]
    tau_candidates[-1] = tau_trusted

    valid = rejection_rate >= REJECTION_FLOOR
    if not np.any(valid):
        return tau_trusted, 0.0, 0.0

    v_purity = purity[valid]
    v_tau = tau_candidates[valid]
    v_rej = rejection_rate[valid]
    best_idx = np.argmax(v_purity)
    return float(v_tau[best_idx]), float(v_rej[best_idx]), float(v_purity[best_idx])


def main():
    load_paths()
    rows = load_val_rows()
    n_total = len(rows)

    out_dir = os.path.join(REPO_ROOT, "results", "ablations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gating_thresholds.csv")

    fieldnames = ["method", "tau_trusted", "tau_rejected", "val_coverage",
                  "val_trusted_precision", "val_false_trusted_rate",
                  "val_rejection_rate", "val_rejected_purity"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for method in METHODS:
            scores = np.array([get_score(r, method) for r in rows], dtype=np.float64)
            correct = np.array([1 if is_correct(r) else 0 for r in rows], dtype=np.float64)

            tau_trusted, coverage, precision, false_trusted_rate = tune_tau_trusted(scores, correct, n_total)
            tau_rejected, rejection_rate, purity = tune_tau_rejected(scores, correct, n_total, tau_trusted)

            writer.writerow({
                "method": method, "tau_trusted": round(tau_trusted, 6),
                "tau_rejected": round(tau_rejected, 6), "val_coverage": round(coverage, 4),
                "val_trusted_precision": round(precision, 4),
                "val_false_trusted_rate": round(false_trusted_rate, 4),
                "val_rejection_rate": round(rejection_rate, 4),
                "val_rejected_purity": round(purity, 4),
            })
            print(f"[OK] {method}: tau_trusted={tau_trusted:.4f} (cov={coverage:.3f}, "
                  f"prec={precision:.4f}) tau_rejected={tau_rejected:.4f} (rej_rate={rejection_rate:.3f}, "
                  f"purity={purity:.4f})")

    print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()
