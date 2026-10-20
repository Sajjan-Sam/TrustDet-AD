"""Core risk-aware gating logic: score a detection into trusted / uncertain / rejected.

Gate rule:
  score >= tau_trusted   -> "trusted"
  score <  tau_rejected  -> "rejected"
  otherwise              -> "uncertain"
Requires tau_rejected <= tau_trusted (enforced by tune_thresholds.py's search space).
"""

METHODS = {
    # method_name -> (score_column_in_reliability_csv, higher_is_better)
    "confidence_only": "confidence_only_score",
    "calibration_only": None,          # derived: 1 - calibration_error
    "consistency_only": "prediction_consistency",
    "uncertainty_only": None,          # derived: 1 - uncertainty_score
    "xai_stability_only": "xai_stability",
    "fixed_trustdet": "fixed_trustdet_score",
    "condition_calibrated_trustdet": "condition_calibrated_score",
}


def get_score(row, method):
    if method == "calibration_only":
        return 1.0 - float(row["calibration_error"])
    if method == "uncertainty_only":
        return 1.0 - float(row["uncertainty_score"])
    col = METHODS[method]
    return float(row[col])


def gate_label(score, tau_trusted, tau_rejected):
    if score >= tau_trusted:
        return "trusted"
    if score < tau_rejected:
        return "rejected"
    return "uncertain"


def is_correct(row):
    return row["is_correct"] == "True" or row["is_correct"] is True
