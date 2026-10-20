"""Condition-calibrated reliability using validation-fitted logistic regression.

Numeric features: confidence, calibration error, prediction consistency,
uncertainty, saliency stability and runtime overhead. Condition IDs are one-hot
encoded. Test detections are scored using the frozen validation-fitted model."""
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

FEATURE_COLS = ["confidence", "calibration_error", "prediction_consistency",
                 "uncertainty_score", "xai_stability", "runtime_overhead"]


class ConditionCalibrator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.fitted_conditions = None

    def _build_features(self, rows, fit_encoder=False):
        numeric = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows])
        condition_ids = np.array([[r["condition_id"]] for r in rows])
        if fit_encoder:
            cond_onehot = self.encoder.fit_transform(condition_ids)
        else:
            cond_onehot = self.encoder.transform(condition_ids)
        return np.hstack([numeric, cond_onehot])

    def fit(self, val_rows):
        X = self._build_features(val_rows, fit_encoder=True)
        y = np.array([int(r["is_correct"] == "True" or r["is_correct"] is True) for r in val_rows])
        if len(set(y.tolist())) < 2:
            raise ValueError("val rows contain only one class -- cannot fit logistic regression")
        self.model.fit(X, y)

    def score(self, rows):
        X = self._build_features(rows, fit_encoder=False)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
