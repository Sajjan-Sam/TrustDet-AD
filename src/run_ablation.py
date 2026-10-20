#!/usr/bin/env python3
"""Compare gating methods with the confidence-only baseline.

Reads results/ablations/gating_metrics_test.csv and writes metric differences to
component_ablation_test.csv. Run after tune_thresholds.py and evaluate_gating.py.
Run: python src/run_ablation.py"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_paths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    load_paths()
    metrics_path = os.path.join(REPO_ROOT, "results", "ablations", "gating_metrics_test.csv")
    if not os.path.isfile(metrics_path):
        print(f"[ERROR] {metrics_path} not found. Run tune_thresholds.py then evaluate_gating.py first.")
        sys.exit(1)

    with open(metrics_path, "r") as f:
        rows = {r["method"]: r for r in csv.DictReader(f)}

    if "confidence_only" not in rows:
        print("[ERROR] confidence_only baseline missing from gating_metrics_test.csv")
        sys.exit(1)
    baseline = rows["confidence_only"]

    out_path = os.path.join(REPO_ROOT, "results", "ablations", "component_ablation_test.csv")
    fieldnames = ["method", "trusted_precision", "trusted_precision_delta_vs_confidence_only",
                  "selective_risk", "selective_risk_delta_vs_confidence_only",
                  "reliability_auroc", "auroc_delta_vs_confidence_only", "coverage",
                  "beats_confidence_only_on_precision", "beats_confidence_only_on_auroc"]

    out_rows = []
    for method, r in rows.items():
        prec = to_float(r["trusted_precision"])
        risk = to_float(r["selective_risk"])
        auroc = to_float(r["reliability_auroc"])
        base_prec = to_float(baseline["trusted_precision"])
        base_risk = to_float(baseline["selective_risk"])
        base_auroc = to_float(baseline["reliability_auroc"])

        prec_delta = (prec - base_prec) if (prec is not None and base_prec is not None) else None
        risk_delta = (risk - base_risk) if (risk is not None and base_risk is not None) else None
        auroc_delta = (auroc - base_auroc) if (auroc is not None and base_auroc is not None) else None

        out_rows.append({
            "method": method,
            "trusted_precision": r["trusted_precision"],
            "trusted_precision_delta_vs_confidence_only": round(prec_delta, 4) if prec_delta is not None else "",
            "selective_risk": r["selective_risk"],
            "selective_risk_delta_vs_confidence_only": round(risk_delta, 4) if risk_delta is not None else "",
            "reliability_auroc": r["reliability_auroc"],
            "auroc_delta_vs_confidence_only": round(auroc_delta, 4) if auroc_delta is not None else "",
            "coverage": r["coverage"],
            "beats_confidence_only_on_precision": (prec_delta is not None and prec_delta > 0),
            "beats_confidence_only_on_auroc": (auroc_delta is not None and auroc_delta > 0),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] wrote {out_path}")
    main_method = rows.get("condition_calibrated_trustdet")
    if main_method:
        prec_delta = to_float(main_method["trusted_precision"]) - to_float(baseline["trusted_precision"]) \
            if to_float(main_method["trusted_precision"]) is not None and to_float(baseline["trusted_precision"]) is not None else None
        verdict = "BEATS" if (prec_delta is not None and prec_delta > 0) else "DOES NOT BEAT"
        print(f"[HEADLINE] condition_calibrated_trustdet {verdict} confidence_only on trusted_precision "
              f"({main_method['trusted_precision']} vs {baseline['trusted_precision']})")


if __name__ == "__main__":
    main()
