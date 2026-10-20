"""Uncertainty score component: test-time-augmentation confidence variance.

Reuses the same 4-transform matches computed for prediction_consistency (no extra inference).
For each transform where the detection persisted (IoU>=0.5, same class -- see
prediction_consistency.consistency_and_matches), we have the matched detection's confidence.
uncertainty_score = population stdev of {original confidence} U {matched confidences}, scaled
into roughly [0,1] and clipped. Fewer than 2 persisting confidences (i.e. the detection vanished
under 3+ of the 4 transforms) means we cannot estimate a meaningful spread -- treated as maximum
uncertainty (1.0), not zero, since a detection that mostly disappears under perturbation is by
definition unreliable.
"""
import statistics

UNCERTAINTY_SCALE = 4.0  # maps typical confidence-jitter stdev (~0.0-0.25) onto [0,1]


def uncertainty_for_detection(confidence, matches):
    """matches: list of (transform_name, matched_det, iou) from consistency_and_matches."""
    confidences = [confidence] + [m[1]["score"] for m in matches]
    if len(confidences) < 2:
        return 1.0
    spread = statistics.pstdev(confidences)
    return max(0.0, min(1.0, spread * UNCERTAINTY_SCALE))
