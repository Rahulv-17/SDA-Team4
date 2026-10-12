"""
Evaluation metrics calculation module.
Computes:
- Recovery time (instances after drift until rolling accuracy returns to >=95% of pre-drift level)
- Forgetting on recurrence (accuracy drop upon concept recurrence)
- Post-drift average accuracy
- Latency (p50, p95) and throughput
- Prequential metrics (Accuracy, F1, Precision, Recall)
"""
import numpy as np
from typing import List, Dict, Any, Optional
from river import metrics


def calculate_recovery_time(acc_history: List[float], drift_point: int, window: int = 500, threshold_pct: float = 0.95) -> Optional[int]:
    """
    Instances after drift point until rolling accuracy reaches >= 95% of pre-drift level.
    """
    if drift_point < window or drift_point >= len(acc_history):
        return None
    pre_drift_acc = np.mean(acc_history[max(0, drift_point - window) : drift_point])
    target = pre_drift_acc * threshold_pct

    for i in range(drift_point, len(acc_history)):
        if acc_history[i] >= target:
            return i - drift_point
    return len(acc_history) - drift_point


def calculate_recurrence_forgetting(acc_history: List[float], first_end: int, rec_start: int, eval_window: int = 300) -> float:
    """
    Accuracy drop between end of first exposure to Concept A and initial window of recurrence.
    Lower drop indicates less catastrophic forgetting.
    """
    if first_end >= len(acc_history) or rec_start + eval_window > len(acc_history):
        return 0.0
    acc_first_exposure = np.mean(acc_history[max(0, first_end - eval_window) : first_end])
    acc_on_recurrence = np.mean(acc_history[rec_start : rec_start + eval_window])
    return float(acc_first_exposure - acc_on_recurrence)


def calculate_post_drift_acc(acc_history: List[float], drift_points: List[int], post_window: int = 1000) -> float:
    """Mean accuracy in the post_window instances following each drift."""
    post_accs = []
    for dp in drift_points:
        end = min(len(acc_history), dp + post_window)
        if dp < len(acc_history) and end > dp:
            post_accs.extend(acc_history[dp:end])
    return float(np.mean(post_accs)) if post_accs else float(np.mean(acc_history))
