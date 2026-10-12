"""
Drift detection wrappers and monitors.
Provides:
- build_detector: factory for ADWIN, KSWIN, DDM
- PerClassDriftMonitor: per-class drift detection for AMR (B3)
- KSUncertaintyMonitor: sliding-window Kolmogorov-Smirnov test on model confidence
"""
from typing import Dict, Any, Optional, Tuple
from river import drift
import numpy as np
from scipy import stats


def build_detector(name: str = "adwin", **kwargs):
    """Factory for River drift detectors."""
    name_lower = name.lower()
    if name_lower == "adwin":
        delta = kwargs.get("delta", 0.002)
        return drift.ADWIN(delta=delta)
    elif name_lower == "kswin":
        alpha = kwargs.get("alpha", 0.005)
        window_size = kwargs.get("window_size", 100)
        stat_size = kwargs.get("stat_size", 30)
        return drift.KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size)
    elif name_lower == "ddm":
        return drift.binary.DDM()
    else:
        raise ValueError(f"Unknown drift detector: {name}. Supported: adwin, kswin, ddm")


class PerClassDriftMonitor:
    """
    Maintains independent drift detectors for each class.
    Used by AMR (B3) and CAM (E) for fine-grained concept tracking.
    """

    def __init__(self, detector_name: str = "adwin", **kwargs):
        self.detector_name = detector_name
        self.kwargs = kwargs
        self.detectors: Dict[int, Any] = {}

    def _get_detector(self, c: int):
        if c not in self.detectors:
            self.detectors[c] = build_detector(self.detector_name, **self.kwargs)
        return self.detectors[c]

    def update(self, y: int, error: int) -> Tuple[bool, Optional[int]]:
        """
        Update detector for class y with error (1 if y != y_pred else 0).
        Returns (drift_detected, drifted_class).
        """
        det = self._get_detector(y)
        det.update(error)
        if det.drift_detected:
            # Reset the drifted detector
            self.detectors[y] = build_detector(self.detector_name, **self.kwargs)
            return True, y
        return False, None

    def reset_class(self, c: int):
        if c in self.detectors:
            self.detectors[c] = build_detector(self.detector_name, **self.kwargs)

    def reset_all(self):
        self.detectors.clear()


class KSUncertaintyMonitor:
    """
    Uncertainty-based drift detector inspired by AMR paper (Ashrafee et al., 2025).
    Runs a 2-sample Kolmogorov-Smirnov test between a reference window and current window
    of prediction confidence scores.
    """

    def __init__(self, window_size: int = 100, p_val_threshold: float = 0.01):
        self.window_size = window_size
        self.p_val_threshold = p_val_threshold
        self.ref_window = []
        self.cur_window = []
        self.drift_detected = False

    def update(self, confidence: float) -> bool:
        """Update with model prediction confidence in [0, 1]."""
        self.drift_detected = False
        if len(self.ref_window) < self.window_size:
            self.ref_window.append(confidence)
            return False

        self.cur_window.append(confidence)
        if len(self.cur_window) >= self.window_size:
            # Perform two-sample KS test
            stat, p_val = stats.ks_2samp(self.ref_window, self.cur_window)
            if p_val < self.p_val_threshold:
                self.drift_detected = True
                # Promote current window to reference after drift
                self.ref_window = list(self.cur_window)
                self.cur_window = []
                return True
            else:
                # Slide window by half
                self.cur_window = self.cur_window[len(self.cur_window) // 2:]
        return False
