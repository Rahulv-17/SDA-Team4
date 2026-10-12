"""
Abstract Base Class for all Streaming Continual Learners.
Defines the shared StreamLearner interface required by Section 6 of PROJECT_SPEC.md.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class StreamLearner(ABC):
    """
    Shared interface for all continual streaming learners:
    - B0: Naive online (no replay, no drift adaptation)
    - B1: Vanilla replay
    - B2: Full relearning (FR)
    - B3: AMR-simplified (per-class drift flush & refill)
    - E:  CAM (Concept-Aware Memory enhancement)
    """

    @abstractmethod
    def predict_one(self, x: Dict[str, float]) -> int:
        """Prequential prediction step before seeing true label y."""
        pass

    @abstractmethod
    def learn_one(self, x: Dict[str, float], y: int) -> Dict[str, Any]:
        """
        Online learning step.
        Returns diagnostic info:
        - drift_flag: bool
        - concept_id: int
        - recurring_hit: bool
        - buffer_size: int
        """
        pass

    @abstractmethod
    def memory_bytes(self) -> int:
        """Returns current memory footprint in bytes."""
        pass


class PlaceholderLearner(StreamLearner):
    """Phase 1 placeholder learner predicting constant or parity."""

    def __init__(self, constant_pred: int = 1):
        self.constant_pred = constant_pred
        self.steps = 0

    def predict_one(self, x: Dict[str, float]) -> int:
        return self.constant_pred

    def learn_one(self, x: Dict[str, float], y: int) -> Dict[str, Any]:
        self.steps += 1
        return {
            "drift_flag": False,
            "concept_id": 0,
            "recurring_hit": False,
            "buffer_size": 0
        }

    def memory_bytes(self) -> int:
        return 128
