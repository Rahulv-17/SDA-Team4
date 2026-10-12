"""
Baseline B2: Full Relearning (FR).
On detected drift, resets the model and fully retrains from the buffer.
AMR paper's expensive reference baseline.
"""
from typing import Dict, Any
import pympler.asizeof
from src.learners.base import StreamLearner
from src.learners.models import create_base_model
from src.memory.reservoir import ClassBalancedReservoir
from src.drift.detectors import build_detector


class B2FullRelearnLearner(StreamLearner):
    """B2: Full Relearning upon detected drift."""

    def __init__(self, model_type: str = "logistic", buffer_size: int = 500,
                 detector_name: str = "adwin", seed: int = 42):
        self.model_type = model_type
        self.seed = seed
        self.model = create_base_model(model_type=model_type, seed=seed)
        self.buffer = ClassBalancedReservoir(capacity=buffer_size, seed=seed)
        self.detector = build_detector(detector_name)
        self.steps = 0
        self.concept_id = 0

    def predict_one(self, x: Dict[str, float]) -> int:
        pred = self.model.predict_one(x)
        if pred is None:
            return 0
        return int(bool(pred))

    def learn_one(self, x: Dict[str, float], y: int) -> Dict[str, Any]:
        self.steps += 1
        y_pred = self.predict_one(x)
        error = 1 if y != y_pred else 0

        # Update drift detector
        self.detector.update(error)
        drift_flag = bool(self.detector.drift_detected)

        if drift_flag:
            self.concept_id += 1
            # Full Relearn: re-instantiate fresh model
            self.model = create_base_model(model_type=self.model_type, seed=self.seed + self.concept_id)
            # Retrain fresh model on all samples currently in buffer
            for bx, by in self.buffer.get_all_samples():
                self.model.learn_one(bx, by)
            # Reset detector
            self.detector = build_detector("adwin")

        # Train on current sample and add to buffer
        self.model.learn_one(x, y)
        self.buffer.add(x, y)

        return {
            "drift_flag": drift_flag,
            "concept_id": self.concept_id,
            "recurring_hit": False,
            "buffer_size": self.buffer.size(),
        }

    def memory_bytes(self) -> int:
        return pympler.asizeof.asizeof(self.model) + self.buffer.memory_bytes()
