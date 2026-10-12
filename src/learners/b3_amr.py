"""
Baseline B3: AMR-simplified (Reproduced Base Paper Baseline).
Holistic Continual Learning under Concept Drift with Adaptive Memory Realignment
(Ashrafee et al., 2025; TMLR 2026).

Core Mechanism:
1. Class-balanced replay buffer holds past samples.
2. Per-class drift detection monitors class prediction error streams.
3. On drift for class c, outdated samples of class c are flushed from the buffer
   and refilled with fresh current-distribution samples, avoiding expensive full retraining.
"""
from typing import Dict, Any
import pympler.asizeof
from src.learners.base import StreamLearner
from src.learners.models import create_base_model
from src.memory.reservoir import ClassBalancedReservoir
from src.drift.detectors import PerClassDriftMonitor


class B3AMRLearner(StreamLearner):
    """B3: AMR-simplified reproduced baseline."""

    def __init__(self, model_type: str = "logistic", buffer_size: int = 500,
                 replay_batch: int = 8, detector_name: str = "adwin", seed: int = 42):
        self.model = create_base_model(model_type=model_type, seed=seed)
        self.buffer = ClassBalancedReservoir(capacity=buffer_size, seed=seed)
        self.per_class_monitor = PerClassDriftMonitor(detector_name=detector_name)
        self.replay_batch = replay_batch
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

        # Per-class drift monitoring (AMR core mechanism)
        drift_flag, drifted_class = self.per_class_monitor.update(y, error)

        if drift_flag and drifted_class is not None:
            self.concept_id += 1
            # AMR Flush: Purge outdated samples for the drifted class
            self.buffer.flush_class(drifted_class)

        # Train on incoming sample and refill reservoir with current distribution
        self.model.learn_one(x, y)
        self.buffer.add(x, y)

        # Replay batch from memory
        for rx, ry in self.buffer.sample(self.replay_batch):
            self.model.learn_one(rx, ry)

        return {
            "drift_flag": drift_flag,
            "concept_id": self.concept_id,
            "recurring_hit": False,
            "drifted_class": drifted_class if drift_flag else None,
            "buffer_size": self.buffer.size(),
        }

    def memory_bytes(self) -> int:
        return pympler.asizeof.asizeof(self.model) + self.buffer.memory_bytes()
