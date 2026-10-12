"""
Baseline B1: Vanilla Replay Learner.
Fixed-size reservoir buffer, replaying a small batch every step, but NO drift adaptation.
"""
from typing import Dict, Any
import pympler.asizeof
from src.learners.base import StreamLearner
from src.learners.models import create_base_model
from src.memory.reservoir import ClassBalancedReservoir


class B1ReplayLearner(StreamLearner):
    """B1: Vanilla Rehearsal Learner without drift awareness."""

    def __init__(self, model_type: str = "logistic", buffer_size: int = 500,
                 replay_batch: int = 8, seed: int = 42):
        self.model = create_base_model(model_type=model_type, seed=seed)
        self.buffer = ClassBalancedReservoir(capacity=buffer_size, seed=seed)
        self.replay_batch = replay_batch
        self.steps = 0

    def predict_one(self, x: Dict[str, float]) -> int:
        pred = self.model.predict_one(x)
        if pred is None:
            return 0
        return int(bool(pred))

    def learn_one(self, x: Dict[str, float], y: int) -> Dict[str, Any]:
        self.steps += 1
        # Train on incoming sample
        self.model.learn_one(x, y)
        self.buffer.add(x, y)

        # Replay small batch from buffer
        for rx, ry in self.buffer.sample(self.replay_batch):
            self.model.learn_one(rx, ry)

        return {
            "drift_flag": False,
            "concept_id": 0,
            "recurring_hit": False,
            "buffer_size": self.buffer.size(),
        }

    def memory_bytes(self) -> int:
        return pympler.asizeof.asizeof(self.model) + self.buffer.memory_bytes()
