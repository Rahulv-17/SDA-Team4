"""
Baseline B0: Naive Online Learner.
Prequential predict-then-learn with no replay buffer and no drift adaptation.
Demonstrates catastrophic forgetting when concepts shift or recur.
"""
from typing import Dict, Any
import pympler.asizeof
from src.learners.base import StreamLearner
from src.learners.models import create_base_model


class B0NaiveLearner(StreamLearner):
    """B0: Naive online learner."""

    def __init__(self, model_type: str = "logistic", seed: int = 42):
        self.model = create_base_model(model_type=model_type, seed=seed)
        self.steps = 0

    def predict_one(self, x: Dict[str, float]) -> int:
        pred = self.model.predict_one(x)
        if pred is None:
            return 0
        return int(bool(pred))

    def learn_one(self, x: Dict[str, float], y: int) -> Dict[str, Any]:
        self.steps += 1
        self.model.learn_one(x, y)
        return {
            "drift_flag": False,
            "concept_id": 0,
            "recurring_hit": False,
            "buffer_size": 0,
        }

    def memory_bytes(self) -> int:
        return pympler.asizeof.asizeof(self.model)
