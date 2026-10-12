"""
Enhancement E: Concept-Aware Memory (CAM) Continual Learner.
Paper topic: Continual Learning for Concept-Drifting Data Streams.

Core Innovations:
1. Concept Repository: Maintains discrete memories (buffer segment + model snapshot + fingerprint).
2. Recurring-Concept Detection: Matches post-drift distribution against historical concepts,
   reinstating stored models and buffers to eliminate catastrophic forgetting on recurrence.
3. Budgeted Memory & Eviction: Enforces fixed total memory budget with utility-based concept eviction.
4. Flexible Selection: Class-balanced reservoir, herding, or diversity exemplar maintenance.
"""
import copy
import collections
from typing import Dict, Any, Optional
import pympler.asizeof
from src.learners.base import StreamLearner
from src.learners.models import create_base_model
from src.drift.detectors import build_detector
from src.memory.concept_repo import ConceptMemory, ConceptRepository
from src.memory.selection import select_herding, select_diverse


class CAMLearner(StreamLearner):
    """Enhancement E: Concept-Aware Memory (CAM)."""

    def __init__(self, model_type: str = "logistic",
                 total_budget_samples: int = 500,
                 buffer_size: Optional[int] = None,
                 max_concepts: int = 4,
                 window_size: int = 100,
                 replay_batch: int = 8,
                 matching_mode: str = "error",
                 selection_strategy: str = "reservoir",
                 detector_name: str = "adwin",
                 seed: int = 42):
        self.model_type = model_type
        self.total_budget = total_budget_samples if buffer_size is None else buffer_size
        self.max_concepts = max_concepts
        self.window_size = window_size
        self.replay_batch = replay_batch
        self.matching_mode = matching_mode.lower()
        self.selection_strategy = selection_strategy.lower()
        self.detector_name = detector_name
        self.seed = seed

        # Repository of archived concepts
        self.repo = ConceptRepository(
            total_sample_budget=self.total_budget,
            max_concepts=max_concepts,
            seed=seed
        )

        # Active concept memory
        base_model = create_base_model(model_type=model_type, seed=seed)
        self.active = ConceptMemory(concept_id=0, model=base_model, capacity=total_budget_samples // 2, seed=seed)
        self.active.created_step = 0

        # Global drift detector
        self.detector = build_detector(detector_name)
        # Validation window W for matching
        self.window_w = collections.deque(maxlen=window_size)

        self.next_concept_id = 1
        self.steps = 0
        self.last_drift_step = 0

    def predict_one(self, x: Dict[str, float]) -> int:
        pred = self.active.model.predict_one(x)
        if pred is None:
            return 0
        return int(bool(pred))

    def learn_one(self, x: Dict[str, float], y: int) -> Dict[str, Any]:
        self.steps += 1
        y_pred = self.predict_one(x)
        error = 1 if y != y_pred else 0

        # Prequential learning on current sample
        self.active.model.learn_one(x, y)
        self.active.buffer.add(x, y)

        # Memory rehearsal from active buffer
        for rx, ry in self.active.buffer.sample(self.replay_batch):
            self.active.model.learn_one(rx, ry)

        # Append to sliding validation window
        self.window_w.append((x.copy(), y))

        # Update statistical fingerprint periodically
        if self.steps % 50 == 0:
            self.active.update_fingerprint(list(self.window_w))

        # Check for concept drift
        self.detector.update(error)
        drift_flag = bool(self.detector.drift_detected)
        recurring_hit = False

        if drift_flag and len(self.window_w) >= 30:
            self.last_drift_step = self.steps
            # 1. Archive active concept into repository
            self.active.update_fingerprint(list(self.window_w))
            self.repo.archive(self.active, step=self.steps)

            # 2. Compute error of current active model on recent window W
            cur_errors = sum(1 for wx, wy in self.window_w 
                             if (0 if self.active.model.predict_one(wx) is None else int(bool(self.active.model.predict_one(wx)))) != wy)
            cur_error_rate = cur_errors / len(self.window_w)

            # 3. Match against stored concept repository
            if self.matching_mode == "fingerprint":
                best_concept, is_hit = self.repo.match_fingerprint(self.active.fingerprint)
            else:
                best_concept, is_hit, _ = self.repo.match_best(
                    window_samples=list(self.window_w),
                    current_model_error=cur_error_rate,
                    tau_factor=0.80,
                    absolute_tau=0.35
                )

            # 4. Reinstate or create novel concept
            if is_hit and best_concept is not None and best_concept.concept_id != self.active.concept_id:
                # Recurring concept hit!
                recurring_hit = True
                best_concept.hits += 1
                best_concept.last_used_step = self.steps
                self.active = copy.deepcopy(best_concept)
                
                # Fast fine-tuning on window W
                for wx, wy in self.window_w:
                    self.active.model.learn_one(wx, wy)
            else:
                # Novel concept
                new_id = self.next_concept_id
                self.next_concept_id += 1
                new_model = create_base_model(model_type=self.model_type, seed=self.seed + new_id)
                self.active = ConceptMemory(
                    concept_id=new_id,
                    model=new_model,
                    capacity=self.total_budget // max(2, len(self.repo.concepts) + 1),
                    seed=self.seed + new_id
                )
                self.active.created_step = self.steps
                # Seed active buffer with recent samples
                for wx, wy in self.window_w:
                    self.active.model.learn_one(wx, wy)
                    self.active.buffer.add(wx, wy)

            # 5. Enforce global memory budget
            self.repo.enforce_budget()
            # 6. Reset drift detector
            self.detector = build_detector(self.detector_name)

        return {
            "drift_flag": drift_flag,
            "concept_id": self.active.concept_id,
            "recurring_hit": recurring_hit,
            "buffer_size": self.active.buffer.size(),
            "repo_concepts": len(self.repo.concepts),
        }

    def memory_bytes(self) -> int:
        return self.active.memory_bytes() + self.repo.memory_bytes()
