"""
Concept-Aware Memory (CAM) Repository.
Maintains archived concept memories, performs error-based and fingerprint-based matching,
and strictly enforces global sample and memory budgets with utility-based eviction.
"""
import copy
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pympler.asizeof
from src.memory.reservoir import ClassBalancedReservoir


class ConceptMemory:
    """Represents a discrete concept state: buffer, model snapshot, and statistical fingerprint."""

    def __init__(self, concept_id: int, model, capacity: int = 150, seed: int = 42):
        self.concept_id = concept_id
        self.model = copy.deepcopy(model)
        self.buffer = ClassBalancedReservoir(capacity=capacity, seed=seed)
        self.fingerprint: Dict[str, Any] = {}
        self.hits: int = 0
        self.last_used_step: int = 0
        self.created_step: int = 0

    def update_fingerprint(self, samples: List[Tuple[Dict[str, float], int]]):
        """Compute running feature mean and class prior."""
        if not samples:
            return
        keys = list(samples[0][0].keys())
        means = {k: float(np.mean([x.get(k, 0.0) for x, _ in samples])) for k in keys}
        std_devs = {k: float(np.std([x.get(k, 0.0) for x, _ in samples])) for k in keys}
        class_prior = float(np.mean([y for _, y in samples]))
        self.fingerprint = {
            "means": means,
            "stds": std_devs,
            "class_prior": class_prior,
        }

    def memory_bytes(self) -> int:
        return pympler.asizeof.asizeof(self.model) + self.buffer.memory_bytes()


class ConceptRepository:
    """
    Budgeted repository of concept memories.
    Detects recurring concepts via model evaluation or fingerprint distance on recent window W.
    """

    def __init__(self, total_sample_budget: int = 500, max_concepts: int = 4, seed: int = 42):
        self.total_sample_budget = total_sample_budget
        self.max_concepts = max_concepts
        self.seed = seed
        self.concepts: List[ConceptMemory] = []

    def get_concept(self, concept_id: int) -> Optional[ConceptMemory]:
        for c in self.concepts:
            if c.concept_id == concept_id:
                return c
        return None

    def archive(self, concept: ConceptMemory, step: int):
        """Archive a concept into the repository."""
        concept.last_used_step = step
        # Check if already exists in repo
        existing = self.get_concept(concept.concept_id)
        if existing is None:
            self.concepts.append(concept)
        else:
            # Update snapshot and buffer
            existing.model = copy.deepcopy(concept.model)
            existing.buffer = copy.deepcopy(concept.buffer)
            existing.fingerprint = dict(concept.fingerprint)
            existing.last_used_step = step

        self.enforce_budget()

    def match_best(self, window_samples: List[Tuple[Dict[str, float], int]],
                   current_model_error: float,
                   tau_factor: float = 0.75,
                   absolute_tau: float = 0.30) -> Tuple[Optional[ConceptMemory], bool, float]:
        """
        Error-based matching: evaluate stored concept models on validation window W.
        Returns (best_concept, is_hit, best_error).
        """
        if not self.concepts or not window_samples:
            return None, False, 1.0

        best_concept = None
        best_error = 1.0

        for cm in self.concepts:
            errors = 0
            for x, y in window_samples:
                pred = cm.model.predict_one(x)
                pred_label = 0 if pred is None else int(bool(pred))
                if pred_label != y:
                    errors += 1
            error_rate = errors / len(window_samples)

            if error_rate < best_error:
                best_error = error_rate
                best_concept = cm

        # Hit condition: significantly better than current model error or below absolute threshold
        threshold = min(absolute_tau, tau_factor * max(0.1, current_model_error))
        is_hit = (best_concept is not None) and (best_error <= threshold)

        return best_concept, is_hit, best_error

    def match_fingerprint(self, cur_fingerprint: Dict[str, Any], dist_threshold: float = 1.5) -> Tuple[Optional[ConceptMemory], bool]:
        """Ablation: Fingerprint distance-based matching."""
        if not self.concepts or not cur_fingerprint:
            return None, False

        cur_means = cur_fingerprint.get("means", {})
        cur_prior = cur_fingerprint.get("class_prior", 0.5)

        best_concept = None
        best_dist = float("inf")

        for cm in self.concepts:
            fp = cm.fingerprint
            if not fp:
                continue
            stored_means = fp.get("means", {})
            dist = sum((cur_means.get(k, 0.0) - stored_means.get(k, 0.0)) ** 2 for k in cur_means) ** 0.5
            dist += abs(cur_prior - fp.get("class_prior", 0.5)) * 2.0

            if dist < best_dist:
                best_dist = dist
                best_concept = cm

        is_hit = (best_concept is not None) and (best_dist < dist_threshold)
        return best_concept, is_hit

    def enforce_budget(self):
        """
        Enforce strict sample and concept count budgets.
        Evicts lowest utility concept: utility = hits * 1000 + last_used_step.
        """
        # 1. Evict excess concepts
        while len(self.concepts) > self.max_concepts:
            # Sort by utility ascending (lowest utility evicted first)
            self.concepts.sort(key=lambda c: (c.hits, c.last_used_step))
            evicted = self.concepts.pop(0)

        # 2. Reallocate buffer capacity equally across stored concepts
        if self.concepts:
            per_concept_capacity = max(20, self.total_sample_budget // len(self.concepts))
            for c in self.concepts:
                c.buffer.capacity = per_concept_capacity
                c.buffer._enforce_global_capacity()

    def total_samples(self) -> int:
        return sum(c.buffer.size() for c in self.concepts)

    def memory_bytes(self) -> int:
        return sum(c.memory_bytes() for c in self.concepts)
