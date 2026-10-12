"""
Unit tests for Phase 3: Enhancement E (CAM - Concept-Aware Memory).
Requirements from Section 8:
- Matching selects the right concept on A-B-A sequence
- Total memory never exceeds budget
- Eviction picks the intended concept (utility-based policy)
- CAM Learner interface compliance
"""
import pytest
import numpy as np
from src.learners.models import create_base_model
from src.memory.concept_repo import ConceptMemory, ConceptRepository
from src.learners.e_cam import CAMLearner


def test_matching_selects_right_concept_on_aba():
    """Verify concept repository matching selects Concept A when A recurs after B."""
    repo = ConceptRepository(total_sample_budget=500, max_concepts=5, seed=42)

    # Train Concept A: separable with x0 > 0 -> 1, x0 <= 0 -> 0
    model_a = create_base_model("logistic")
    samples_a = [({"x0": 2.0, "x1": 1.0}, 1), ({"x0": -2.0, "x1": -1.0}, 0)] * 50
    for x, y in samples_a:
        model_a.learn_one(x, y)
    mem_a = ConceptMemory(concept_id=1, model=model_a, capacity=100)
    for x, y in samples_a[:50]:
        mem_a.buffer.add(x, y)
    mem_a.update_fingerprint(samples_a[:50])
    repo.archive(mem_a, step=100)

    # Train Concept B: opposite relationship x0 > 0 -> 0, x0 <= 0 -> 1
    model_b = create_base_model("logistic")
    samples_b = [({"x0": 2.0, "x1": 1.0}, 0), ({"x0": -2.0, "x1": -1.0}, 1)] * 50
    for x, y in samples_b:
        model_b.learn_one(x, y)
    mem_b = ConceptMemory(concept_id=2, model=model_b, capacity=100)
    for x, y in samples_b[:50]:
        mem_b.buffer.add(x, y)
    mem_b.update_fingerprint(samples_b[:50])
    repo.archive(mem_b, step=200)

    # Test recurrence: incoming validation window from Concept A
    eval_window_a = [({"x0": 2.5, "x1": 0.5}, 1), ({"x0": -2.5, "x1": -0.5}, 0)] * 20
    best_concept, is_hit, error_rate = repo.match_best(
        eval_window_a, current_model_error=0.8, tau_factor=0.8, absolute_tau=0.3
    )

    assert best_concept is not None, "Repo must find a candidate concept"
    assert is_hit is True, "Must register as a recurring concept hit"
    assert best_concept.concept_id == 1, f"Expected Concept 1 (A), got {best_concept.concept_id}"
    assert error_rate < 0.1, f"Error rate on Concept A should be near 0, got {error_rate}"


def test_total_memory_budget_enforcement():
    """Verify total sample budget across all archived concepts is strictly enforced."""
    budget = 120
    repo = ConceptRepository(total_sample_budget=budget, max_concepts=3, seed=42)

    for cid in range(1, 4):
        model = create_base_model("logistic")
        cm = ConceptMemory(concept_id=cid, model=model, capacity=100)
        # Try to overfill buffer with 80 samples each
        for i in range(80):
            cm.buffer.add({"x": float(i)}, i % 2)
        repo.archive(cm, step=cid * 50)

    assert len(repo.concepts) <= 3
    # Total samples in repo must not exceed total budget
    assert repo.total_samples() <= budget, f"Total samples {repo.total_samples()} exceeded budget {budget}"


def test_eviction_utility_policy():
    """Verify eviction discards the concept with lowest utility (fewest hits, oldest)."""
    repo = ConceptRepository(total_sample_budget=500, max_concepts=2, seed=42)

    # Concept 1: 0 hits, used at step 100
    cm1 = ConceptMemory(concept_id=1, model=create_base_model("logistic"))
    cm1.hits = 0
    repo.archive(cm1, step=100)

    # Concept 2: 3 hits, used at step 200
    cm2 = ConceptMemory(concept_id=2, model=create_base_model("logistic"))
    cm2.hits = 3
    repo.archive(cm2, step=200)

    # Add Concept 3: exceeds max_concepts=2, must evict cm1 (0 hits)
    cm3 = ConceptMemory(concept_id=3, model=create_base_model("logistic"))
    cm3.hits = 1
    repo.archive(cm3, step=300)

    remaining_ids = [c.concept_id for c in repo.concepts]
    assert 1 not in remaining_ids, "Concept 1 (lowest utility) should have been evicted"
    assert 2 in remaining_ids, "Concept 2 (high hits) must be retained"
    assert 3 in remaining_ids, "Concept 3 must be retained"


def test_cam_learner_interface_and_execution():
    """Verify CAM learner operates through the standard StreamLearner interface."""
    learner = CAMLearner(total_budget_samples=200, max_concepts=3)
    x = {"x0": 1.2, "x1": -0.5, "x2": 0.8}

    pred = learner.predict_one(x)
    assert pred in (0, 1)

    info = learner.learn_one(x, y=1)
    assert "drift_flag" in info
    assert "concept_id" in info
    assert "recurring_hit" in info
    assert learner.memory_bytes() > 0
