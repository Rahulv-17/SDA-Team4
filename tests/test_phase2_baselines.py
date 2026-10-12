"""
Unit tests for Phase 2: Baselines B0 to B3, Buffers, and Drift Detection.
Tests required by Section 8:
- Buffer class balance
- Reservoir uniformity and capacity enforcement
- AMR flush-and-refill correctness
- Detector fires on synthetic abrupt shift
- Shared StreamLearner interface compliance
"""
import pytest
import numpy as np
from src.memory.reservoir import ClassBalancedReservoir
from src.drift.detectors import build_detector, PerClassDriftMonitor
from src.learners.b0_naive import B0NaiveLearner
from src.learners.b1_replay import B1ReplayLearner
from src.learners.b2_full_relearn import B2FullRelearnLearner
from src.learners.b3_amr import B3AMRLearner


def test_buffer_class_balance():
    """Verify ClassBalancedReservoir balances sample slots evenly across classes."""
    buffer = ClassBalancedReservoir(capacity=100, seed=42)
    # Feed 200 samples of class 0 and 50 samples of class 1
    for i in range(200):
        buffer.add({"x0": float(i)}, 0)
    for i in range(50):
        buffer.add({"x0": float(i)}, 1)

    counts = buffer.class_counts()
    assert 0 in counts and 1 in counts
    assert buffer.size() <= 100
    # Class 0 should not starve class 1; each should be close to 50
    assert counts[0] <= 55
    assert counts[1] <= 55
    assert counts[1] >= 40


def test_reservoir_capacity_and_sampling():
    """Verify buffer strictly respects capacity and produces non-empty batches."""
    capacity = 50
    buffer = ClassBalancedReservoir(capacity=capacity, seed=123)
    for i in range(300):
        buffer.add({"val": float(i)}, i % 2)

    assert buffer.size() == capacity
    batch = buffer.sample(batch_size=10)
    assert len(batch) == 10
    for x, y in batch:
        assert isinstance(x, dict)
        assert y in (0, 1)


def test_amr_flush_and_refill():
    """Verify AMR core mechanism: flushing class c removes c while preserving other classes, then refills."""
    buffer = ClassBalancedReservoir(capacity=100, seed=42)
    for i in range(50):
        buffer.add({"x": float(i)}, 0)
        buffer.add({"x": float(i)}, 1)

    assert buffer.class_counts()[0] > 0
    assert buffer.class_counts()[1] > 0

    # Flush class 0
    flushed = buffer.flush_class(0)
    assert flushed > 0
    assert buffer.class_counts().get(0, 0) == 0
    # Class 1 must remain intact
    assert buffer.class_counts()[1] > 0

    # Refill class 0 with fresh samples
    for i in range(20):
        buffer.add({"x": float(i + 100)}, 0)
    assert buffer.class_counts()[0] == 20


def test_detector_fires_on_abrupt_shift():
    """Verify drift detector (ADWIN) triggers on an abrupt shift in stream error rate."""
    detector = build_detector("adwin", delta=0.002)
    fired = False

    # Stable good performance: 1000 correct predictions (error = 0)
    for _ in range(1000):
        detector.update(0)

    # Abrupt concept drift: high error rate (error = 1)
    for _ in range(200):
        detector.update(1)
        if detector.drift_detected:
            fired = True
            break

    assert fired, "ADWIN detector must fire on abrupt concept drift shift."


def test_per_class_drift_monitor():
    """Verify PerClassDriftMonitor isolates drift to the specific class that degraded."""
    monitor = PerClassDriftMonitor(detector_name="adwin", delta=0.002)
    drift_seen = False
    drifted_class = None

    # Stable for both classes
    for _ in range(800):
        monitor.update(0, 0)
        monitor.update(1, 0)

    # Degrade only class 1
    for _ in range(200):
        drift_flag, c = monitor.update(1, 1)
        if drift_flag:
            drift_seen = True
            drifted_class = c
            break

    assert drift_seen, "Per-class monitor must detect drift on class 1."
    assert drifted_class == 1, f"Expected drifted class 1, got {drifted_class}"


@pytest.mark.parametrize("learner_cls", [B0NaiveLearner, B1ReplayLearner, B2FullRelearnLearner, B3AMRLearner])
def test_learner_interface(learner_cls):
    """Verify all baselines strictly satisfy the StreamLearner contract."""
    learner = learner_cls()
    x = {"x0": 1.5, "x1": -0.8}
    pred = learner.predict_one(x)
    assert pred in (0, 1)

    info = learner.learn_one(x, y=1)
    assert isinstance(info, dict)
    assert "drift_flag" in info
    assert "concept_id" in info
    assert "buffer_size" in info
    assert learner.memory_bytes() > 0
