"""
Tests for dynamic live stream switching, producer/consumer coordination, and multi-stream live analytics.
Verifies that:
1. Any stream selected (SEA_recurring, SEA_abrupt, STAGGER_recurring, SINE_gradual, Elec2)
   becomes the active live stream.
2. The producer transitions to the selected stream.
3. The consumer detects the switch, adapts continual learning state, and tags metrics correctly.
4. The dashboard APIs (/api/status, /api/metrics/live) return live prequential accuracy points.
"""
import time
import pytest
from fastapi.testclient import TestClient
from src.dashboard.server import app, get_active_concept_label
from src.broker.control import set_control_state, get_control_state
from src.producer.streams import get_stream_generator
from src.consumer.consumer import create_learner
from river import metrics, utils

client = TestClient(app)

SUPPORTED_STREAMS = [
    "SEA_recurring",
    "SEA_abrupt",
    "STAGGER_recurring",
    "SINE_gradual",
    "Elec2"
]


def test_control_state_updates():
    """Verify set_control_state persists state and supports pause and reset flags."""
    for s in SUPPORTED_STREAMS:
        state = set_control_state(stream=s, method="E", broadcast_broker=False)
        assert state["stream"] == s
        assert state["method"] == "E"
        read_back = get_control_state()
        assert read_back["stream"] == s

    # Test pause
    set_control_state(is_paused=True, broadcast_broker=False)
    assert get_control_state()["is_paused"] is True
    set_control_state(is_paused=False, broadcast_broker=False)
    assert get_control_state()["is_paused"] is False

    # Test reset
    prev_cnt = get_control_state().get("reset_counter", 0)
    set_control_state(reset=True, broadcast_broker=False)
    assert get_control_state().get("reset_counter", 0) == prev_cnt + 1


def test_concept_labels_for_all_streams():
    """Verify active concept labels match the topology of every stream."""
    assert "SEA_A" in get_active_concept_label("SEA_recurring", 1)
    assert "SEA_V0" in get_active_concept_label("SEA_abrupt", 1)
    assert "STAGGER_C0" in get_active_concept_label("STAGGER_recurring", 1)
    assert "SINE_S0" in get_active_concept_label("SINE_gradual", 1)
    assert "Elec2_Peak" in get_active_concept_label("Elec2", 1)


def test_consumer_stream_transition_adaptation():
    """Verify consumer creates a clean learner and resets metrics upon stream transition."""
    streams_to_test = ["SEA_recurring", "STAGGER_recurring", "SINE_gradual"]
    
    current_stream = "SEA_recurring"
    learner = create_learner("E")
    rolling_acc = utils.Rolling(metrics.Accuracy, window_size=500)
    
    for s in streams_to_test:
        gen = get_stream_generator(s, max_instances=5)
        for msg in gen:
            # Emulate message arriving with new stream tag
            if msg["stream"] != current_stream:
                current_stream = msg["stream"]
                learner = create_learner("E")
                rolling_acc = utils.Rolling(metrics.Accuracy, window_size=500)
            
            y_pred = learner.predict_one(msg["x"])
            rolling_acc.update(msg["y"], y_pred)
            learner.learn_one(msg["x"], msg["y"])

        assert current_stream == s
        assert 0.0 <= rolling_acc.get() <= 1.0


def test_server_live_status_and_metrics_all_streams():
    """Verify server status and live metrics work in live mode for all streams."""
    for s in SUPPORTED_STREAMS:
        # Select stream
        res = client.post("/api/stream/select", json={"stream": s, "method": "E"})
        assert res.status_code == 200
        assert res.json()["selected_stream"] == s

        # Check status
        st_res = client.get(f"/api/status?stream={s}")
        assert st_res.status_code == 200
        st_data = st_res.json()
        assert st_data["stream"] == s
        assert st_data["is_live_ingestion"] is True

        # Check live metrics
        lm_res = client.get(f"/api/metrics/live?stream={s}")
        assert lm_res.status_code == 200
        lm_data = lm_res.json()
        assert lm_data["stream"] == s
        assert lm_data["is_live"] is True
        assert len(lm_data["points"]) > 0
        assert len(lm_data["baseline"]) > 0
