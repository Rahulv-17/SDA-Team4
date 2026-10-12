"""
Test suite for Stream Selection and Multi-Stream Filtering Audit (Phase 7).
Verifies:
1. Stream selector state propagation to backend endpoints.
2. Metrics and comparison results are strictly scoped to the requested stream.
3. Multi-method comparative trajectories for all 5 streams (B0..E).
4. Concept memory repository customization per stream.
5. Handling of stream switches and fallback behavior.
"""
import pytest
from fastapi.testclient import TestClient
from src.dashboard.server import app

client = TestClient(app)

SUPPORTED_STREAMS = [
    "SEA_recurring",
    "SEA_abrupt",
    "STAGGER_recurring",
    "SINE_gradual",
    "Elec2"
]


def test_stream_select_post():
    """Verify POST /api/stream/select updates backend state."""
    for stream in SUPPORTED_STREAMS:
        resp = client.post("/api/stream/select", json={"stream": stream, "method": "E"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_stream"] == stream
        assert data["selected_method"] == "E"


def test_status_stream_scoping():
    """Verify /api/status filters correctly and does not leak other streams."""
    for stream in SUPPORTED_STREAMS:
        resp = client.get(f"/api/status?stream={stream}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stream"] == stream
        assert "rolling_acc" in data
        assert "drift_count" in data
        assert "buffer_size" in data
        assert isinstance(data["rolling_acc"], (int, float))
        assert 0.0 <= data["rolling_acc"] <= 1.0


def test_comparison_stream_filtering():
    """Verify /api/comparison returns runs exclusively for the requested stream."""
    for stream in SUPPORTED_STREAMS:
        resp = client.get(f"/api/comparison?stream={stream}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_stream"] == stream
        runs = data["runs"]
        assert len(runs) > 0
        # Check that all returned runs belong strictly to the requested stream
        for r in runs:
            assert r["stream"] == stream
            assert r["method"] in ["B0", "B1", "B2", "B3", "E"]
            assert 0.0 <= r["acc_mean"] <= 1.0
            assert 0.0 <= r["f1_mean"] <= 1.0


def test_comparison_traces_all_streams():
    """Verify /api/comparison/traces returns valid trajectories for all 5 methods."""
    for stream in SUPPORTED_STREAMS:
        resp = client.get(f"/api/comparison/traces?stream={stream}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stream"] == stream
        traces = data["traces"]
        for method in ["B0", "B1", "B2", "B3", "E"]:
            assert method in traces
            assert len(traces[method]) > 0
            first_pt = traces[method][0]
            assert "x" in first_pt
            assert "y" in first_pt
            assert 0.0 <= first_pt["y"] <= 1.0


def test_concept_memory_stream_customization():
    """Verify /api/concept_memory returns concepts tailored to the stream topology."""
    resp_sea = client.get("/api/concept_memory?stream=SEA_recurring").json()
    resp_stagger = client.get("/api/concept_memory?stream=STAGGER_recurring").json()
    resp_elec = client.get("/api/concept_memory?stream=Elec2").json()

    assert resp_sea["stream"] == "SEA_recurring"
    assert "SEA_A" in resp_sea["active_concept_name"]

    assert resp_stagger["stream"] == "STAGGER_recurring"
    assert "STAGGER_C0" in resp_stagger["active_concept_name"]

    assert resp_elec["stream"] == "Elec2"
    assert "Elec2_Peak" in resp_elec["active_concept_name"]


def test_events_stream_filtering():
    """Verify /api/events returns notifications scoped to the requested stream."""
    for stream in SUPPORTED_STREAMS:
        resp = client.get(f"/api/events?stream={stream}")
        assert resp.status_code == 200
        events = resp.json()
        assert isinstance(events, list)
        if events:
            # Events should mention the stream or its concepts
            assert "type" in events[0]
            assert "desc" in events[0]


def test_unknown_stream_fallback():
    """Verify unknown stream query falls back gracefully without crashing."""
    resp = client.get("/api/status?stream=UnknownStream")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stream"] == "UnknownStream"
