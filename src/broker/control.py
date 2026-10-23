"""
Dynamic Stream Pipeline Control Manager.
Provides IPC state coordination and MQTT/Kafka control channel for real-time
stream switching, pause/resume, and stream resetting between Dashboard, Producer, and Consumer.
"""
import os
import json
import time
import pathlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("StreamControl")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONTROL_FILE = RESULTS_DIR / "stream_control.json"
CONTROL_TOPIC = os.getenv("CONTROL_TOPIC", "stream.control")

DEFAULT_STATE = {
    "stream": "SEA_recurring",
    "method": "E",
    "is_paused": False,
    "reset_counter": 0,
    "updated_at": time.time(),
}


def get_control_state() -> Dict[str, Any]:
    """Read the current control state from disk safely with retries for Windows file locks."""
    if not CONTROL_FILE.exists():
        return dict(DEFAULT_STATE)
    
    for _ in range(5):
        try:
            with open(CONTROL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_STATE, **data}
        except (json.JSONDecodeError, PermissionError, IOError):
            time.sleep(0.02)
    return dict(DEFAULT_STATE)


def is_broker_alive(port: int = 1883, host: str = "127.0.0.1") -> bool:
    """Quick non-blocking check if broker port is open."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.05)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def set_control_state(
    stream: Optional[str] = None,
    method: Optional[str] = None,
    is_paused: Optional[bool] = None,
    reset: bool = False,
    broadcast_broker: bool = True
) -> Dict[str, Any]:
    """Update control state and broadcast to MQTT control topic and local IPC file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = get_control_state()

    if stream is not None:
        state["stream"] = stream
    if method is not None:
        state["method"] = method
    if is_paused is not None:
        state["is_paused"] = is_paused
    if reset:
        state["reset_counter"] = state.get("reset_counter", 0) + 1
    state["updated_at"] = time.time()

    # Write atomically
    temp_file = CONTROL_FILE.with_suffix(".tmp")
    for _ in range(5):
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            if temp_file.exists():
                temp_file.replace(CONTROL_FILE)
            break
        except (PermissionError, IOError):
            time.sleep(0.02)

    # Broadcast via message broker topic if requested and broker is active
    if broadcast_broker and is_broker_alive():
        try:
            from src.broker.client import BrokerProducer
            producer = BrokerProducer()
            producer.send(CONTROL_TOPIC, state)
            producer.close()
        except Exception as e:
            logger.debug(f"Broker broadcast omitted: {e}")

    return state

