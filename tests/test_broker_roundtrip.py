"""
Test verification gate for Phase 0: Test message round-trips through the message broker.
"""
import time
import pytest
import threading
from src.broker.client import BrokerProducer, BrokerConsumer


def test_broker_roundtrip():
    test_topic = "test.roundtrip"
    received_messages = []

    consumer = BrokerConsumer(topic=test_topic)
    time.sleep(0.5)  # Allow MQTT / Kafka consumer subscription to settle

    def consume_worker():
        for msg in consumer.iter_messages(timeout=0.5):
            received_messages.append(msg)
            if len(received_messages) >= 3:
                break

    thread = threading.Thread(target=consume_worker, daemon=True)
    thread.start()

    producer = BrokerProducer()
    test_data = [
        {"seq": 1, "test": "hello_stream", "val": 42.0},
        {"seq": 2, "test": "concept_drift", "val": 84.0},
        {"seq": 3, "test": "memory_realign", "val": 126.0},
    ]

    for item in test_data:
        producer.send(test_topic, item)
        time.sleep(0.05)

    producer.flush()
    thread.join(timeout=3.0)
    producer.close()
    consumer.close()

    assert len(received_messages) == 3, f"Expected 3 messages, received {len(received_messages)}"
    assert received_messages[0]["test"] == "hello_stream"
    assert received_messages[1]["val"] == 84.0
    assert received_messages[2]["seq"] == 3
