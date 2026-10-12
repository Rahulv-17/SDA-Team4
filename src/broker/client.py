"""
Unified messaging broker client supporting Apache Kafka and MQTT (Mosquitto/amqtt).
Ensures data strictly flows through a real streaming message broker.
"""
import os
import json
import time
import queue
import logging
from typing import Generator, Dict, Any, Optional

logger = logging.getLogger("BrokerClient")

# Defaults from environment or fallback
DEFAULT_BROKER_TYPE = os.getenv("BROKER_TYPE", "mqtt").lower()
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
MQTT_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))


class BrokerProducer:
    """Publishes streaming JSON messages to Kafka or MQTT."""

    def __init__(self, broker_type: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None):
        self.broker_type = (broker_type or DEFAULT_BROKER_TYPE).lower()
        self.host = host or (KAFKA_SERVERS if self.broker_type == "kafka" else MQTT_HOST)
        self.port = port or MQTT_PORT
        self.client = None

        if self.broker_type == "kafka":
            try:
                from kafka import KafkaProducer
                self.client = KafkaProducer(
                    bootstrap_servers=self.host,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=3
                )
                logger.info(f"Connected KafkaProducer to {self.host}")
            except Exception as e:
                logger.warning(f"Failed to connect Kafka producer ({e}), falling back to MQTT.")
                self.broker_type = "mqtt"

        if self.broker_type == "mqtt":
            import paho.mqtt.client as mqtt
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            else:
                self.client = mqtt.Client()
            self.client.connect(MQTT_HOST, self.port, 60)
            self.client.loop_start()
            logger.info(f"Connected MQTT Producer to {MQTT_HOST}:{self.port}")

    def send(self, topic: str, message: Dict[str, Any]):
        """Publish a dict as a JSON message to the specified topic."""
        if self.broker_type == "kafka":
            self.client.send(topic, message)
        else:
            payload = json.dumps(message)
            self.client.publish(topic, payload, qos=1)

    def flush(self):
        if self.broker_type == "kafka":
            self.client.flush()

    def close(self):
        if self.broker_type == "kafka":
            self.client.flush()
            self.client.close()
        elif self.broker_type == "mqtt" and self.client:
            self.client.loop_stop()
            self.client.disconnect()


class BrokerConsumer:
    """Subscribes to topics and yields deserialized JSON messages."""

    def __init__(self, topic: str, broker_type: Optional[str] = None, group_id: str = "stream_consumer"):
        self.topic = topic
        self.broker_type = (broker_type or DEFAULT_BROKER_TYPE).lower()
        self.group_id = group_id
        self.msg_queue = queue.Queue(maxsize=10000)
        self.running = True
        self.client = None

        if self.broker_type == "kafka":
            try:
                from kafka import KafkaConsumer
                self.client = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=KAFKA_SERVERS,
                    group_id=self.group_id,
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
                )
                logger.info(f"Connected KafkaConsumer to {KAFKA_SERVERS} on topic {self.topic}")
            except Exception as e:
                logger.warning(f"Failed to connect Kafka consumer ({e}), falling back to MQTT.")
                self.broker_type = "mqtt"

        if self.broker_type == "mqtt":
            import paho.mqtt.client as mqtt
            
            def on_message(client, userdata, msg):
                try:
                    payload = json.loads(msg.payload.decode("utf-8"))
                    self.msg_queue.put(payload, timeout=0.5)
                except Exception as err:
                    logger.error(f"Error parsing MQTT message: {err}")

            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            else:
                self.client = mqtt.Client()
            self.client.on_message = on_message
            self.client.connect(MQTT_HOST, MQTT_PORT, 60)
            self.client.subscribe(self.topic, qos=1)
            self.client.loop_start()
            logger.info(f"Connected MQTT Consumer to {MQTT_HOST}:{MQTT_PORT} on topic {self.topic}")

    def iter_messages(self, timeout: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """Yields messages as they arrive."""
        if self.broker_type == "kafka":
            for msg in self.client:
                if not self.running:
                    break
                yield msg.value
        else:
            while self.running:
                try:
                    msg = self.msg_queue.get(timeout=timeout)
                    yield msg
                except queue.Empty:
                    continue

    def close(self):
        self.running = False
        if self.broker_type == "kafka" and self.client:
            self.client.close()
        elif self.broker_type == "mqtt" and self.client:
            self.client.loop_stop()
            self.client.disconnect()
