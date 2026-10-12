"""
Metrics Sink: Consumes from 'stream.metrics' topic and writes to InfluxDB 2.x and SQLite fallback.
"""
import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import sqlite3
import logging
import argparse
from typing import Dict, Any, Optional
from src.broker.client import BrokerConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MetricsSink] %(message)s")
logger = logging.getLogger("MetricsSink")

SQLITE_PATH = os.getenv("SQLITE_DB_PATH", "results/metrics.db")
INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "ioe413")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "streaming_metrics")


class MetricsStorageSink:
    """Writes streaming evaluation metrics to SQLite and optionally InfluxDB."""

    def __init__(self, db_path: str = SQLITE_PATH, enable_influx: bool = True):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_sqlite()

        self.influx_client = None
        self.write_api = None
        if enable_influx:
            self._init_influx()

    def _init_sqlite(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq INTEGER,
                ts REAL,
                method TEXT,
                stream TEXT,
                y_true INTEGER,
                y_pred INTEGER,
                rolling_acc REAL,
                drift_flag INTEGER,
                concept_id INTEGER,
                recurring_hit INTEGER,
                mem_bytes INTEGER,
                latency_ms REAL,
                buffer_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stream_method ON metrics(stream, method)")
        self.conn.commit()

    def _init_influx(self):
        try:
            from influxdb_client import InfluxDBClient
            client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=2000)
            health = client.health()
            if health.status == "pass":
                self.influx_client = client
                self.write_api = client.write_api()
                logger.info("Connected successfully to InfluxDB 2.x")
            else:
                logger.info(f"InfluxDB health check status '{health.status}', using SQLite storage.")
        except Exception as e:
            logger.info(f"InfluxDB not reachable ({e}). SQLite active at {self.db_path}")

    def write_metric(self, m: Dict[str, Any]):
        """Write single metric event to storage."""
        # Write to SQLite
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (
                    seq, ts, method, stream, y_true, y_pred, rolling_acc,
                    drift_flag, concept_id, recurring_hit, mem_bytes, latency_ms, buffer_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.get("seq", 0),
                m.get("ts", time.time()),
                m.get("method", "Unknown"),
                m.get("stream", "Unknown"),
                int(m.get("y_true", 0)),
                int(m.get("y_pred", 0)),
                float(m.get("rolling_acc", 0.0)),
                1 if m.get("drift_flag") else 0,
                int(m.get("concept_id", 0)),
                1 if m.get("recurring_hit") else 0,
                int(m.get("mem_bytes", 0)),
                float(m.get("latency_ms", 0.0)),
                int(m.get("buffer_size", 0))
            ))
            self.conn.commit()
        except Exception as err:
            logger.error(f"Error inserting metric into SQLite: {err}")

        # Write to InfluxDB if available
        if self.write_api:
            try:
                from influxdb_client import Point
                point = (
                    Point("stream_metrics")
                    .tag("method", str(m.get("method", "Unknown")))
                    .tag("stream", str(m.get("stream", "Unknown")))
                    .field("seq", int(m.get("seq", 0)))
                    .field("rolling_acc", float(m.get("rolling_acc", 0.0)))
                    .field("drift_flag", 1 if m.get("drift_flag") else 0)
                    .field("concept_id", int(m.get("concept_id", 0)))
                    .field("recurring_hit", 1 if m.get("recurring_hit") else 0)
                    .field("mem_bytes", int(m.get("mem_bytes", 0)))
                    .field("latency_ms", float(m.get("latency_ms", 0.0)))
                    .field("buffer_size", int(m.get("buffer_size", 0)))
                )
                self.write_api.write(bucket=INFLUX_BUCKET, record=point)
            except Exception as e:
                logger.debug(f"Influx write error: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
        if self.influx_client:
            self.influx_client.close()


def run_sink(topic: str = "stream.metrics", broker_type: Optional[str] = None):
    logger.info(f"Starting Metrics Sink listening to topic: {topic}")
    consumer = BrokerConsumer(topic=topic, broker_type=broker_type, group_id="metrics_sink")
    sink = MetricsStorageSink()

    count = 0
    try:
        for msg in consumer.iter_messages():
            sink.write_metric(msg)
            count += 1
            if count % 1000 == 0:
                logger.info(f"Persisted {count} metric records to database.")
    except KeyboardInterrupt:
        logger.info("Metrics sink stopped.")
    finally:
        consumer.close()
        sink.close()


def main():
    parser = argparse.ArgumentParser(description="Metrics InfluxDB/SQLite Sink")
    parser.add_argument("--topic", type=str, default=os.getenv("METRICS_TOPIC", "stream.metrics"))
    parser.add_argument("--broker", type=str, default=None)
    args = parser.parse_args()
    run_sink(topic=args.topic, broker_type=args.broker)


if __name__ == "__main__":
    main()
