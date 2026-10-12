"""
Streaming Consumer CLI.
Listens to 'stream.raw', runs the prequential test-then-train loop with the configured
StreamLearner, and publishes performance events to 'stream.metrics' and storage.
"""
import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import argparse
import logging
from typing import Optional

from river import metrics, utils
from src.broker.client import BrokerConsumer, BrokerProducer
from src.consumer.sink_influx import MetricsStorageSink
from src.learners.base import StreamLearner, PlaceholderLearner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Consumer] %(message)s")
logger = logging.getLogger("Consumer")


def create_learner(method_name: str, config: dict = None) -> StreamLearner:
    """Factory creating learners B0, B1, B2, B3, E, or Placeholder."""
    m = method_name.upper()
    cfg = dict(config or {})
    if "total_budget_samples" in cfg and "buffer_size" not in cfg:
        cfg["buffer_size"] = cfg["total_budget_samples"]

    if m in ("PLACEHOLDER", "PH"):
        return PlaceholderLearner()
    elif m == "B0":
        from src.learners.b0_naive import B0NaiveLearner
        return B0NaiveLearner(model_type=cfg.get("model_type", "logistic"), seed=cfg.get("seed", 42))
    elif m == "B1":
        from src.learners.b1_replay import B1ReplayLearner
        return B1ReplayLearner(
            model_type=cfg.get("model_type", "logistic"),
            buffer_size=cfg.get("buffer_size", 500),
            replay_batch=cfg.get("replay_batch", 8),
            seed=cfg.get("seed", 42)
        )
    elif m == "B2":
        from src.learners.b2_full_relearn import B2FullRelearnLearner
        return B2FullRelearnLearner(
            model_type=cfg.get("model_type", "logistic"),
            buffer_size=cfg.get("buffer_size", 500),
            detector_name=cfg.get("detector_name", "adwin"),
            seed=cfg.get("seed", 42)
        )
    elif m == "B3":
        from src.learners.b3_amr import B3AMRLearner
        return B3AMRLearner(
            model_type=cfg.get("model_type", "logistic"),
            buffer_size=cfg.get("buffer_size", 500),
            replay_batch=cfg.get("replay_batch", 8),
            detector_name=cfg.get("detector_name", "adwin"),
            seed=cfg.get("seed", 42)
        )
    elif m in ("E", "CAM"):
        from src.learners.e_cam import CAMLearner
        return CAMLearner(**cfg)
    else:
        logger.warning(f"Unknown method '{method_name}', defaulting to PlaceholderLearner.")
        return PlaceholderLearner()


def run_consumer(stream_name: str, method_name: str, max_samples: Optional[int] = None,
                 raw_topic: str = "stream.raw", metrics_topic: str = "stream.metrics",
                 broker_type: Optional[str] = None, auto_sink: bool = True):
    logger.info(f"Starting Consumer: stream={stream_name}, method={method_name}, raw_topic={raw_topic}")
    
    learner = create_learner(method_name)
    consumer = BrokerConsumer(topic=raw_topic, broker_type=broker_type, group_id=f"consumer_{method_name}")
    producer = BrokerProducer(broker_type=broker_type)
    sink = MetricsStorageSink() if auto_sink else None

    # Prequential evaluation metric: Rolling Accuracy over 500 window
    rolling_acc = utils.Rolling(metrics.Accuracy, window_size=500)
    cumulative_acc = metrics.Accuracy()
    
    total_processed = 0
    drift_count = 0
    recurring_hits = 0
    start_time = time.time()

    try:
        for raw_msg in consumer.iter_messages(timeout=2.0):
            seq = raw_msg.get("seq", total_processed + 1)
            x = raw_msg.get("x", {})
            y = int(raw_msg.get("y", 0))
            
            # CRITICAL REQUIREMENT: Never inspect or pass true_concept to learner
            t0 = time.perf_counter()
            y_pred = learner.predict_one(x)
            
            # Prequential evaluation: update accuracy before training
            rolling_acc.update(y, y_pred)
            cumulative_acc.update(y, y_pred)
            
            # Train step
            info = learner.learn_one(x, y)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            total_processed += 1
            if info.get("drift_flag"):
                drift_count += 1
            if info.get("recurring_hit"):
                recurring_hits += 1

            metric_msg = {
                "seq": seq,
                "ts": time.time(),
                "method": method_name,
                "stream": stream_name,
                "y_true": y,
                "y_pred": int(y_pred),
                "rolling_acc": float(rolling_acc.get()),
                "cumulative_acc": float(cumulative_acc.get()),
                "drift_flag": bool(info.get("drift_flag", False)),
                "concept_id": int(info.get("concept_id", 0)),
                "recurring_hit": bool(info.get("recurring_hit", False)),
                "mem_bytes": int(learner.memory_bytes()),
                "latency_ms": float(latency_ms),
                "buffer_size": int(info.get("buffer_size", 0))
            }

            producer.send(metrics_topic, metric_msg)
            if sink:
                sink.write_metric(metric_msg)

            if total_processed % 1000 == 0:
                elapsed = time.time() - start_time
                fps = total_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Processed {total_processed} instances ({fps:.1f} inst/s) | "
                    f"Rolling Acc: {rolling_acc.get():.3f} | Cumulative Acc: {cumulative_acc.get():.3f} | "
                    f"Drifts: {drift_count} | Hits: {recurring_hits} | Latency: {latency_ms:.2f}ms"
                )

            if max_samples and total_processed >= max_samples:
                break

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        consumer.close()
        if sink:
            sink.close()
        elapsed = time.time() - start_time
        logger.info(f"Consumer finished: {total_processed} processed in {elapsed:.2f}s.")


def main():
    parser = argparse.ArgumentParser(description="Streaming Consumer & Prequential Evaluator")
    parser.add_argument("--stream", type=str, default=os.getenv("DEFAULT_STREAM", "SEA_recurring"))
    parser.add_argument("--method", type=str, default=os.getenv("DEFAULT_METHOD", "PLACEHOLDER"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--raw-topic", type=str, default=os.getenv("RAW_TOPIC", "stream.raw"))
    parser.add_argument("--metrics-topic", type=str, default=os.getenv("METRICS_TOPIC", "stream.metrics"))
    parser.add_argument("--broker", type=str, default=None)
    parser.add_argument("--no-auto-sink", action="store_true", default=False)

    args = parser.parse_args()
    run_consumer(
        stream_name=args.stream,
        method_name=args.method,
        max_samples=args.max_samples,
        raw_topic=args.raw_topic,
        metrics_topic=args.metrics_topic,
        broker_type=args.broker,
        auto_sink=not args.no_auto_sink
    )


if __name__ == "__main__":
    main()
