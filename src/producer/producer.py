"""
Stream Producer CLI.
Streams synthetic or real non-stationary data into Kafka / MQTT topic 'stream.raw'.
"""
import sys
import os
import pathlib

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import argparse
import logging
from src.producer.streams import get_stream_generator
from src.broker.client import BrokerProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Producer] %(message)s")
logger = logging.getLogger("Producer")


def run_producer(stream_name: str, rate: float, seed: int, loop: bool, max_samples: int = None, topic: str = "stream.raw", broker_type: str = None):
    logger.info(f"Starting Stream Producer: stream={stream_name}, rate={rate} msgs/s, seed={seed}, loop={loop}, topic={topic}")
    producer = BrokerProducer(broker_type=broker_type)
    
    interval = (1.0 / rate) if rate > 0 else 0.0
    iteration = 0
    total_published = 0
    start_time = time.time()

    try:
        while True:
            iteration += 1
            gen = get_stream_generator(stream_name, seed=seed + iteration - 1, max_instances=max_samples)
            for msg in gen:
                producer.send(topic, msg)
                total_published += 1

                if total_published % 1000 == 0:
                    elapsed = time.time() - start_time
                    fps = total_published / elapsed if elapsed > 0 else 0
                    logger.info(f"Published {total_published} messages ({fps:.1f} msgs/s) [latest seq: {msg['seq']}, concept: {msg.get('true_concept')}]")

                if max_samples and total_published >= max_samples:
                    break

                if interval > 0:
                    time.sleep(interval)

            if not loop or (max_samples and total_published >= max_samples):
                break

            logger.info("Loop enabled: restarting stream cycle...")

    except KeyboardInterrupt:
        logger.info("Producer interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        fps = total_published / elapsed if elapsed > 0 else 0
        logger.info(f"Producer finished: Published {total_published} messages in {elapsed:.2f}s ({fps:.1f} msgs/s)")


def main():
    parser = argparse.ArgumentParser(description="Stream Producer for Continual Learning Benchmarks")
    parser.add_argument("--stream", type=str, default=os.getenv("DEFAULT_STREAM", "SEA_recurring"),
                        help="Stream name: SEA_recurring, SEA_abrupt, STAGGER_recurring, SINE_gradual, Elec2")
    parser.add_argument("--rate", type=float, default=float(os.getenv("DEFAULT_RATE", "100")),
                        help="Publish rate in msgs/sec (0 for max speed)")
    parser.add_argument("--seed", type=int, default=int(os.getenv("DEFAULT_SEED", "42")),
                        help="Random seed for reproducibility")
    parser.add_argument("--loop", action="store_true", default=False,
                        help="Loop stream indefinitely")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to publish")
    parser.add_argument("--topic", type=str, default=os.getenv("RAW_TOPIC", "stream.raw"),
                        help="Destination topic")
    parser.add_argument("--broker", type=str, default=None,
                        help="Broker type: kafka or mqtt")

    args = parser.parse_args()
    run_producer(
        stream_name=args.stream,
        rate=args.rate,
        seed=args.seed,
        loop=args.loop,
        max_samples=args.max_samples,
        topic=args.topic,
        broker_type=args.broker
    )


if __name__ == "__main__":
    main()
