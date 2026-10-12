"""
Lightweight MQTT Broker runner using amqtt for environments without Docker/Kafka.
Binds to 127.0.0.1:1883 with anonymous access enabled.
"""
import sys
import logging
import asyncio
from amqtt.broker import Broker

logger = logging.getLogger("BrokerServer")

BROKER_CONFIG = {
    "listeners": {
        "default": {
            "type": "tcp",
            "bind": "127.0.0.1:1883",
            "max_connections": 1000,
        }
    },
    "sys_interval": 10,
    "auth": {
        "allow-anonymous": True
    },
    "topic-check": {
        "enabled": False
    }
}


async def start_broker():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    broker = Broker(BROKER_CONFIG)
    logger.info("Starting local MQTT Broker on 127.0.0.1:1883...")
    await broker.start()
    logger.info("MQTT Broker is active and listening for streaming connections.")
    try:
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down broker...")
        await broker.shutdown()
        logger.info("Broker stopped.")


def main():
    try:
        asyncio.run(start_broker())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
