"""
Unified One-Command Launcher for Concept Drift Lab.
Launches:
  1. Message Broker (MQTT / Kafka fallback)
  2. Stitch Research Dashboard (FastAPI on http://localhost:8501)
  3. River Streaming Consumer (evaluates B0..B3 or E)
  4. Stream Producer (synthetic or real stream generator)
  5. Opens browser to http://localhost:8501

Usage:
  python run_demo.py
  python run_demo.py --stream SEA_recurring --method E --rate 100
"""

import sys
import os
import time
import socket
import signal
import argparse
import subprocess
import webbrowser
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main():
    parser = argparse.ArgumentParser(description="Concept Drift Lab - Single-Command Pipeline Launcher")
    parser.add_argument("--stream", type=str, default="SEA_recurring",
                        choices=["SEA_abrupt", "SEA_recurring", "STAGGER_recurring", "SINE_gradual", "Elec2"],
                        help="Data stream to generate")
    parser.add_argument("--method", type=str, default="E",
                        choices=["B0", "B1", "B2", "B3", "E"],
                        help="Online continual learner algorithm")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Emission rate (instances/second)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible drift")
    parser.add_argument("--port", type=int, default=8501,
                        help="Dashboard HTTP port")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open browser")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Loop stream indefinitely (default: True for continuous live streaming)")
    parser.add_argument("--no-loop", action="store_false", dest="loop",
                        help="Run stream once through then stop")
    parser.add_argument("--reset", action="store_true", default=False,
                        help="Reset live metrics database for fresh start")
    args = parser.parse_args()

    processes = []

    def cleanup(signum=None, frame=None):
        print("\n[Shutting down] Terminating all streaming services gracefully...")
        for p in reversed(processes):
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[Shutdown complete] All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 72)
    print("  [CONCEPT DRIFT LAB] ONE-COMMAND STREAMING PIPELINE LAUNCHER")
    print("=" * 72)

    # Reset live metrics in SQLite if requested
    if args.reset:
        sqlite_db = PROJECT_ROOT / "results" / "metrics.db"
        if sqlite_db.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(sqlite_db))
                conn.execute("DELETE FROM metrics")
                conn.commit()
                conn.close()
                print("[*] Reset live metrics in results/metrics.db for fresh streaming run.")
            except Exception:
                pass

    # 1. Start Broker if not already running on port 1883 or 9092
    broker_running = is_port_in_use(1883) or is_port_in_use(9092)
    if not broker_running:
        print("[1/4] Starting local message broker (port 1883)...")
        p_broker = subprocess.Popen(
            [PYTHON_EXE, "src/broker/run_broker.py"],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(p_broker)
        time.sleep(1.0)
    else:
        print("[1/4] Message broker already running on port 1883/9092. Reusing.")

    # 2. Start Stitch Dashboard server if port not in use
    dash_running = is_port_in_use(args.port)
    if not dash_running:
        print(f"[2/4] Starting Google Stitch Dashboard (http://localhost:{args.port})...")
        p_dash = subprocess.Popen(
            [PYTHON_EXE, "src/dashboard/server.py", "--port", str(args.port)],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(p_dash)
        time.sleep(1.5)
    else:
        print(f"[2/4] Dashboard already active on http://localhost:{args.port}. Reusing.")

    # 3. Start River Consumer
    print(f"[3/4] Starting River Streaming Consumer (Method: {args.method}, Stream: {args.stream})...")
    p_consumer = subprocess.Popen(
        [PYTHON_EXE, "src/consumer/consumer.py", "--stream", args.stream, "--method", args.method],
        cwd=str(PROJECT_ROOT)
    )
    processes.append(p_consumer)
    time.sleep(0.8)

    # 4. Start Stream Producer
    loop_tag = " (continuous loop)" if args.loop else " (single pass)"
    print(f"[4/4] Starting Stream Producer ({args.stream} @ {args.rate} inst/s, seed={args.seed}){loop_tag}...")
    producer_cmd = [PYTHON_EXE, "src/producer/producer.py", "--stream", args.stream, "--rate", str(args.rate), "--seed", str(args.seed)]
    if args.loop:
        producer_cmd.append("--loop")
    p_producer = subprocess.Popen(
        producer_cmd,
        cwd=str(PROJECT_ROOT)
    )
    processes.append(p_producer)

    dash_url = f"http://localhost:{args.port}"
    print("=" * 72)
    print("  [ONLINE] PIPELINE IS ACTIVELY STREAMING:")
    print(f"  * UI Dashboard: {dash_url}")
    print(f"  * Stream:       {args.stream} (Emission: {args.rate} inst/s, loop={args.loop})")
    print(f"  * Model:        {args.method} (Concept-Aware Memory / River)")
    print(f"  * Broker Flow:  stream.raw -> Consumer -> stream.metrics -> Storage")
    print("=" * 72)
    print("  Press Ctrl+C at any time to stop the entire pipeline.\n")

    if not args.no_browser:
        try:
            webbrowser.open(dash_url)
        except Exception:
            pass

    # Keep alive while monitoring child processes
    try:
        while True:
            # Check if producer has finished in non-loop mode
            if not args.loop and p_producer.poll() is not None:
                time.sleep(2.0)
                print("\n" + "=" * 72)
                print(f"  [STREAM COMPLETE] Finished streaming all instances for {args.stream}.")
                print(f"  Dashboard remains active at {dash_url}.")
                print("  Press Ctrl+C to terminate all services.")
                print("=" * 72)
                while True:
                    time.sleep(1.0)
            time.sleep(1.0)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
