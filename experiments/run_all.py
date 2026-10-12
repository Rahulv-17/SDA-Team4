"""
Comprehensive Experiments Runner.
Executes benchmarks across all streams, methods (B0, B1, B2, B3, E), and seeds (5+ seeds).
Saves results to results/*.csv and environment metadata to results/metadata.json.
"""
import sys
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import yaml
import logging
import platform
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import river
import sklearn
from river import metrics, utils
from src.producer.streams import get_stream_generator
from src.consumer.consumer import create_learner
from src.metrics.eval import (
    calculate_recovery_time,
    calculate_recurrence_forgetting,
    calculate_post_drift_acc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Experiments] %(message)s")
logger = logging.getLogger("Experiments")

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_stream_drift_points(stream_name: str, max_samples: int = 12000) -> List[int]:
    """Known drift points for synthetic streams."""
    name = stream_name.upper()
    if "SEA_RECURRING" in name:
        # 6 segments of 2500: drifts at 2500, 5000, 7500, 10000, 12500
        return [2500, 5000, 7500, 10000, 12500]
    elif "SEA_ABRUPT" in name:
        return [2500, 5000, 7500, 10000]
    elif "STAGGER" in name:
        return [2000, 4000, 6000, 8000, 10000]
    elif "SINE" in name:
        return [6000]
    else:
        return []


def run_single_experiment(stream_name: str, method_name: str, seed: int,
                           max_samples: int = 10000, learner_config: dict = None,
                           save_trace: bool = False) -> Dict[str, Any]:
    """Run a single streaming trial and collect comprehensive metrics."""
    cfg = dict(learner_config or {})
    cfg["seed"] = seed
    learner = create_learner(method_name, cfg)
    gen = get_stream_generator(stream_name, seed=seed, max_instances=max_samples)

    acc = metrics.Accuracy()
    f1 = metrics.MacroF1()
    precision = metrics.MacroPrecision()
    recall = metrics.MacroRecall()
    rolling_acc = utils.Rolling(metrics.Accuracy, window_size=200)

    acc_history = []
    trace_seqs = []
    latencies = []
    drift_events = []
    recurring_hits = []

    t_start = time.perf_counter()
    for item in gen:
        seq = item["seq"]
        x = item["x"]
        y = item["y"]

        # Prequential test-then-train
        t0 = time.perf_counter()
        y_pred = learner.predict_one(x)
        t_pred = time.perf_counter()

        acc.update(y, y_pred)
        f1.update(y, y_pred)
        precision.update(y, y_pred)
        recall.update(y, y_pred)
        rolling_acc.update(y, y_pred)

        info = learner.learn_one(x, y)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        cur_roll_acc = rolling_acc.get()
        acc_history.append(cur_roll_acc)
        if save_trace:
            trace_seqs.append((seq, cur_roll_acc, info.get("drift_flag", False), info.get("recurring_hit", False)))

        if info.get("drift_flag"):
            drift_events.append(seq)
        if info.get("recurring_hit"):
            recurring_hits.append(seq)

        if max_samples and seq >= max_samples:
            break

    total_time = time.perf_counter() - t_start
    drift_points = get_stream_drift_points(stream_name, max_samples)

    # Compute Recovery Times across known drift points
    rec_times = []
    for dp in drift_points:
        rt = calculate_recovery_time(acc_history, dp, window=300)
        if rt is not None:
            rec_times.append(rt)
    mean_rec_time = float(np.mean(rec_times)) if rec_times else 0.0

    # Compute Forgetting on Recurrence
    # On SEA_recurring, Concept A first ends at 2500, recurs at 5000 and 10000
    forgetting_5k = calculate_recurrence_forgetting(acc_history, first_end=2500, rec_start=5000, eval_window=300)
    forgetting_10k = calculate_recurrence_forgetting(acc_history, first_end=2500, rec_start=10000, eval_window=300)
    avg_forgetting = (forgetting_5k + forgetting_10k) / 2.0 if "RECURRING" in stream_name.upper() else 0.0

    post_drift_acc = calculate_post_drift_acc(acc_history, drift_points, post_window=500)

    result = {
        "stream": stream_name,
        "method": method_name,
        "seed": seed,
        "instances": len(acc_history),
        "accuracy": float(acc.get()),
        "f1_macro": float(f1.get()),
        "precision": float(precision.get()),
        "recall": float(recall.get()),
        "recovery_time": mean_rec_time,
        "forgetting": float(avg_forgetting),
        "post_drift_acc": post_drift_acc,
        "drift_count": len(drift_events),
        "recurring_hits": len(recurring_hits),
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": len(latencies) / total_time if total_time > 0 else 0.0,
        "memory_bytes": int(learner.memory_bytes()),
    }

    if save_trace:
        result["trace"] = trace_seqs

    return result


def save_metadata():
    """Record system environment and tool versions."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "river_version": river.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved environment metadata to results/metadata.json")


def run_main_comparison():
    logger.info(">>> Starting Experiment 1: Main Comparison across 5 seeds...")
    streams = ["SEA_recurring", "SEA_abrupt", "STAGGER_recurring", "Elec2"]
    methods = ["B0", "B1", "B2", "B3", "E"]
    seeds = [42, 43, 44, 45, 46]

    records = []
    traces = {}

    for stream in streams:
        for method in methods:
            for seed in seeds:
                logger.info(f"Running Main: Stream={stream}, Method={method}, Seed={seed}")
                save_trace = (stream == "SEA_recurring" and seed == 42)
                res = run_single_experiment(
                    stream_name=stream,
                    method_name=method,
                    seed=seed,
                    max_samples=10000 if stream != "Elec2" else 15000,
                    save_trace=save_trace
                )
                if save_trace and "trace" in res:
                    traces[method] = res.pop("trace")
                records.append(res)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "main_comparison.csv", index=False)
    logger.info(f"Saved {len(df)} records to results/main_comparison.csv")

    # Save detailed traces for Figure 1
    if traces:
        rows = []
        for method, trace_data in traces.items():
            for seq, r_acc, d_flag, r_hit in trace_data:
                rows.append({"seq": seq, "method": method, "rolling_acc": r_acc, "drift_flag": d_flag, "recurring_hit": r_hit})
        df_trace = pd.DataFrame(rows)
        df_trace.to_csv(RESULTS_DIR / "traces_sea_recurring.csv", index=False)
        logger.info("Saved trace history to results/traces_sea_recurring.csv")


def run_budget_sweep():
    logger.info(">>> Starting Experiment 2: Memory Budget Sweep...")
    budgets = [100, 250, 500, 1000]
    seeds = [42, 43, 44]
    methods = ["B3", "E"]
    records = []

    for budget in budgets:
        for method in methods:
            for seed in seeds:
                logger.info(f"Budget Sweep: Budget={budget}, Method={method}, Seed={seed}")
                cfg = {"buffer_size": budget, "total_budget_samples": budget}
                res = run_single_experiment("SEA_recurring", method, seed, max_samples=8000, learner_config=cfg)
                res["budget_samples"] = budget
                records.append(res)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "budget_sweep.csv", index=False)
    logger.info(f"Saved {len(df)} records to results/budget_sweep.csv")


def run_ablations():
    logger.info(">>> Starting Experiments 3-5: Ablation Studies...")
    seeds = [42, 43, 44]

    # 3. Detector Ablation: ADWIN vs KSWIN vs DDM
    det_records = []
    for det in ["adwin", "kswin", "ddm"]:
        for m in ["B3", "E"]:
            for seed in seeds:
                cfg = {"detector_name": det}
                res = run_single_experiment("SEA_recurring", m, seed, max_samples=8000, learner_config=cfg)
                res["detector"] = det
                det_records.append(res)
    pd.DataFrame(det_records).to_csv(RESULTS_DIR / "detector_ablation.csv", index=False)

    # 4. Matching Ablation: error-based vs fingerprint-based
    match_records = []
    for mode in ["error", "fingerprint"]:
        for seed in seeds:
            cfg = {"matching_mode": mode}
            res = run_single_experiment("SEA_recurring", "E", seed, max_samples=8000, learner_config=cfg)
            res["matching_mode"] = mode
            match_records.append(res)
    pd.DataFrame(match_records).to_csv(RESULTS_DIR / "matching_ablation.csv", index=False)

    # 5. Selection Strategy Ablation: reservoir vs herding vs diverse
    sel_records = []
    for strat in ["reservoir", "herding", "diverse"]:
        for seed in seeds:
            cfg = {"selection_strategy": strat}
            res = run_single_experiment("SEA_recurring", "E", seed, max_samples=8000, learner_config=cfg)
            res["selection_strategy"] = strat
            sel_records.append(res)
    pd.DataFrame(sel_records).to_csv(RESULTS_DIR / "selection_ablation.csv", index=False)

    logger.info("Saved all ablation CSVs to results/")


def main():
    logger.info("Starting complete experimental evaluation suite...")
    t0 = time.time()
    save_metadata()
    if not (RESULTS_DIR / "main_comparison.csv").exists():
        run_main_comparison()
    else:
        logger.info("results/main_comparison.csv already generated (100 runs). Skipping main comparison.")
    run_budget_sweep()
    run_ablations()
    elapsed = time.time() - t0
    logger.info(f"All experiments successfully completed in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()
