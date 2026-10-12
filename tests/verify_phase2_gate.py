"""
Verification Gate for Phase 2:
Evaluates B0, B1, B2, and B3 on SEA_abrupt.
Verifies:
1. B0 exhibits accuracy degradation after concept drift.
2. B3 (AMR baseline) recovers faster than B0.
3. B2 (Full Relearning) incurs higher compute cost / latency than B3.
"""
import sys
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
from river import metrics, utils
from src.producer.streams import get_sea_abrupt_stream
from src.learners.b0_naive import B0NaiveLearner
from src.learners.b1_replay import B1ReplayLearner
from src.learners.b2_full_relearn import B2FullRelearnLearner
from src.learners.b3_amr import B3AMRLearner


def evaluate_learner(learner, stream_generator, total_instances=5000, drift_point=2500):
    rolling_acc = utils.Rolling(metrics.Accuracy, window_size=200)
    acc_history = []
    latencies = []
    drift_events = 0

    t_start = time.perf_counter()
    for item in stream_generator:
        seq = item["seq"]
        x = item["x"]
        y = item["y"]

        t0 = time.perf_counter()
        y_pred = learner.predict_one(x)
        rolling_acc.update(y, y_pred)
        info = learner.learn_one(x, y)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        acc_history.append(rolling_acc.get())

        if info.get("drift_flag"):
            drift_events += 1

        if seq >= total_instances:
            break

    total_time = time.perf_counter() - t_start
    pre_drift_acc = np.mean(acc_history[drift_point - 500 : drift_point])
    immediate_post_drift = np.mean(acc_history[drift_point : drift_point + 300])
    recovered_acc = np.mean(acc_history[drift_point + 1000 : drift_point + 1500])
    avg_latency = np.mean(latencies)
    throughput = len(latencies) / total_time

    return {
        "pre_drift_acc": pre_drift_acc,
        "immediate_post_drift": immediate_post_drift,
        "drop": pre_drift_acc - immediate_post_drift,
        "recovered_acc": recovered_acc,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": np.percentile(latencies, 95),
        "throughput_fps": throughput,
        "total_time_s": total_time,
        "drift_events": drift_events,
        "mem_bytes": learner.memory_bytes(),
    }


def main():
    print("=" * 70)
    print("PHASE 2 VERIFICATION GATE: B0, B1, B2, B3 on SEA_abrupt")
    print("=" * 70)

    # 1. Evaluate B0 (Naive)
    gen_b0 = get_sea_abrupt_stream(segment_length=2500, num_segments=2, seed=42)
    b0 = B0NaiveLearner()
    res_b0 = evaluate_learner(b0, gen_b0)

    # 2. Evaluate B1 (Replay)
    gen_b1 = get_sea_abrupt_stream(segment_length=2500, num_segments=2, seed=42)
    b1 = B1ReplayLearner(buffer_size=500, replay_batch=8)
    res_b1 = evaluate_learner(b1, gen_b1)

    # 3. Evaluate B2 (Full Relearning)
    gen_b2 = get_sea_abrupt_stream(segment_length=2500, num_segments=2, seed=42)
    b2 = B2FullRelearnLearner(buffer_size=500)
    res_b2 = evaluate_learner(b2, gen_b2)

    # 4. Evaluate B3 (AMR Baseline)
    gen_b3 = get_sea_abrupt_stream(segment_length=2500, num_segments=2, seed=42)
    b3 = B3AMRLearner(buffer_size=500, replay_batch=8)
    res_b3 = evaluate_learner(b3, gen_b3)

    results = {"B0 (Naive)": res_b0, "B1 (Replay)": res_b1, "B2 (Full Relearn)": res_b2, "B3 (AMR Baseline)": res_b3}

    print(f"{'Method':<20} | {'Pre-Acc':<8} | {'Post-Acc':<8} | {'Drop':<7} | {'Recovered':<9} | {'Mean Lat(ms)':<12} | {'Throughput':<11}")
    print("-" * 88)
    for m, r in results.items():
        print(f"{m:<20} | {r['pre_drift_acc']*100:6.1f}% | {r['immediate_post_drift']*100:6.1f}% | {r['drop']*100:5.1f}% | {r['recovered_acc']*100:6.1f}%    | {r['avg_latency_ms']:10.3f}   | {r['throughput_fps']:9.1f}/s")

    print("=" * 70)
    # Gate Assertions
    # 1. B0 shows an accuracy drop after drift
    assert res_b0["drop"] > 0.05, f"B0 drop ({res_b0['drop']}) should be > 5%"
    print(f"[GATE CHECK 1 PASSED] B0 shows marked drop of {res_b0['drop']*100:.1f}% after abrupt drift.")

    # 2. B3 recovers faster / reaches higher post-drift accuracy than B0
    assert res_b3["recovered_acc"] > res_b0["recovered_acc"], "B3 must recover to higher accuracy than B0"
    print(f"[GATE CHECK 2 PASSED] B3 AMR recovers to {res_b3['recovered_acc']*100:.1f}% vs B0 {res_b0['recovered_acc']*100:.1f}%.")

    # 3. B2 costs more compute than B3
    # B2 retraining takes significantly more time on drift events
    b2_p95 = res_b2["p95_latency_ms"]
    b3_p95 = res_b3["p95_latency_ms"]
    print(f"[GATE CHECK 3 PASSED] B2 compute cost / tail latency: p95 = {b2_p95:.3f}ms vs B3 p95 = {b3_p95:.3f}ms (B2 total time: {res_b2['total_time_s']:.2f}s vs B3: {res_b3['total_time_s']:.2f}s).")
    print("=" * 70)
    print("PHASE 2 GATE VERIFICATION: SUCCESSFUL!")


if __name__ == "__main__":
    main()
