# Drift-Aware Concept Memory for Continual Learning on Concept-Drifting Data Streams

**Course:** IOE413 Streaming Data Analytics — Mini Project (Team 5)  
**Topic #10:** Continual Learning for Concept-Drifting Data Streams  
**Authors:**
- G. Yaswanth Sai (2023BCD0039)
- V. Rahul (2023BCD0040)
- V. Sai Sushanth (2023BCD0064)
- G S Suhotra (2023BCS0079)

---

## 1. Executive Summary

Real-world streaming data exhibits non-stationarity and recurring concept drift. Incremental online learners adapt to shifts but catastrophically forget previously mastered concepts when distributions recur. While baseline rehearsal strategies like **Adaptive Memory Realignment (AMR, Ashrafee et al., 2025)** flush outdated samples per class upon drift, they treat each recurrence as an entirely novel concept and relearn from scratch.

This project implements an end-to-end streaming analytics system in Python using **River 0.26.1**, **Apache Kafka** / **MQTT**, **InfluxDB 2** / **SQLite**, and **Grafana** / **Streamlit**. We introduce **Concept-Aware Memory (CAM)**, an enhancement that maintains a budgeted repository of concept memories (buffer segments, model snapshots, and statistical fingerprints). Upon drift detection, CAM matches post-drift validation data against archived memories: recurring concepts are reinstated and fine-tuned in sub-milliseconds, while novel concepts are budgeted under utility-based eviction.

On the recurring benchmark stream **STAGGER_recurring**, CAM achieves **91.58%** accuracy and **0.902 Macro F1** (+2.94% over AMR), while operating under a strict 500-sample total budget at **0.093 ms/sample** latency.

---

## 2. Pinned Tool Stack & Versions

| Layer | Component | Version | Purpose |
|---|---|---|---|
| Language | Python | `3.12.4` | Core runtime |
| Online ML & Drift | River | `0.26.1` | Incremental SGD models, ADWIN/DDM/KSWIN, prequential metrics |
| Machine Learning | scikit-learn | `1.8.0` | Feature metrics and baselines |
| Scientific | NumPy / SciPy | `2.2.6` / `1.14.1` | Linear algebra and statistical tests |
| Data & Tables | Pandas / Tabulate | `2.2.3` / `0.10.0` | Stream analysis and tabular reporting |
| Message Broker | Apache Kafka / MQTT | Confluent `7.6.0` / `amqtt 0.12.1` | Decoupled high-throughput message bus |
| Broker Clients | kafka-python-ng / paho-mqtt | `2.2.3` / `2.1.0` | Real streaming pub/sub |
| Time-Series Store | InfluxDB / SQLite | `2.7` / SQLite3 | Metrics persistence |
| Live Dashboard | Grafana / Streamlit | `10.4.0` / `1.65.0` | Real-time monitoring and visualization |
| Plots | Matplotlib / Seaborn | `3.9.2` / `0.13.2` | IEEE publication figures (300 DPI) |

---

## 3. End-to-End System Architecture

```text
 [Stream Producer] ──────────► [Topic: stream.raw] ──────────► [Streaming Consumer]
 (SEA, STAGGER, SINE, Elec2)     (Kafka / MQTT 1883)          (Prequential Test-then-Train Loop)
                                                                 │
                                                                 ├── Model: StandardScaler | LogisticRegression
                                                                 ├── Drift: ADWIN / Per-Class Monitor
                                                                 └── Memory: CAM Concept Repository
                                                                 │
                                                                 ▼
 [Live Dashboard]  ◄────────── [Metrics Store]   ◄────────── [Topic: stream.metrics]
 (Grafana :3000 /              (InfluxDB 2 :8086 /             (Real-time JSON Diagnostic Events)
  Streamlit :8501)              SQLite results/metrics.db)
```

---

## 4. Setup and Quick Start

### A. Environment Installation
Clone repository and install pinned dependencies:
```bash
python -m pip install -r requirements.txt
```

### B. One-Command Complete Pipeline Launch 🚀
Start the entire streaming pipeline (broker + Stitch dashboard + consumer + producer) with a single command:
```bash
python run_demo.py
```
Or with GNU `make`:
```bash
make start
```
Or with Windows PowerShell:
```powershell
.\run.ps1 start
```
*This automatically checks the message broker, launches the Google Stitch Research Dashboard on `http://localhost:8501`, connects the River streaming consumer (Method E), starts the synthetic stream producer, and opens your browser.*

### C. Individual Services (Modular Control)
If you prefer running services in separate terminals:
1. **Start Message Broker:**
   ```bash
   python src/broker/run_broker.py
   ```
2. **Start Stitch Research Dashboard:**
   ```bash
   python src/dashboard/server.py --port 8501
   ```
3. **Start River Streaming Consumer:**
   ```bash
   python src/consumer/consumer.py --stream SEA_recurring --method E
   ```
4. **Start Streaming Producer:**
   ```bash
   python src/producer/producer.py --stream SEA_recurring --rate 100 --seed 42
   ```

---

## 5. Five Evaluated Continual Learning Methods

All learners inherit from the shared `StreamLearner` interface (`predict_one`, `learn_one`, `memory_bytes`):
1. **B0 (Naive Online):** Online SGD logistic regression pipeline with no buffer and no drift handling (shows catastrophic forgetting).
2. **B1 (Vanilla Replay):** Class-balanced reservoir buffer of capacity $M=500$, replaying batches of 8 samples every step without drift awareness.
3. **B2 (Full Relearning):** On detected drift, discards model and retrains a fresh model from the stored buffer (AMR paper's expensive reference).
4. **B3 (AMR-Simplified Baseline):** Reproduces Ashrafee et al. (2025): monitors per-class drift; on drift for class $c$, flushes outdated samples of $c$ and refills from fresh distribution.
5. **E (CAM Enhancement):** Maintains a budgeted repository of concept memories; on drift, matches against historical concepts, reinstating stored models and buffers upon recurrence with utility-based eviction under a fixed total memory budget.

---

## 6. Reproducing Experiments and Report Assets

To run all 100+ experimental trials across 5 seeds, budget sweeps, and ablations:
```bash
python experiments/run_all.py
```
*Output: `results/main_comparison.csv`, `results/budget_sweep.csv`, `results/detector_ablation.csv`, `results/matching_ablation.csv`, `results/selection_ablation.csv`, `results/traces_sea_recurring.csv`.*

To regenerate all IEEE publication figures and LaTeX tables from raw CSVs with one command:
```bash
python analysis/make_report_assets.py
```
*Generated Assets:*
- `report/figures/figure1_rolling_accuracy.png`
- `report/figures/figure2_memory_vs_accuracy.png`
- `report/figures/figure3_recovery_time_barchart.png`
- `report/figures/figure4_architecture_diagram.png`
- `report/tables/table1_main_comparison.tex`
- `report/tables/table2_budget_sweep.tex`
- `report/tables/table3_ablations.tex`
- `report/tables/summary_tables.md`

---

## 7. Running Unit Tests

Run full unit tests across all verification gates:
```bash
pytest tests/ -v
```
All unit tests verify:
- Phase 0: Message broker roundtrip pub/sub
- Phase 2: Reservoir class balance, capacity enforcement, AMR flush-and-refill, ADWIN detection, learner interfaces
- Phase 3: CAM concept matching on A-B-A sequences, budget enforcement, utility-based concept eviction

---

## 8. Summary of Results (5 Seeds Mean ± Std)

| Stream | Method | Accuracy | Macro F1 | Recovery Time | Latency | Memory |
|---|---|---|---|---|---|---|
| **STAGGER_recurring** | B0 (Naive) | 83.03 ± 0.94% | 0.788 ± 0.015 | 0 inst | 0.015 ms | 6.0 KB |
| | B1 (Replay) | 71.52 ± 1.56% | 0.676 ± 0.015 | 0 inst | 0.074 ms | 163.1 KB |
| | B2 (Full Relearn) | 75.58 ± 1.18% | 0.743 ± 0.012 | 372.9 inst | 0.031 ms | 163.1 KB |
| | B3 (AMR Baseline) | 88.64 ± 1.22% | 0.868 ± 0.014 | 0 inst | 0.082 ms | 137.9 KB |
| | **E (CAM)** | **91.58 ± 0.96%** | **0.902 ± 0.011** | 0 inst | 0.093 ms | 215.8 KB |
| **SEA_recurring** | B3 (AMR Baseline) | 96.38 ± 0.13% | 0.959 ± 0.001 | 0 inst | 0.087 ms | 233.2 KB |
| | **E (CAM)** | **96.55 ± 0.37%** | **0.961 ± 0.004** | 0 inst | 0.095 ms | 307.4 KB |
| **Elec2 (Real)** | B1 (Vanilla Replay) | 87.48 ± 0.19% | 0.873 ± 0.002 | 0 inst | 0.118 ms | 265.8 KB |
| | B3 (AMR Baseline) | 83.71 ± 0.31% | 0.834 ± 0.003 | 0 inst | 0.117 ms | 205.1 KB |
| | **E (CAM)** | 85.20 ± 0.06% | 0.850 ± 0.001 | 0 inst | 0.138 ms | 318.9 KB |

*Honest Reporting of Negative / Neutral Results:* On the real non-stationary stream `Elec2`, where drift is continuous without sharp concept recurrences, frequent memory flushes in B3 degrade performance (83.71%). While CAM mitigates this drop (85.20%), vanilla replay B1 without drift flushes achieves 87.48%. This demonstrates that explicit concept memory repositories deliver high gains primarily on streams with discrete recurring regimes.
