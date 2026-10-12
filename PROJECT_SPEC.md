# PROJECT SPEC: Drift-Aware Concept Memory for Continual Learning on Concept-Drifting Data Streams

> **Instructions for the coding agent:** Build this project end to end from this spec. Work phase by phase (Section 8). After each phase, run the verification step before moving on. Do not skip the streaming infrastructure: the course explicitly rejects a static file read once, so data must flow through a message broker. Keep everything reproducible (fixed seeds, pinned versions) and write the README as you go.

---

## 1. Context

- **Course:** IOE413 Streaming Data Analytics, Mini Project (20% of the course grade, graded out of 50)
- **Team 5:** G. Yaswanth Sai (2023BCD0039), V. Rahul (2023BCD0040), V. Sai Sushanth (2023BCD0064), G S Suhotra (2023BCS0079)
- **Topic #10:** Continual Learning for Concept-Drifting Data Streams
- **Suggested course track:** Track B (Streaming ML for Anomaly and Drift): broker, then streaming consumer, then River online learning with drift detectors, then dashboard.

### Hard course requirements (must be satisfied)
1. A real or realistically simulated stream through Kafka or MQTT. A static CSV read once is not acceptable.
2. Three or more tools with real data flowing between them (not isolated scripts).
3. Reproduce at least a simplified version of one base paper's core method as a **baseline**.
4. Build a clearly scoped **enhancement** that is actually implemented, not just proposed.
5. Quantitative baseline vs. enhancement comparison, including honest reporting of negative results.
6. Deliverables: code and README (setup, run instructions, tool versions), a 6-8 page IEEE/ACM two-column report, and a demo of at most 10 minutes showing the pipeline running end to end.

### Grading weights (design priorities)
| Component | Points |
|---|---|
| Implementation and tool integration (real data flow) | 15 |
| Enhancement / novelty (implemented, non-trivial) | 8 |
| Results and evaluation | 7 |
| Report quality and IEEE/ACM formatting | 7 |
| Proposal and literature review | 5 |
| Q&A / defense of design choices | 5 |
| Demo | 3 |

---

## 2. Problem Statement

Real-world data arrives as a stream whose distribution changes over time (concept drift). Models trained once degrade after drift. Incremental learners adapt but can catastrophically forget earlier concepts that later **recur**. This project builds and evaluates a continual-learning approach that adapts to drift while retaining and reusing historical knowledge under a **fixed memory budget** and limited compute.

---

## 3. Base Papers

| # | Paper | Role |
|---|---|---|
| 1 | **CAGGA**: Concept Accumulation and Gradient-Guided Adaption for Continual Learning in Evolving Streaming (2024). https://www.sciencedirect.com/science/article/pii/S0925231224009652 . Code: https://github.com/Mr-XiongLin/CAGGA | Related work / reference |
| 2 | **ATNN**: Adaptive Tree-like Neural Network: Overcoming Catastrophic Forgetting to Classify Streaming Data with Concept Drifts (2024). https://www.sciencedirect.com/science/article/pii/S0950705124002715 | Related work / reference |
| 3 | **AMR**: Holistic Continual Learning under Concept Drift with Adaptive Memory Realignment (Ashrafee et al., 2025; TMLR 2026). https://arxiv.org/abs/2507.02310 . Code: https://github.com/AlifAshrafee/CL-Under-Concept-Drift | **Reproduced as the baseline (simplified)** |

Per-paper details (research question, data, metrics, findings, model) are in the submitted proposal PDF.

### Baseline paper: AMR (core mechanism to reproduce, simplified)
AMR is a drift-aware mechanism for rehearsal-based continual learning:
1. A replay buffer holds past samples.
2. A lightweight drift detector monitors each class (the paper uses a KS test on an uncertainty measure).
3. When drift is detected for class *c*, the outdated samples of *c* are **removed** from the buffer and replaced with a small number of fresh current-distribution samples (instead of expensive full relearning).

**Adaptation note (state this in the report):** the official AMR repo is PyTorch code for vision benchmarks organised as task sequences. We reproduce the *mechanism* (per-class drift detection plus buffer flush-and-refill) in River on tabular streams. We do not reproduce its vision experiments.

---

## 4. Proposed Enhancement

> **Concept-Aware Memory (CAM):** instead of one flat replay buffer that is flushed on drift, keep a **budgeted repository of concept memories**. Each entry holds a representative buffer segment plus a model snapshot. On drift, the system checks whether the new data matches a stored concept. If it does (a **recurring concept**), it **reinstates** that memory and model. If not, it opens a new concept memory and evicts the least useful stored concept when over budget.

How it differs from AMR (the baseline):
| Aspect | AMR baseline (simplified) | Enhancement (CAM) |
|---|---|---|
| On drift | Flush outdated samples and refill from the current distribution | Archive the old concept, then match against stored concepts |
| Recurring concepts | Relearned from scratch each time | Detected and reused (reinstated memory and model) |
| Buffer content | Reservoir / random samples | Representative selection (class-balanced reservoir plus herding or diversity) |
| Memory | Single fixed buffer | Fixed **total** budget split across concepts, with eviction policy |

One-sentence claim for the report: *"Relative to AMR, we add concept-aware memory that detects recurring concepts and reuses stored buffers and models under a fixed total memory budget, reducing adaptation delay and forgetting on recurring-drift streams."*

---

## 5. System Architecture

```
 [Stream Producer] --> [Kafka topic: stream.raw] --> [Streaming Consumer + River model]
   (replays SEA, STAGGER,                                  |  prequential test-then-train
    SINE, Elec2, with drift)                               |  drift detector + replay memory
                                                           v
                                              [Kafka topic: stream.metrics]
                                                           |
                                                           v
                                          [Metrics sink -> InfluxDB] --> [Grafana live dashboard]
                                                           |
                                                           v
                                          [Offline analysis: Matplotlib/Pandas -> report figures]
```

### Tool stack (7 tools, spanning ingest, process, model, store, visualize)
| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.10+ | Everything |
| Ingest | **Apache Kafka** (Docker, single-broker KRaft) | Real streaming source. **Fallback:** MQTT (Mosquitto) if Kafka is too heavy |
| Online ML and drift | **River** | Incremental learners, ADWIN / DDM / KSWIN, streaming metrics |
| Utilities | **scikit-learn, NumPy, Pandas** | Preprocessing, helpers, KS test (`scipy.stats`), offline analysis |
| Store | **InfluxDB 2.x** | Time-series metrics store. **Fallback:** SQLite |
| Dashboard | **Grafana** | Live view for the demo. **Fallback:** Streamlit |
| Plots | **Matplotlib** | Static figures for the report |

Pin versions in `requirements.txt` and record them in the README. Check the River API of the installed version (for example `river.drift.binary.DDM` vs `river.drift.DDM`).

### Message schemas (JSON)
`stream.raw`:
```json
{"seq": 1234, "ts": 1739999999.12, "x": {"x0": 0.3, "x1": 7.2, "x2": 4.1},
 "y": 1, "stream": "SEA_recurring", "true_concept": "A"}
```
(`true_concept` is ground truth for synthetic streams only and is used for evaluation, **never** given to the model.)

`stream.metrics` (one per instance or per N instances):
```json
{"seq": 1234, "ts": 1739999999.20, "method": "CAM", "stream": "SEA_recurring",
 "y_true": 1, "y_pred": 1, "rolling_acc": 0.91, "drift_flag": false,
 "concept_id": 2, "recurring_hit": false, "mem_bytes": 183000,
 "latency_ms": 0.42, "buffer_size": 500}
```

---

## 6. Methods to Implement

All methods share the same interface so they can be swapped by config:
```python
class StreamLearner:
    def predict_one(self, x) -> int: ...
    def learn_one(self, x, y) -> dict:  # returns info e.g. drift_flag, concept_id
        ...
    def memory_bytes(self) -> int: ...
```

| ID | Method | Description |
|---|---|---|
| **B0** | Naive online | River learner, `predict_one` then `learn_one`, no replay, no drift handling (shows forgetting) |
| **B1** | Vanilla replay | Fixed-size reservoir buffer, replay a small batch every step, **no** drift adaptation |
| **B2** | Full relearning (FR) | On detected drift, reset the model and retrain from the buffer or newest window (AMR paper's expensive reference) |
| **B3** | **AMR-simplified (reproduced baseline)** | B1 plus per-class drift detection; on drift for class *c*, delete buffer samples of *c* and refill with a few fresh samples of *c* |
| **E** | **CAM (enhancement)** | Concept repository plus recurring-concept matching plus representative selection under a total memory budget |

### Learners (River)
- Primary: `preprocessing.StandardScaler() | linear_model.LogisticRegression()` (replay actually helps these)
- Secondary: `tree.HoeffdingAdaptiveTreeClassifier` or `forest.ARFClassifier` (to show behaviour across model types)

### Drift detection
- Default: `drift.ADWIN` on the 0/1 error stream (global). For B3 and E, also per-class error streams.
- Ablation: `KSWIN`, `DDM`. The AMR paper's KS-test-on-uncertainty monitor may be added as an optional detector (`scipy.stats.ks_2samp` on prediction-confidence windows).

### CAM algorithm (implement precisely)
```
state: repo = []            # list of ConceptMemory {id, buffer, model_snapshot, fingerprint, last_used, hits}
       active = new ConceptMemory
       total_budget_samples = M (e.g., 500) shared across all concepts

for each (x, y) in stream:
    y_hat = active.model.predict_one(x)                # test
    update metrics
    active.model.learn_one(x, y)                       # train on current
    active.buffer.add(x, y)                            # class-balanced reservoir
    replay batch (size r) from active.buffer -> active.model.learn_one
    update fingerprint (running feature mean/var + class prior in window)

    if detector.update(error): # drift detected
        archive(active) into repo
        # matching: evaluate each stored concept's model on the most recent window W (e.g., last 100 labeled samples)
        best = argmin error over repo models on W
        if best.error < tau (e.g., 0.7 * current_model_error_on_W):
            active = reinstate(best)                   # recurring concept hit
            active.model.learn_one over W for quick fine-tune
        else:
            active = new ConceptMemory seeded with W  # novel concept
        enforce_budget(repo)                           # evict lowest (hits, recency) concept; shrink buffers proportionally
        detector.reset()
```
Matching can alternatively use the fingerprint distance (mean/var plus class prior). Implement error-based matching first, and fingerprint matching as an ablation if time allows.

### Representative selection (optional sub-enhancement)
Class-balanced reservoir by default. Option: herding or diversity-based replacement (keep samples closest to the class mean, or farthest-first), compared against plain reservoir in an ablation.

---

## 7. Streams / Datasets (all replayed through Kafka)

| Stream | Source | Drift pattern |
|---|---|---|
| `SEA_abrupt` | `river.datasets.synth.SEA` variants chained with `ConceptDriftStream` | Abrupt (e.g., every 5k instances, 50k total) |
| `SEA_recurring` | SEA variants in sequence **A-B-A-C-A-B** | Abrupt and **recurring** (key stream for the enhancement) |
| `STAGGER_recurring` | `river.datasets.synth.STAGGER` (functions 0, 1, 2) | Abrupt and recurring |
| `SINE_gradual` | `river.datasets.synth.Sine` plus `ConceptDriftStream` with width > 1 | Gradual |
| `Hyperplane` (optional) | `river.datasets.synth.Hyperplane` | Incremental |
| `Elec2` (real) | `river.datasets.Elec2` | Real non-stationary, unknown drift points |

Producer requirements:
- Publishes one JSON message per instance with a **configurable rate** (e.g., `--rate 500` msgs/s, `--rate 0` for max speed benchmarking).
- Deterministic by seed.
- Embeds `true_concept` for synthetic streams (evaluation only).
- Supports `--loop` and a stream-name argument.

---

## 8. Build Plan (phases, each with a verification gate)

### Phase 0: Environment
- Create repo layout (Section 10), `requirements.txt` (pinned), `docker-compose.yml` (Kafka KRaft, InfluxDB 2, Grafana), `.env.example`, `Makefile` with `make up`, `make down`, `make run-producer`, `make run-consumer`.
- **Gate:** `docker compose up -d` brings all services healthy. A test message round-trips through Kafka.

### Phase 1: Ingestion and storage skeleton (no ML yet)
- Producer, consumer with a placeholder model (predict a constant), metrics publisher, InfluxDB sink, provisioned Grafana dashboard (JSON in `dashboards/`).
- **Gate:** Run producer. Grafana shows live points updating. This proves real data flow.

### Phase 2: Baselines B0 to B3
- Implement `StreamLearner` implementations B0, B1, B2, B3 using River. Prequential (test-then-train) loop inside the consumer.
- Unit tests: buffer class balance, reservoir uniformity, AMR flush-and-refill correctness, detector fires on a synthetic abrupt shift.
- **Gate:** On `SEA_abrupt`, B0 shows an accuracy drop after drift. B3 recovers faster than B0 and B2 costs more compute than B3. Record the numbers.

### Phase 3: Enhancement E (CAM)
- Implement the concept repository, matching, reinstate/new/evict logic, budget enforcement.
- Unit tests: matching selects the right concept on A-B-A, total memory never exceeds budget, eviction picks the intended concept.
- **Gate:** On `SEA_recurring`, E shows `recurring_hit=true` at the recurrences of A and recovers faster than B3.

### Phase 4: Experiments and analysis
- `experiments/run_all.py`: for each stream x method x seed (**at least 5 seeds**), run the full pipeline (via Kafka) and save raw metrics to `results/*.csv`.
- `analysis/make_report_assets.py`: produce tables and figures (Section 9).
- **Gate:** All figures and tables are regenerated from raw CSV with one command.

### Phase 5: Documentation, report assets, demo
- README, report figures, architecture diagram, demo script (Section 12).

> **Schedule note:** The course timeline puts experiments at Oct 8-14 and final submission, demo, and presentation at Oct 15-21. Prioritise in this order if time is short: Phases 0-1 (infrastructure), then 2 (baseline), then 3 (enhancement), then 4 (results), then 5. A working end-to-end pipeline with a simple enhancement beats an ambitious unfinished one.

---

## 9. Evaluation

### Metrics
| Metric | Definition / implementation |
|---|---|
| Prequential accuracy, F1 (macro), precision, recall | `river.metrics`, cumulative and rolling window (e.g., 500) |
| **Adaptation / recovery time** | Instances after a drift point until rolling accuracy returns to at least 95% of its pre-drift level |
| **Forgetting on recurrence** | Accuracy in the first W instances after concept *A* recurs, minus accuracy at the end of the first exposure to *A* (lower drop is better) |
| Post-drift average accuracy | Mean accuracy in the 1000 instances after each drift |
| **Memory** | Buffer plus model snapshots, in bytes (`pympler.asizeof` or `tracemalloc`), reported against the budget |
| **Runtime** | Throughput (instances/s) and per-instance latency (p50, p95) |
| Drift detection quality | Detection delay and false alarms vs. known drift points on synthetic streams |

### Experiments
1. **Main comparison:** B0, B1, B2, B3 (AMR baseline), E (CAM) on all streams, 5+ seeds, report mean ± std.
2. **Memory budget sweep:** buffer budget in {100, 250, 500, 1000} samples. Plot accuracy vs. memory.
3. **Detector ablation:** ADWIN vs. KSWIN vs. DDM for B3 and E.
4. **Matching ablation (E):** error-based vs. fingerprint-based matching.
5. **Representative selection ablation (E):** reservoir vs. herding/diversity.
6. **Throughput test:** producer at several rates, to show the pipeline keeps up. Report max sustainable rate.
7. **Real stream (`Elec2`):** compare methods. No ground-truth drift, so report accuracy, F1, runtime, memory.

### Required outputs for the report
- Table: main comparison (accuracy, F1, recovery time, forgetting, memory, latency).
- Figure: rolling accuracy over time per method on `SEA_recurring`, with drift points and recurrence markers.
- Figure: accuracy vs. memory budget.
- Figure: recovery time bar chart.
- Figure: architecture diagram showing every tool and the data flow (Section 5).
- **Report negative results honestly** (for example, if CAM does not help on gradual drift or on `Elec2`, say so and explain why).

---

## 10. Repository Layout

```
project/
├── README.md                 # setup, run, tool versions, results summary
├── requirements.txt          # pinned versions
├── docker-compose.yml        # kafka, influxdb, grafana
├── Makefile
├── .env.example
├── src/
│   ├── producer/
│   │   ├── streams.py        # stream factories (SEA, STAGGER, SINE, Elec2, recurring/gradual builders)
│   │   └── producer.py       # Kafka producer CLI (--stream --rate --seed --loop)
│   ├── consumer/
│   │   ├── consumer.py       # Kafka consumer + prequential loop + metrics publisher
│   │   └── sink_influx.py    # stream.metrics -> InfluxDB
│   ├── learners/
│   │   ├── base.py           # StreamLearner interface
│   │   ├── b0_naive.py
│   │   ├── b1_replay.py
│   │   ├── b2_full_relearn.py
│   │   ├── b3_amr.py         # reproduced baseline
│   │   └── e_cam.py          # enhancement
│   ├── memory/
│   │   ├── reservoir.py      # class-balanced reservoir
│   │   ├── concept_repo.py   # concept memories, matching, eviction, budget
│   │   └── selection.py      # herding / diversity selection
│   ├── drift/
│   │   └── detectors.py      # ADWIN/KSWIN/DDM wrappers, per-class monitor, KS-uncertainty monitor
│   └── metrics/
│       └── eval.py           # recovery time, forgetting, memory, latency
├── experiments/
│   ├── configs/              # YAML per experiment
│   └── run_all.py
├── analysis/
│   └── make_report_assets.py
├── dashboards/               # Grafana dashboard JSON + provisioning
├── tests/                    # pytest unit tests
├── results/                  # raw CSVs (git-ignored except samples)
└── report/                   # IEEE/ACM LaTeX source + figures
```

---

## 11. Coding Requirements and Conventions

- Python type hints, docstrings on public functions, `pytest` tests for buffers, detectors, matching, and budget logic.
- All randomness seeded (`--seed`). Same seed gives identical streams and results.
- Config via YAML or CLI flags, no hard-coded paths.
- Consumer must be **prequential** (predict first, then learn) and must **never** see `true_concept`.
- Log each experiment's config and tool versions alongside its results CSV.
- Keep Kafka client code isolated (`confluent-kafka` or `kafka-python-ng`) so MQTT or an in-process fallback can be swapped in if needed.

---

## 12. Deliverables Checklist

### Code and README
- [ ] One-command start (`make up` and run scripts), setup and run instructions
- [ ] Tools and pinned versions listed (Python, River, Kafka image, InfluxDB, Grafana, scikit-learn, NumPy, Pandas, Matplotlib)
- [ ] How to reproduce every table and figure

### Final report (6-8 pages, IEEE or ACM two-column, PDF **and** editable source)
Required sections: Title, authors, affiliation, abstract (150-250 words); I. Introduction; II. Related Work (CAGGA, ATNN, AMR plus the gap); III. System Architecture (diagram with every tool); IV. Proposed Enhancement; V. Implementation (streams, versions, key config); VI. Experimental Setup and Results; VII. Discussion (limitations, threats to validity); VIII. Conclusion and Future Work; References (IEEE or ACM style).

Gap statement to use: *AMR realigns memory after drift but treats every drift as a new distribution, so recurring concepts are relearned and memory is not managed across concepts. CAGGA and ATNN reuse classifiers or branches but rely on heavier neural meta-learning. We add lightweight, budgeted, concept-aware memory in a true streaming pipeline.*

### Demo (at most 10 minutes)
1. Show architecture and `docker compose up`.
2. Start producer (`SEA_recurring`) and Grafana live dashboard.
3. Run B3 vs. E side by side. Point out drift events and a **recurring-concept hit** where E recovers instantly.
4. Show the memory budget and latency panels.
5. Show the results table and conclusion.

---

## 13. Acceptance Criteria (definition of done)

1. `make up`, then producer, then consumer runs end to end, with data visibly flowing through Kafka to the River model to InfluxDB to Grafana.
2. B0, B1, B2, B3, and E all run through the same pipeline and produce comparable metrics.
3. On `SEA_recurring`, E demonstrates recurring-concept reuse (logged `recurring_hit` events) and a measured improvement, or an honestly reported lack of one, versus B3.
4. Memory never exceeds the configured budget (asserted in tests).
5. All report tables and figures are regenerated from `results/` by a single command.
6. README lets a new person reproduce the setup and results from scratch.

---

## 14. Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Kafka setup too heavy | Mosquitto MQTT with the same topic layout |
| InfluxDB or Grafana problems | SQLite plus Streamlit live dashboard |
| Replay does not help tree learners | Use logistic regression as the primary learner and report tree results as secondary |
| CAM shows no gain on some streams | Report it honestly and analyse why (a negative result is explicitly rewarded) |
| Time pressure | Cut ablations 4 and 5, keep the main comparison and the budget sweep |
