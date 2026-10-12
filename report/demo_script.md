# IOE413 Mini Project Demo & Presentation Script (Team 5)

**Project Title:** Drift-Aware Concept Memory for Continual Learning on Concept-Drifting Data Streams  
**Time Limit:** At most 10 minutes  
**Speakers / Presenters:**
- G. Yaswanth Sai (2023BCD0039)
- V. Rahul (2023BCD0040)
- V. Sai Sushanth (2023BCD0064)
- G S Suhotra (2023BCS0079)

---

## Timeline & Scene-by-Scene Walkthrough

### 0:00 - 1:30 | Scene 1: Problem Statement, Research Gap & Motivation
- **Presenter:** G. Yaswanth Sai
- **Action:** Display slide / report Title page and Figure 4 (Architecture Diagram).
- **Spoken Talking Points:**
  > "Hello everyone. We are Team 5, presenting our IOE413 project on *Continual Learning for Concept-Drifting Data Streams*. In real-world high-velocity streams—such as financial transactions or sensor networks—data distributions evolve over time. While online models adapt, when a previously seen concept *recurs*, conventional online learners suffer from catastrophic forgetting and relearn everything from scratch.
  >
  > Our research gap directly builds upon the state-of-the-art **AMR baseline (Ashrafee et al., 2025; TMLR 2026)**. AMR realigns memory after drift by flushing outdated samples per class, but treats every drift as a brand new distribution. We introduce **Concept-Aware Memory (CAM)**: a budgeted repository of concept memories that detects recurring regimes, reinstating stored model weights and buffer segments under a strict global memory budget."

---

### 1:30 - 3:30 | Scene 2: Infrastructure & Streaming Pipeline Launch
- **Presenter:** V. Rahul
- **Action:** Open terminal and run broker startup:
  ```powershell
  .\run.ps1 up
  ```
  Show terminal output confirming broker is listening on `127.0.0.1:1883` (or Docker Kafka on `9092`).
- **Spoken Talking Points:**
  > "Following the hard course requirements, our data is never read statically from a CSV into a local model. All messages strictly travel through a decoupled message broker.
  >
  > Here, our broker is running on port 1883. We now launch our live real-time dashboard on port 8501:
  ```powershell
  streamlit run src/dashboard/app.py
  ```
  > In the browser, the dashboard is live, awaiting streaming metric events."

---

### 3:30 - 6:30 | Scene 3: Live Side-by-Side Execution (AMR Baseline vs. CAM Enhancement)
- **Presenter:** V. Sai Sushanth
- **Action:**
  1. Terminal 1: Launch Streaming Consumer with Enhancement E:
     ```bash
     python src/consumer/consumer.py --stream SEA_recurring --method E
     ```
  2. Terminal 2: Launch Streaming Producer for `SEA_recurring` at 150 msgs/sec:
     ```bash
     python src/producer/producer.py --stream SEA_recurring --rate 150 --seed 42
     ```
  3. Switch to browser window showing `http://localhost:8501`.
- **Spoken Talking Points:**
  > "Watch the live dashboard as `SEA_recurring` streams through the message broker. The stream follows the cyclic concept progression: Concept A, then Concept B, then Concept A recurs, Concept C, Concept A recurs again, then Concept B.
  >
  > Notice the first 2,500 instances: the model masters Concept A with >97% rolling accuracy.
  > At instance 2,500, an abrupt drift shifts the stream to Concept B. ADWIN detects the shift, archives Concept A into our Concept Repository, and opens a new concept.
  >
  > Now, observe what happens at instance 5,000 when Concept A recurs!
  > In the baseline AMR model, the learner must flush its buffer and relearn Concept A from scratch, suffering an adaptation delay.
  > In CAM, our validation window matches against the archived Concept A snapshot. Look at the KPI panel: **`Recurring Hits: 1`** triggers instantly! The model reinstates Concept A's weights and exemplars, adapting in sub-milliseconds without forgetting."

---

### 6:30 - 8:00 | Scene 4: Memory Budget Enforcement & Latency Inspection
- **Presenter:** G S Suhotra
- **Action:** Highlight the bottom panels on the dashboard:
  - *Memory Footprint (Bytes)* area chart.
  - *Per-Instance Latency (ms)* line chart.
- **Spoken Talking Points:**
  > "A continual learning algorithm is impractical if its memory grows unbounded. CAM operates under a strict total sample budget ($M = 500$ samples across all stored concepts).
  >
  > Look at the memory area chart: as new concepts are discovered, CAM rescales per-concept buffer sizes proportionally. If the repository exceeds its capacity, our utility policy evicts the concept with the lowest score ($\text{hits} \times 1000 + \text{last\_used\_step}$).
  >
  > Looking at the latency chart, our prequential predict-then-learn latency remains between **0.08 ms and 0.14 ms per sample**, allowing the pipeline to easily sustain over **10,000 instances/second**."

---

### 8:00 - 10:00 | Scene 5: Quantitative Results, Honest Reporting & Conclusion
- **Presenter:** G. Yaswanth Sai
- **Action:** Display Figure 1 (`report/figures/figure1_rolling_accuracy.png`), Table 1, and Figure 3.
- **Spoken Talking Points:**
  > "To validate our findings rigorously, we ran over 100 benchmark trials across 5 random seeds:
  > 1. On **STAGGER_recurring**, CAM achieves **91.58% accuracy and 0.902 Macro F1**, outperforming the AMR baseline by **+2.94%**.
  > 2. On **SEA_recurring**, CAM eliminates recovery delay on recurrence, maintaining higher post-drift stability.
  > 3. In our ablation studies, ADWIN delivered the highest drift detection precision, and error-based validation window matching outperformed static fingerprint matching.
  >
  > **Honest Reporting of Neutral Results:** On the real-world non-stationary `Elec2` stream, where drift is continuous and gradual without sharp concept recurrence, frequent drift flushes in AMR actually degraded performance (83.71%). While CAM mitigated this (85.20%), vanilla replay without flushes achieved 87.48%. This demonstrates that concept repositories provide decisive advantages specifically when recurring regimes occur.
  >
  > In conclusion, CAM proves that lightweight, budgeted concept-aware memory can be integrated directly into a production streaming pipeline to overcome catastrophic forgetting on concept-drifting streams. Thank you! We are open for questions."
