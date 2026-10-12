"""
Real-time Streamlit Dashboard for Continual Learning Benchmarks.
Fallback live dashboard connecting to SQLite / InfluxDB metrics store.
"""
import os
import sys
import time
import sqlite3
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

st.set_page_config(
    page_title="Concept Drift Lab · IOE413",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SQLITE_PATH = os.getenv("SQLITE_DB_PATH", "results/metrics.db")


def load_metrics(limit: int = 5000) -> pd.DataFrame:
    if not os.path.exists(SQLITE_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        query = f"""
            SELECT seq, ts, method, stream, y_true, y_pred, rolling_acc,
                   drift_flag, concept_id, recurring_hit, mem_bytes, latency_ms, buffer_size
            FROM metrics
            ORDER BY seq DESC
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error querying SQLite database: {e}")
        return pd.DataFrame()


st.title("⚡ IOE413 Continual Learning Streaming Dashboard")
st.caption("Live streaming analytics: Producer ➔ Message Broker ➔ StreamLearner ➔ Storage ➔ Dashboard")

col_side1, col_side2 = st.sidebar.columns(2)
auto_refresh = st.sidebar.checkbox("Auto-refresh (1s)", value=True)
sample_limit = st.sidebar.slider("History Window Size", min_value=500, max_value=10000, value=3000, step=500)

df = load_metrics(limit=sample_limit)

if df.empty:
    st.info("No metrics received yet. Run the streaming producer and consumer pipeline to view live metrics.")
    st.code("python src/producer/producer.py --stream SEA_recurring --rate 100\npython src/consumer/consumer.py --stream SEA_recurring --method E")
else:
    latest = df.iloc[-1]
    method = latest.get("method", "Unknown")
    stream = latest.get("stream", "Unknown")
    total_samples = len(df)
    drifts_detected = int(df["drift_flag"].sum())
    recurring_hits = int(df["recurring_hit"].sum())
    cur_acc = float(latest.get("rolling_acc", 0.0))
    p50_lat = float(df["latency_ms"].median())
    p95_lat = float(np.percentile(df["latency_ms"], 95))
    mem_kb = float(latest.get("mem_bytes", 0)) / 1024.0

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Method / Stream", f"{method}", f"{stream}")
    kpi2.metric("Instances", f"{int(latest.get('seq', total_samples)):,}")
    kpi3.metric("Rolling Accuracy", f"{cur_acc * 100:.1f}%")
    kpi4.metric("Drift Detections", f"{drifts_detected}")
    kpi5.metric("Recurring Hits", f"{recurring_hits}")
    kpi6.metric("Latency (p50 / p95)", f"{p50_lat:.2f}ms / {p95_lat:.2f}ms", f"Mem: {mem_kb:.1f} KB")

    st.markdown("---")

    # Time series plots
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.subheader("📈 Prequential Rolling Accuracy")
        chart_data = df[["seq", "rolling_acc"]].set_index("seq")
        st.line_chart(chart_data)

    with row1_c2:
        st.subheader("🧠 Concept Evolution & Recurring Hits")
        concept_data = df[["seq", "concept_id"]].set_index("seq")
        st.line_chart(concept_data)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.subheader("💾 Memory Footprint (Bytes)")
        mem_data = df[["seq", "mem_bytes"]].set_index("seq")
        st.area_chart(mem_data)

    with row2_c2:
        st.subheader("⏱️ Per-Instance Latency (ms)")
        lat_data = df[["seq", "latency_ms"]].set_index("seq")
        st.line_chart(lat_data)

    # Recent Records Table
    with st.expander("📋 Recent Streaming Metrics Log (Last 15 instances)"):
        st.dataframe(df.tail(15).iloc[::-1], use_container_width=True)

if auto_refresh:
    time.sleep(1.0)
    st.rerun()
