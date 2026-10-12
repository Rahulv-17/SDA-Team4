"""
Report Assets Generator.
Generates all required IEEE/ACM publication figures and tables directly from raw results/*.csv.
Outputs:
- report/figures/figure1_rolling_accuracy.png
- report/figures/figure2_memory_vs_accuracy.png
- report/figures/figure3_recovery_time_barchart.png
- report/figures/figure4_architecture_diagram.png
- report/tables/table1_main_comparison.tex
- report/tables/table2_budget_sweep.tex
- report/tables/table3_ablations.tex
- report/tables/summary_tables.md
"""
import sys
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_figure1_rolling_accuracy():
    trace_path = RESULTS_DIR / "traces_sea_recurring.csv"
    if not trace_path.exists():
        print(f"Skipping Figure 1: {trace_path} not found.")
        return

    df = pd.read_csv(trace_path)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    colors = {
        "B0": "#d9534f",
        "B1": "#f0ad4e",
        "B2": "#5bc0de",
        "B3": "#0275d8",
        "E": "#28a745",
    }
    labels = {
        "B0": "B0: Naive Online",
        "B1": "B1: Vanilla Replay",
        "B2": "B2: Full Relearning",
        "B3": "B3: AMR Baseline",
        "E": "E: CAM (Enhancement)",
    }

    for method in ["B0", "B1", "B2", "B3", "E"]:
        sub = df[df["method"] == method]
        if not sub.empty:
            ax.plot(sub["seq"], sub["rolling_acc"], label=labels.get(method, method),
                    color=colors.get(method, "gray"), linewidth=1.8 if method == "E" else 1.2,
                    alpha=0.9 if method == "E" else 0.75)

    # Concept boundaries on SEA_recurring: 2500, 5000, 7500, 10000
    concepts = [
        (0, 2500, "Concept A"),
        (2500, 5000, "Concept B"),
        (5000, 7500, "Concept A (Recur 1)"),
        (7500, 10000, "Concept C"),
    ]
    for start, end, name in concepts:
        if end <= df["seq"].max():
            ax.axvline(x=end, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text((start + end) / 2, 0.62, name, ha="center", fontsize=9, style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.8))

    ax.set_title("Prequential Rolling Accuracy on SEA_recurring (A-B-A-C-A-B)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Instance Sequence (t)", fontsize=11)
    ax.set_ylabel("Rolling Accuracy (Window = 200)", fontsize=11)
    ax.set_ylim(0.60, 1.01)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)

    out_path = FIG_DIR / "figure1_rolling_accuracy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_figure2_memory_sweep():
    sweep_path = RESULTS_DIR / "budget_sweep.csv"
    if not sweep_path.exists():
        print(f"Skipping Figure 2: {sweep_path} not found.")
        return

    df = pd.read_csv(sweep_path)
    grouped = df.groupby(["method", "budget_samples"])["accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    for method, col, marker in [("B3", "#0275d8", "o"), ("E", "#28a745", "s")]:
        sub = grouped[grouped["method"] == method]
        if not sub.empty:
            label = "B3: AMR Baseline" if method == "B3" else "E: CAM (Enhancement)"
            ax.errorbar(sub["budget_samples"], sub["mean"], yerr=sub["std"], label=label,
                        color=col, marker=marker, linewidth=2, capsize=4, markersize=7)

    ax.set_title("Prequential Accuracy vs. Memory Budget (Samples)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Total Buffer Budget (Samples M)", fontsize=11)
    ax.set_ylabel("Prequential Accuracy", fontsize=11)
    ax.legend(loc="lower right", frameon=True)

    out_path = FIG_DIR / "figure2_memory_vs_accuracy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_figure3_recovery_time_barchart():
    main_path = RESULTS_DIR / "main_comparison.csv"
    if not main_path.exists():
        print(f"Skipping Figure 3: {main_path} not found.")
        return

    df = pd.read_csv(main_path)
    # Focus on synthetic streams with drift points
    df_rec = df[df["stream"].isin(["SEA_recurring", "SEA_abrupt", "STAGGER_recurring"])]
    grouped = df_rec.groupby(["stream", "method"])["recovery_time"].mean().unstack()

    methods_order = [m for m in ["B0", "B1", "B2", "B3", "E"] if m in grouped.columns]
    grouped = grouped[methods_order]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#0275d8", "#28a745"][:len(methods_order)]

    grouped.plot(kind="bar", ax=ax, color=colors, width=0.8, edgecolor="black", linewidth=0.5)

    ax.set_title("Adaptation Delay: Recovery Time after Drift (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Stream Benchmark", fontsize=11)
    ax.set_ylabel("Mean Recovery Time (Instances)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Method", loc="upper right")

    out_path = FIG_DIR / "figure3_recovery_time_barchart.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_figure4_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axis("off")

    def draw_box(x, y, w, h, text, subtext, color="#e9ecef", border="#495057"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                      facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.05, text, ha="center", va="center", fontsize=10, fontweight="bold")
        if subtext:
            ax.text(x + w / 2, y + h / 2 - 0.09, subtext, ha="center", va="center", fontsize=8, color="#495057")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#212529"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label, ha="center", va="center", fontsize=8, fontweight="bold", color="#0056b3")

    # Layer 1: Producer
    draw_box(0.02, 0.45, 0.18, 0.35, "Stream Producer", "SEA, STAGGER, SINE, Elec2\n(Deterministic Rate/Seed)", "#d1ecf1", "#0c5460")

    # Layer 2: Message Broker
    draw_box(0.26, 0.45, 0.18, 0.35, "Message Broker", "Apache Kafka KRaft /\nMQTT (127.0.0.1:1883)", "#fff3cd", "#856404")

    # Layer 3: Streaming Consumer + River Learners
    draw_box(0.50, 0.45, 0.22, 0.35, "Streaming Consumer", "River Prequential Loop\nBaselines B0-B3 & Enhancement E", "#d4edda", "#155724")

    # Layer 4: Storage
    draw_box(0.78, 0.65, 0.20, 0.25, "Metrics Store", "InfluxDB 2.x /\nSQLite (results/metrics.db)", "#e2e3e5", "#383d41")

    # Layer 5: Visualization
    draw_box(0.78, 0.25, 0.20, 0.25, "Live Dashboard", "Grafana (Port 3000) /\nStreamlit (Port 8501)", "#f8d7da", "#721c24")

    # Connections
    draw_arrow(0.20, 0.62, 0.26, 0.62, "stream.raw")
    draw_arrow(0.44, 0.62, 0.50, 0.62, "JSON events")
    draw_arrow(0.72, 0.68, 0.78, 0.72, "stream.metrics")
    draw_arrow(0.88, 0.65, 0.88, 0.50, "Metrics Query")

    ax.set_title("System Architecture: Real Streaming Data Flow & Continual Learning", fontsize=13, fontweight="bold", pad=15)

    out_path = FIG_DIR / "figure4_architecture_diagram.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def generate_tables():
    main_path = RESULTS_DIR / "main_comparison.csv"
    if not main_path.exists():
        print(f"Skipping tables: {main_path} not found.")
        return

    df = pd.read_csv(main_path)

    # 1. Main comparison summary (mean ± std across seeds)
    agg_funcs = {
        "accuracy": ["mean", "std"],
        "f1_macro": ["mean", "std"],
        "recovery_time": ["mean"],
        "forgetting": ["mean"],
        "mean_latency_ms": ["mean"],
        "memory_bytes": ["mean"],
    }
    grouped = df.groupby(["stream", "method"]).agg(agg_funcs).reset_index()

    # Create formatted strings
    rows = []
    for _, r in grouped.iterrows():
        acc_str = f"{r[('accuracy', 'mean')]*100:.2f} ± {r[('accuracy', 'std')]*100:.2f}%"
        f1_str = f"{r[('f1_macro', 'mean')]:.3f} ± {r[('f1_macro', 'std')]:.3f}"
        rec_str = f"{r[('recovery_time', 'mean')]:.1f}"
        forg_str = f"{r[('forgetting', 'mean')]*100:.2f}%"
        lat_str = f"{r[('mean_latency_ms', 'mean')]:.3f} ms"
        mem_str = f"{r[('memory_bytes', 'mean')]/1024:.1f} KB"
        rows.append({
            "Stream": r[("stream", "")],
            "Method": r[("method", "")],
            "Accuracy": acc_str,
            "Macro F1": f1_str,
            "Recovery Time": rec_str,
            "Forgetting Drop": forg_str,
            "Latency": lat_str,
            "Memory": mem_str,
        })

    summary_df = pd.DataFrame(rows)

    # Save Markdown Table
    md_content = "# Experimental Results Summary\n\n"
    md_content += "### Table 1: Main Comparison Across Streams and Methods (5 Seeds Mean ± Std)\n\n"
    md_content += summary_df.to_markdown(index=False) + "\n\n"

    # Save LaTeX Table
    tex_content = summary_df.to_latex(index=False, caption="Quantitative Main Comparison across Benchmark Streams", label="tab:main_results")
    with open(TAB_DIR / "table1_main_comparison.tex", "w") as f:
        f.write(tex_content)

    # Budget Sweep Table if exists
    sweep_path = RESULTS_DIR / "budget_sweep.csv"
    if sweep_path.exists():
        df_sw = pd.read_csv(sweep_path)
        g_sw = df_sw.groupby(["budget_samples", "method"])["accuracy"].agg(["mean", "std"]).reset_index()
        g_sw["Accuracy"] = g_sw.apply(lambda r: f"{r['mean']*100:.2f} ± {r['std']*100:.2f}%", axis=1)
        piv = g_sw.pivot(index="budget_samples", columns="method", values="Accuracy").reset_index()
        piv.columns.name = None
        piv.rename(columns={"budget_samples": "Sample Budget"}, inplace=True)
        md_content += "### Table 2: Memory Budget Sweep on SEA_recurring\n\n"
        md_content += piv.to_markdown(index=False) + "\n\n"
        with open(TAB_DIR / "table2_budget_sweep.tex", "w") as f:
            f.write(piv.to_latex(index=False, caption="Memory Budget vs Accuracy", label="tab:budget_sweep"))

    # Ablations Table if exists
    det_path = RESULTS_DIR / "detector_ablation.csv"
    if det_path.exists():
        df_det = pd.read_csv(det_path)
        g_det = df_det.groupby(["detector", "method"])["accuracy"].mean().unstack().reset_index()
        md_content += "### Table 3: Drift Detector Ablation (Accuracy)\n\n"
        md_content += g_det.to_markdown(index=False) + "\n\n"
        with open(TAB_DIR / "table3_ablations.tex", "w") as f:
            f.write(g_det.to_latex(index=False, caption="Drift Detector Ablation", label="tab:detector_ablation"))

    with open(TAB_DIR / "summary_tables.md", "w") as f:
        f.write(md_content)

    print(f"Generated LaTeX & Markdown tables in {TAB_DIR}")


def main():
    print("Generating all report assets and publication figures from results/*.csv...")
    plot_figure1_rolling_accuracy()
    plot_figure2_memory_sweep()
    plot_figure3_recovery_time_barchart()
    plot_figure4_architecture_diagram()
    generate_tables()
    print("Report assets generation finished successfully!")


if __name__ == "__main__":
    main()
