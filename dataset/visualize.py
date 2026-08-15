"""Generate visualization plots and charts for Glint-Research/Fable-5-traces."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")


def generate_plots(report_json_path: str = "dataset/reports/inspection_report.json", output_dir: str = "dataset/reports/figures"):
    """Generate publication-quality charts from inspection_report.json."""
    r_path = Path(report_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not r_path.exists():
        print(f"[!] Report JSON not found at {r_path}. Please run inspect_dataset.py first.")
        return

    with open(r_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Style settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
    })

    # 1. Context Window Budget Bar Chart
    plt.figure(figsize=(10, 6), dpi=300)
    budget = report.get("context_window_budget", {})
    labels = [k.replace("_tokens", "").upper() for k in budget.keys()]
    fit_pcts = [v["fit_percentage"] for v in budget.values()]

    colors = ["#2b5c8f" if p >= 95 else "#e06666" if p < 50 else "#f6b26b" for p in fit_pcts]
    bars = plt.bar(labels, fit_pcts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    for bar, pct in zip(bars, fit_pcts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.title(f"Dataset Context Window Fit % ({report.get('tokenizer', 'Qwen/Qwen3.8-27B')})", pad=15)
    plt.xlabel("Maximum Sequence Length (Tokens)")
    plt.ylabel("Samples Fit (% of Total)")
    plt.ylim(0, 110)
    plt.axhline(100, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = out_dir / "cumulative_context_coverage.png"
    plt.savefig(p1)
    plt.close()
    print(f"[+] Saved: {p1.resolve()}")

    # 2. Tool Usage Frequency
    plt.figure(figsize=(11, 7), dpi=300)
    tools = report.get("tool_usage_distribution", {})
    top_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)[:14]
    tool_names = [t[0] for t in reversed(top_tools)]
    tool_counts = [t[1] for t in reversed(top_tools)]

    y_pos = np.arange(len(tool_names))
    bars2 = plt.barh(y_pos, tool_counts, color="#1f77b4", edgecolor="black", linewidth=0.8, height=0.65)

    for bar, count in zip(bars2, tool_counts):
        xval = bar.get_width()
        total_calls = sum(tools.values())
        plt.text(xval + max(tool_counts) * 0.01, bar.get_y() + bar.get_height() / 2.0, f"{count:,} ({count/total_calls*100:.1f}%)", ha="left", va="center", fontsize=9.5)

    plt.yticks(y_pos, tool_names)
    plt.xlabel("Number of Invocations")
    plt.title("Top Tool Invocations in Fable-5 Traces", pad=15)
    plt.xlim(0, max(tool_counts) * 1.18)
    plt.tight_layout()
    p2 = out_dir / "tool_usage_distribution.png"
    plt.savefig(p2)
    plt.close()
    print(f"[+] Saved: {p2.resolve()}")

    # 3. Action Output Type Breakdown (Pie Chart)
    plt.figure(figsize=(8, 6), dpi=300)
    out_types = report.get("output_type_distribution", {})
    type_labels = [f"Tool Call\n({out_types.get('tool_use', 0):,})" if k == "tool_use" else f"Final Text\n({out_types.get('text', 0):,})" for k in out_types.keys()]
    type_vals = list(out_types.values())

    plt.pie(
        type_vals,
        labels=type_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#3498db", "#2ecc71"],
        explode=(0.05, 0),
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "black", "linewidth": 1},
    )
    plt.title("Fable-5 Action Distribution: Tool Use vs. Text Output", pad=15)
    plt.tight_layout()
    p3 = out_dir / "output_type_breakdown.png"
    plt.savefig(p3)
    plt.close()
    print(f"[+] Saved: {p3.resolve()}")

    # 4. Token Length Quantiles Comparison (Boxplot-style bar chart)
    plt.figure(figsize=(10, 6), dpi=300)
    tstats = report.get("token_stats", {})
    categories = ["Context (Prompt)", "CoT (<think>)", "Target Action", "Total Sequence"]
    keys = ["context_prompt", "cot_reasoning", "target_action", "total_sequence"]

    medians = [tstats.get(k, {}).get("p50_median", 0) for k in keys]
    p90s = [tstats.get(k, {}).get("p90", 0) for k in keys]
    means = [tstats.get(k, {}).get("mean", 0) for k in keys]

    x = np.arange(len(categories))
    width = 0.25

    plt.bar(x - width, medians, width, label="Median (P50)", color="#3498db", edgecolor="black", linewidth=0.8)
    plt.bar(x, means, width, label="Mean", color="#9b59b6", edgecolor="black", linewidth=0.8)
    plt.bar(x + width, p90s, width, label="P90", color="#e67e22", edgecolor="black", linewidth=0.8)

    for i in x:
        plt.text(i - width, medians[i] + 50, f"{medians[i]:.0f}", ha="center", va="bottom", fontsize=8.5)
        plt.text(i, means[i] + 50, f"{means[i]:.0f}", ha="center", va="bottom", fontsize=8.5)
        plt.text(i + width, p90s[i] + 50, f"{p90s[i]:.0f}", ha="center", va="bottom", fontsize=8.5)

    plt.xticks(x, categories)
    plt.ylabel("Token Count (Qwen/Qwen3.8-27B)")
    plt.title("Token Length Profiling across Dialogue Components", pad=15)
    plt.legend(frameon=True)
    plt.tight_layout()
    p4 = out_dir / "token_length_profiling.png"
    plt.savefig(p4)
    plt.close()
    print(f"[+] Saved: {p4.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots from inspection report.")
    parser.add_argument("--report", type=str, default="dataset/reports/inspection_report.json", help="Path to inspection_report.json")
    parser.add_argument("--out-dir", type=str, default="dataset/reports/figures", help="Output directory for charts")

    args = parser.parse_args()
    generate_plots(args.report, args.out_dir)


if __name__ == "__main__":
    main()
