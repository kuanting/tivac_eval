#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Big 2x3 scatter plot for F1 vs Avg Time per Sample.

- X-axis: Avg Time per Sample (seconds)
- Y-axis: F1 score
- Same PANEL_PATTERNS as previous figures
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIG（與前面一致）
# =========================

PANEL_PATTERNS = [
    "results/evaluation_results_chain_firms*.json",
    "results/evaluation_results_firm_chain*.json",
    "results/evaluation_results_competitors*.json",
    "results/rag_evaluation_results_chain_firms*.json",
    "results/rag_evaluation_results_firm_chain*.json",
    "results/rag_evaluation_results_competitors*.json",
]

PANEL_TITLES = [
    "(a) Exp 1: Task 1 (Chain-to-Firms)",
    "(c) Exp 1: Task 2 (Firm-to-Chains)",
    "(e) Exp 1: Task 3 (Competitors)",
    "(b) Exp 2: Task 1 (Chain-to-Firms)",
    "(d) Exp 2: Task 2 (Firm-to-Chains)",
    "(f) Exp 2: Task 3 (Competitors)",
]

OUTPUT_DIR = Path("results/comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "f1_vs_time.png"
OUT_PDF = OUTPUT_DIR / "f1_vs_time.pdf"

# =========================
# Utils
# =========================

def load_json(fp: str) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_method(data: Dict, filename: str) -> str:
    if "graphrag_config" in data:
        return "GraphRAG"
    if "rag_config" in data or Path(filename).name.startswith("rag_"):
        return "RAG"
    return "Direct"

def normalize_model_name(model: str) -> str:
    return {
        "hf.co/unsloth/gpt-oss-20b-GGUF:F32": "gpt-oss 20b",
    }.get(model, model)

def load_panel(pattern: str) -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(pattern)):
        data = load_json(fp)
        avg = data.get("average_metrics", {})

        rows.append({
            "model": normalize_model_name(data.get("model", "unknown")),
            "method": detect_method(data, fp),
            "f1": avg.get("f1", 0.0),
            "time": avg.get("avg_time_per_sample", 0.0),
        })

    return pd.DataFrame(rows)

def build_color_map(models: List[str]) -> Dict[str, Tuple]:
    cmap = plt.get_cmap("tab20")
    uniq = list(dict.fromkeys(models))
    return {m: cmap(i % cmap.N) for i, m in enumerate(uniq)}

# =========================
# Plot
# =========================

def plot_big_scatter():
    fig, axes = plt.subplots(2, 3, figsize=(22, 10), dpi=200)
    fig.subplots_adjust(
        left=0.06, right=0.995,
        top=0.92, bottom=0.12,
        wspace=0.18, hspace=0.35
    )

    panel_dfs = []
    all_models = []

    for pat in PANEL_PATTERNS:
        df = load_panel(pat)
        panel_dfs.append(df)
        if not df.empty:
            all_models.extend(df["model"].tolist())

    color_map = build_color_map(all_models)

    for idx, ax in enumerate(axes.flat):
        df = panel_dfs[idx]
        ax.set_title(PANEL_TITLES[idx], fontsize=12, fontweight="bold")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        for _, row in df.iterrows():
            ax.scatter(
                row["time"],
                row["f1"],
                color=color_map[row["model"]],
                marker="o",              # ✅ 全部圓形
                s=90,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )

        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)

        if idx % 3 == 0:
            ax.set_ylabel("F1 Score")
        if idx >= 3:
            ax.set_xlabel("Avg Time per Sample (s)")

    # ===== Legends =====
    # Color legend (models)
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[m], markeredgecolor="black",
                   markersize=9, label=m)
        for m in color_map
    ]

    # Marker legend (methods)
    method_handles = [
    ]

    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=9
    )

    fig.suptitle(
        "F1 Score vs Inference Time Trade-off",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = OUT_PNG.with_name(f"{OUT_PNG.stem}_{timestamp}{OUT_PNG.suffix}")
    pdf_path = OUT_PDF.with_name(f"{OUT_PDF.stem}_{timestamp}{OUT_PDF.suffix}")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")

# =========================
# Main
# =========================

if __name__ == "__main__":
    plot_big_scatter()