#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Big comparison figure (2x3) for Empty Response Rate only.

Empty Response Rate = empty_responses / total_samples
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
# 🔧 CONFIG（固定，不用 CLI）
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

OUT_PNG = OUTPUT_DIR / "empty_response_rate.png"
OUT_PDF = OUTPUT_DIR / "empty_response_rate.pdf"

# =========================
# Utils
# =========================

def load_json(fp: str) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_model_name(model: str) -> str:
    return {
    }.get(model, model)

def load_panel(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    rows = []

    for fp in files:
        data = load_json(fp)
        avg = data.get("average_metrics", {})
        err = data.get("error_analysis", {})

        total = avg.get("total_samples", avg.get("evaluated_samples", 0))
        empty = err.get("empty_responses", 0)

        if total == 0:
            continue

        empty_rate = empty / total

        rows.append({
            "model": normalize_model_name(data.get("model", "unknown")),
            "empty_rate": empty_rate,
        })

    return pd.DataFrame(rows)

def build_color_map(models: List[str]) -> Dict[str, Tuple]:
    cmap = plt.get_cmap("tab20")
    uniq = list(dict.fromkeys(models))
    return {m: cmap(i % cmap.N) for i, m in enumerate(uniq)}

# =========================
# Plot
# =========================

def plot_big_figure():
    fig, axes = plt.subplots(2, 3, figsize=(22, 10), dpi=200)
    fig.subplots_adjust(
        left=0.05, right=0.995,
        top=0.92, bottom=0.12,
        wspace=0.18, hspace=0.35
    )

    all_models = []
    panel_dfs = []

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

        x = np.arange(len(df))
        bars = ax.bar(
            x,
            df["empty_rate"],
            color=[color_map[m] for m in df["model"]],
            width=0.7
        )

        # annotate
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        if idx % 3 == 0:
            ax.set_ylabel("Empty Response Rate")

    fig.suptitle(
        "Empty Response Rate Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # ---- Add timestamp ----
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
    plot_big_figure()