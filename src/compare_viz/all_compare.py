#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a single big multi-panel figure (2x3) with:
- consistent model colors across all subplots
- ONE shared legend for the whole figure
- larger fonts / readable layout
- output to a single large PNG/PDF

Assumes each panel loads multiple evaluation JSON files (same schema as your visualizer).
"""

import json
import glob
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- Config --------------------
# Metrics shown in each subplot (keep style similar to your grouped bars)
METRICS = ["recall", "precision", "f1", "exact_match_rate"]

# Titles for 6 panels (2 rows x 3 cols)
PANEL_TITLES = [
    "(a) Exp 1: Task 1 (Chain-to-Firms)",
    "(c) Exp 1: Task 2 (Firm-to-Chains)",
    "(e) Exp 1: Task 3 (Competitors)",
    "(b) Exp 2: Task 1 (Chain-to-Firms)",
    "(d) Exp 2: Task 2 (Firm-to-Chains)",
    "(f) Exp 2: Task 3 (Competitors)",
]

# >>>> YOU ONLY NEED TO EDIT THIS <<<<
# Each entry is a glob pattern to load JSONs for that panel.
# Example patterns you might use:
#   "results/evaluation_results_chain_firms_qa_local_*.json"
#   "results/rag_evaluation_results_chain_firms_qa_local_*.json"
PANEL_PATTERNS = [
    "results/evaluation_results_chain_firms*.json",
    "results/evaluation_results_firm_chain*.json",
    "results/evaluation_results_competitors*.json",
    "results/rag_evaluation_results_chain_firms*.json",
    "results/rag_evaluation_results_firm_chain*.json",
    "results/rag_evaluation_results_competitors*.json",
]

# Optional: normalize ugly model names -> short names shown on legend
MODEL_NAME_MAP = {
    # "hf.co/unsloth/gpt-oss-20b-GGUF:F32": "gpt-oss 20b",
    # add more if needed
}

# Output file
OUT_PNG = r"results\comparisons\all_comparison.png"
OUT_PDF = r"results\comparisons\all_comparison.pdf"

PANEL_METRICS = [
    ["recall", "precision", "f1", "exact_match_rate", "mAP"],  # (a) Exp1 Task1
    ["recall", "precision", "f1", "exact_match_rate", "mAP"],  # (b) Exp1 Task2
    ["recall", "precision", "f1", "exact_match_rate", "mAP"],  # (c) Exp1 Task3
    ["recall", "precision", "f1", "exact_match_rate"],         # (d) Exp2 Task1
    ["recall", "precision", "f1", "exact_match_rate"],         # (e) Exp2 Task2
    ["recall", "precision", "f1", "exact_match_rate"],         # (f) Exp2 Task3
]


# -------------------- Loader --------------------
@dataclass
class LoadedResult:
    display: str     # what will show in legend
    provider: str
    model: str
    method: str
    metrics: Dict[str, float]


def detect_method(data: Dict[str, Any], filename: str) -> str:
    if "graphrag_config" in data:
        return "GraphRAG"
    if "rag_config" in data or Path(filename).name.startswith("rag_"):
        return "RAG"
    return "Direct"


def normalize_model_name(model: str) -> str:
    return MODEL_NAME_MAP.get(model, model)


def load_results_from_glob(pattern: str) -> List[LoadedResult]:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"⚠️ No files matched: {pattern}")
        return []

    out: List[LoadedResult] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        provider = data.get("provider", "unknown")
        raw_model = data.get("model", "unknown")
        model = normalize_model_name(raw_model)
        method = detect_method(data, fp)

        avg = data.get("average_metrics", {}) or {}
        metrics = {
            "recall": float(avg.get("recall", 0.0)),
            "precision": float(avg.get("precision", 0.0)),
            "f1": float(avg.get("f1", 0.0)),
            "exact_match_rate": float(avg.get("exact_match_rate", 0.0)),
            "mAP": float(avg.get("mAP", 0.0)),
            "avg_time_per_sample": float(avg.get("avg_time_per_sample", 0.0)),
        }

        # display name: you can customize here if you want
        # e.g. "RAG:gpt-oss 20b" or just "gpt-oss 20b"
        display = model

        out.append(
            LoadedResult(
                display=display,
                provider=provider,
                model=model,
                method=method,
                metrics=metrics,
            )
        )
    return out


def to_df(results: List[LoadedResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "display": r.display,
            "provider": r.provider,
            "model": r.model,
            "method": r.method,
            **r.metrics
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # stable order (optional): by display name
        df = df.sort_values("display").reset_index(drop=True)
    return df


# -------------------- Plotting --------------------
def build_global_color_map(all_model_names: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Assign consistent colors to model names across all subplots.
    Uses matplotlib tab20 / tab10-like cycle.
    """
    cmap = plt.get_cmap("tab20")
    unique = list(dict.fromkeys(all_model_names))  # preserve order, unique
    color_map = {}
    for i, name in enumerate(unique):
        color_map[name] = cmap(i % cmap.N)
    return color_map

METRIC_LABEL_MAP = {
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "exact_match_rate": "Exact Match Rate",
    "mAP": "mAP",
}
def plot_grouped_bars(ax, df: pd.DataFrame, metrics: List[str], color_map, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return []

    n_models = len(df)
    x = np.arange(len(metrics))
    width = 0.82 / max(n_models, 1)

    handles, labels = [], []

    # best[m] = (best_value, best_x_center, best_y)
    best = {m: (-1.0, None, None) for m in metrics}

    for i, (_, row) in enumerate(df.iterrows()):
        name = row["display"]
        vals = [float(row.get(m, 0.0)) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width

        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=color_map.get(name, (0.5, 0.5, 0.5, 1.0)),
            edgecolor="none",
            label=name,
        )

        # legend handle once
        if name not in labels:
            handles.append(bars[0])
            labels.append(name)

        # update best per metric
        for j, (b, v) in enumerate(zip(bars, vals)):
            m = metrics[j]
            if v > best[m][0]:
                cx = b.get_x() + b.get_width() / 2
                cy = b.get_height()
                best[m] = (v, cx, cy)

    # ===== Annotation phase =====
    for j, m in enumerate(metrics):
        v, cx, cy = best[m]

        # Case 1: all zero → put "0.00" at center
        if v <= 0.0:
            ax.text(
                x[j],              # metric group center
                0.02,              # slightly above baseline
                "0.00",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
        # Case 2: normal → annotate top-1
        else:
            ax.text(
                cx,
                cy + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_LABEL_MAP.get(m, m) for m in metrics],
        fontsize=10
    )
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Score", fontsize=10)

    return list(zip(handles, labels))


def main():
    # Load all panels
    panel_dfs: List[pd.DataFrame] = []
    panel_results: List[List[LoadedResult]] = []

    for pat in PANEL_PATTERNS:
        res = load_results_from_glob(pat)
        panel_results.append(res)
        panel_dfs.append(to_df(res))

    # Build global color map from ALL models appearing in ANY panel
    all_names = []
    for df in panel_dfs:
        if not df.empty:
            all_names.extend(df["display"].tolist())
    color_map = build_global_color_map(all_names)

    # Big figure
    fig, axes = plt.subplots(
        2, 3,
        figsize=(22, 10),  # bigger => readable in LaTeX
        dpi=200,
        # constrained_layout=True
    )

    # Collect legend items globally
    legend_dict = {}  # name -> handle

    for idx, ax in enumerate(axes.flat):
        title = PANEL_TITLES[idx] if idx < len(PANEL_TITLES) else f"Panel {idx+1}"
        df = panel_dfs[idx] if idx < len(panel_dfs) else pd.DataFrame()
        metrics_this_panel = PANEL_METRICS[idx]
        pairs = plot_grouped_bars(ax, df, metrics_this_panel, color_map, title)

        for h, name in pairs:
            legend_dict[name] = h  # ensure one per model

        # Remove y-label from inner plots (optional cleaner look)
        if idx % 3 != 0:
            ax.set_ylabel("")

    # One shared legend (bottom)
    legend_names = list(legend_dict.keys())
    legend_handles = [legend_dict[n] for n in legend_names]

    fig.legend(
        legend_handles,
        legend_names,
        loc="lower center",
        # ncol=min(6, max(1, len(legend_names))),
        ncol=5,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01)
    )

    # Add a suptitle (optional)
    fig.suptitle("Multi-Task Result Comparison", fontsize=16, fontweight="bold", y=0.95)

    # Save (PDF recommended for LaTeX; PNG for quick view)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = Path(OUT_PNG)
    pdf_path = Path(OUT_PDF)

    png_with_time = png_path.with_name(f"{png_path.stem}_{timestamp}{png_path.suffix}")
    pdf_with_time = pdf_path.with_name(f"{pdf_path.stem}_{timestamp}{pdf_path.suffix}")

    # Ensure output directory exists
    png_with_time.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_with_time, bbox_inches="tight")
    fig.savefig(pdf_with_time, bbox_inches="tight")

    print(f"✓ Saved: {png_with_time}")
    print(f"✓ Saved: {pdf_with_time}")


if __name__ == "__main__":
    main()