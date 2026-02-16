#!/usr/bin/env python3
"""
Quick Compare: Fast CLI comparison of two or more evaluation result files.

Prints a side-by-side summary table of key metrics without generating charts.

Usage:
    python quick_compare.py results/eval1.json results/eval2.json
    python quick_compare.py results/eval1.json results/eval2.json results/eval3.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_result(file_path: str) -> Dict[str, Any]:
    """Load a single evaluation result JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_method(data: Dict[str, Any], file_path: str) -> str:
    """Detect evaluation method from result data."""
    if "rag_config" in data or Path(file_path).name.startswith("rag_"):
        return "RAG"
    return "Direct"


def format_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.2f}%"


def quick_compare(file_paths: List[str]) -> None:
    """Print a quick comparison table for the given result files."""
    results = []
    for fp in file_paths:
        data = load_result(fp)
        method = detect_method(data, fp)
        provider = data.get("provider", "unknown")
        model = data.get("model", "unknown")
        avg = data.get("average_metrics", {})
        errors = data.get("error_analysis", {})

        results.append({
            "file": Path(fp).name,
            "method": method,
            "provider": provider,
            "model": model,
            "recall": avg.get("recall", 0),
            "precision": avg.get("precision", 0),
            "f1": avg.get("f1", 0),
            "exact_match_rate": avg.get("exact_match_rate", 0),
            "mAP": avg.get("mAP", 0),
            "samples": avg.get("evaluated_samples", 0),
            "time": avg.get("elapsed_time", 0),
            "api_errors": errors.get("api_errors", 0),
            "empty_responses": errors.get("empty_responses", 0),
        })

    # Print header
    print()
    print("=" * 90)
    print("QUICK COMPARISON")
    print("=" * 90)

    # Determine column width
    col_w = max(30, max(len(r["file"]) for r in results) + 2)
    header_labels = [r["file"] for r in results]

    # Row formatter
    def row(label: str, values: List[str]) -> str:
        cells = "".join(v.rjust(col_w) for v in values)
        return f"  {label:<22}{cells}"

    print(row("", header_labels))
    print("  " + "-" * (22 + col_w * len(results)))

    # Model info
    print(row("Method", [r["method"] for r in results]))
    print(row("Provider", [r["provider"] for r in results]))
    print(row("Model", [r["model"] for r in results]))
    print(row("Samples", [str(r["samples"]) for r in results]))
    print()

    # Metrics
    print(row("Recall", [format_pct(r["recall"]) for r in results]))
    print(row("Precision", [format_pct(r["precision"]) for r in results]))
    print(row("F1", [format_pct(r["f1"]) for r in results]))
    print(row("Exact Match Rate", [format_pct(r["exact_match_rate"]) for r in results]))
    print(row("mAP", [format_pct(r["mAP"]) for r in results]))
    print()

    # Errors & time
    print(row("Time (s)", [f"{r['time']:.1f}" for r in results]))
    print(row("API Errors", [str(r["api_errors"]) for r in results]))
    print(row("Empty Responses", [str(r["empty_responses"]) for r in results]))

    print("=" * 90)

    # Highlight best
    if len(results) > 1:
        best_f1_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
        print(f"\n  Best F1: {results[best_f1_idx]['file']} ({format_pct(results[best_f1_idx]['f1'])})")

        best_recall_idx = max(range(len(results)), key=lambda i: results[i]["recall"])
        if best_recall_idx != best_f1_idx:
            print(f"  Best Recall: {results[best_recall_idx]['file']} ({format_pct(results[best_recall_idx]['recall'])})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Quick comparison of evaluation result JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_compare.py results/eval_direct.json results/eval_rag.json
  python quick_compare.py results/*.json
  python quick_compare.py results/   # compare all .json under results/
        """,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more evaluation result JSON files (supports wildcards like results/*.json or directories).",
    )
    args = parser.parse_args()

    # Expand globs + directories (so Windows CMD can use * patterns)
    expanded_files: List[str] = []
    for item in args.files:
        p = Path(item)

        # If it's a directory, include all .json files under it
        if p.exists() and p.is_dir():
            expanded_files.extend(str(x) for x in sorted(p.glob("*.json")))
            continue

        # Otherwise: expand wildcard patterns (even if path doesn't exist yet)
        matches = sorted(Path().glob(item))
        if matches:
            expanded_files.extend(str(m) for m in matches)
        else:
            # Keep original item so we can show a useful error
            expanded_files.append(item)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in expanded_files:
        if f not in seen:
            deduped.append(f)
            seen.add(f)

    # Validate existence, and filter only existing files
    valid_files = [f for f in deduped if Path(f).exists() and Path(f).is_file()]
    missing = [f for f in deduped if not (Path(f).exists() and Path(f).is_file())]

    if not valid_files:
        print("Error: No valid files matched your inputs.")
        if missing:
            print("Tried:")
            for m in missing:
                print(f"  - {m}")
        sys.exit(1)

    if missing:
        print("Warning: Some inputs did not match any file and will be ignored:")
        for m in missing:
            print(f"  - {m}")
        print()

    quick_compare(valid_files)


if __name__ == "__main__":
    main()
