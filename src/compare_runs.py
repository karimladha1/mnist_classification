# src/compare_runs.py
"""
Compare multiple MNIST training runs by reading JSON metrics files.

Usage:
    python -m src.compare_runs
    python -m src.compare_runs --sort accuracy
    python -m src.compare_runs --sort loss
    python -m src.compare_runs --sort epochs
    python -m src.compare_runs --metrics-dir data/metrics
"""

import argparse
import json
import os
from typing import List, Dict, Any


def load_runs(metrics_dir: str) -> List[Dict[str, Any]]:
    """Load all *.json metric files from metrics_dir."""
    runs = []
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(metrics_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        run_name = os.path.splitext(fname)[0]
        runs.append(
            {
                "name": run_name,
                "file": fname,
                "accuracy": float(data.get("accuracy", 0.0)),
                "loss": float(data.get("loss", 0.0)),
                "epochs": int(data.get("epochs", 0)),
            }
        )
    return runs


def sort_runs(runs: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    """Sort runs according to the requested field."""
    if sort_by is None:
        return runs

    # For accuracy & epochs we want highest first, for loss lowest first,
    # for name/file we use normal ascending order.
    reverse = sort_by in {"accuracy", "epochs"}
    key = lambda r: r[sort_by]
    return sorted(runs, key=key, reverse=reverse)


def print_table(runs: List[Dict[str, Any]], metrics_dir: str, sort_by: str | None):
    print(f"Loading runs from: {metrics_dir}")
    print("-" * 76)
    print(f"{'#':<3} {'Run name':<20} {'File':<30} {'Accuracy':>8} {'Loss':>8} {'Epochs':>6}")
    print("-" * 76)

    for idx, r in enumerate(runs, start=1):
        print(
            f"{idx:<3} "
            f"{r['name']:<20} "
            f"{r['file']:<30} "
            f"{r['accuracy']:>8.4f} "
            f"{r['loss']:>8.4f} "
            f"{r['epochs']:>6d}"
        )

    print("-" * 76)
    if sort_by:
        print(f"(sorted by {sort_by})")


def main():
    parser = argparse.ArgumentParser(description="Compare MNIST training runs.")
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="data/metrics",
        help="Directory containing run*.json metric files (default: data/metrics)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["name", "file", "accuracy", "loss", "epochs"],
        help="Sort runs by this field",
    )
    args = parser.parse_args()

    runs = load_runs(args.metrics_dir)
    if not runs:
        print(f"No JSON metrics found in {args.metrics_dir}")
        return

    runs = sort_runs(runs, args.sort)
    print_table(runs, args.metrics_dir, args.sort)


if __name__ == "__main__":
    main()
