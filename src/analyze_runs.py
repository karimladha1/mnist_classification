"""
Analyze all training runs by reading JSON metrics from data/metrics.

Usage (from project root):

    python -m src.analyze_runs
"""

import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"


def load_metrics_files() -> List[Path]:
    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Metrics directory not found: {METRICS_DIR}")
    return sorted(METRICS_DIR.glob("*.json"))


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = d.get(key, default)
    # Avoid printing 'None' as 'NoneType' – keep it simple
    if value is None:
        return default
    return value


def main() -> None:
    files = load_metrics_files()
    if not files:
        print(f"No metrics JSON files found in {METRICS_DIR}")
        return

    runs: List[Dict[str, Any]] = []

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {path.name}: {e}")
            continue

        run = {
            "file": path.name,
            "run_name": safe_get(data, "run_name", path.stem),
            "schedule": safe_get(data, "schedule", "none"),
            "epochs": safe_get(data, "epochs", "?"),
            "batch_size": safe_get(data, "batch_size", "?"),
            "test_acc": safe_get(data, "test_accuracy", 0.0),
            "test_loss": safe_get(data, "test_loss", 0.0),
            "best_val_acc": safe_get(data, "best_val_accuracy", 0.0),
            "best_val_loss": safe_get(data, "best_val_loss", 0.0),
        }
        runs.append(run)

    # Sort by best validation accuracy (desc), fallback to test_acc
    runs.sort(key=lambda r: (r["best_val_acc"] or r["test_acc"]), reverse=True)

    # Pretty print table
    header = (
        f"{'#':>2}  "
        f"{'Run name':<20}  "
        f"{'Schedule':<14}  "
        f"{'Epochs':>6}  "
        f"{'Batch':>6}  "
        f"{'Test acc':>9}  "
        f"{'Val acc':>9}  "
        f"{'Test loss':>10}  "
        f"{'Val loss':>10}"
    )
    print(f"Loading runs from: {METRICS_DIR}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for idx, r in enumerate(runs, start=1):
        print(
            f"{idx:>2}  "
            f"{r['run_name']:<20.20}  "
            f"{str(r['schedule']):<14.14}  "
            f"{str(r['epochs']):>6}  "
            f"{str(r['batch_size']):>6}  "
            f"{r['test_acc']:>9.4f}  "
            f"{r['best_val_acc']:>9.4f}  "
            f"{r['test_loss']:>10.4f}  "
            f"{r['best_val_loss']:>10.4f}"
        )

    print("-" * len(header))


if __name__ == "__main__":
    main()
