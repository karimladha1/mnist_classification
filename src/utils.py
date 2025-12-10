import os
import json
from pathlib import Path
from datetime import datetime

import yaml  # make sure pyyaml is installed in your venv


# -------------------------------------------------
# Paths
# -------------------------------------------------

# Project root = one level above this file (…/mnist_classification)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Config
# -------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """
    Load YAML config from the project root.

    Default: <project_root>/config.yaml
    """
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Model saving
# -------------------------------------------------

def save_model(model, model_dir: str | Path, filename: str = "mnist_model.h5") -> str:
    """
    Save a Keras model under models/ directory (by default).
    Returns the full path as a string.
    """
    model_dir = PROJECT_ROOT / model_dir
    ensure_dir(model_dir)

    full_path = model_dir / filename
    model.save(full_path)
    print(f"[utils] Saved model to: {full_path}")
    return str(full_path)


# -------------------------------------------------
# Metrics saving
# -------------------------------------------------

def save_metrics(metrics: dict, metrics_dir: str | Path, filename: str = "run1.json") -> str:
    """
    Save metrics dictionary as JSON under data/metrics/ (by default).
    """
    metrics_dir = PROJECT_ROOT / metrics_dir
    ensure_dir(metrics_dir)

    full_path = metrics_dir / filename
    with open(full_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[utils] Saved metrics to: {full_path}")
    return str(full_path)


# ---------------------------------------------------
# Time stamp
# ----------------------------------------------------
def make_timestamped_name(base: str, ext: str) -> str:
    """
    Create a filesystem-safe filename with a timestamp.

    Example:
      make_timestamped_name("run1", "json")
      -> "run1_20251209-135945.json"
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{ts}.{ext}"
