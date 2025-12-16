"""
Utility helpers for the MNIST classification project.

Used by:
- src.main (training)
- src.api_fastapi (inference)

Goals:
- Robust path handling (works from repo root or any cwd)
- Safe YAML loading
- Atomic JSON saving for metrics
- Stable, safe model saving paths
- Training curve plots (PNG) into reports/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------


def _detect_project_root(start: Path) -> Path:
    """
    Walk upward from 'start' looking for common repo markers.
    Falls back to 'start' if nothing found.
    """
    start = start.resolve()
    markers = ("config.yaml", "pyproject.toml", ".git")
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return start


# utils.py location might be:
# - <repo_root>/utils.py  (your current layout)
# - <repo_root>/src/utils.py (common alternative)
PROJECT_ROOT: Path = _detect_project_root(Path(__file__).resolve().parent)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return it as a Path."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_under_root(p: str | Path) -> Path:
    """
    Resolve a path under PROJECT_ROOT unless it is already absolute.
    If absolute, it must still be under PROJECT_ROOT (prevents ../ traversal).
    """
    p = Path(p).expanduser()
    candidate = p if p.is_absolute() else (PROJECT_ROOT / p)

    # Normalize/resolve without requiring the file to exist
    candidate = candidate.resolve(strict=False)

    # Enforce: must be inside repo root
    root = PROJECT_ROOT.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes PROJECT_ROOT: {candidate} (root={root})")

    return candidate


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """
    Load YAML config.
    - If config_path is relative, interpreted as PROJECT_ROOT / config_path.
    - Raises FileNotFoundError if missing.
    - Raises ValueError if YAML is empty or not a dict.
    """
    cfg_path = resolve_under_root(config_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {cfg_path}")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/dict, got {type(cfg)} in {cfg_path}")

    return cfg


# ---------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text to a file: write to temp -> replace."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    """JSON serializer for common non-JSON types."""
    try:
        import numpy as np  # optional

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)

    # Last resort
    return str(obj)


# ---------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------


def save_model(model: Any, model_dir: str | Path, filename: str = "mnist_model.keras") -> Path:
    """
    Save a Keras model under model_dir (relative to PROJECT_ROOT unless absolute).
    Defaults to native Keras format (.keras). Accepts .h5 if you really want legacy.

    Returns full path.
    """
    model_dir_path = resolve_under_root(model_dir)
    ensure_dir(model_dir_path)

    full_path = model_dir_path / filename

    # If no suffix, default to .keras
    if full_path.suffix == "":
        full_path = full_path.with_suffix(".keras")

    # Only allow known formats here
    if full_path.suffix not in (".keras", ".h5"):
        raise ValueError(f"Unsupported model extension: {full_path.suffix} (use .keras or .h5)")

    model.save(full_path)
    print(f"[utils] Saved model to: {full_path}")
    return full_path


# ---------------------------------------------------------------------
# Metrics saving
# ---------------------------------------------------------------------


def save_metrics(metrics: dict[str, Any], metrics_dir: str | Path, filename: str | Path) -> Path:
    """
    Save metrics dict as JSON in metrics_dir (relative to PROJECT_ROOT unless absolute).
    Uses atomic write to avoid partial/corrupt files.
    """
    metrics_dir_path = resolve_under_root(metrics_dir)
    ensure_dir(metrics_dir_path)

    out_path = metrics_dir_path / Path(filename)

    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")

    text = json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    _atomic_write_text(out_path, text + "\n")
    print(f"[utils] Saved metrics to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------


def make_timestamped_name(base: str, ext: str) -> str:
    """Create a filesystem-safe filename with a timestamp."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ext = ext.lstrip(".")
    return f"{base}_{ts}.{ext}"


# ---------------------------------------------------------------------
# Save training curves
# ---------------------------------------------------------------------


def save_history_plots(history: Any, reports_dir: str | Path, run_name: str) -> Path:
    """
    Save accuracy and loss curves (train + validation) to a PNG file in reports_dir.
    history: Keras History object from model.fit()
    Returns output path.
    """
    reports_dir_path = resolve_under_root(reports_dir)
    ensure_dir(reports_dir_path)

    hist = getattr(history, "history", {}) or {}
    acc = hist.get("accuracy", [])
    val_acc = hist.get("val_accuracy", [])
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Accuracy
    ax = axes[0]
    if acc:
        ax.plot(acc, label="train_accuracy")
    if val_acc:
        ax.plot(val_acc, label="val_accuracy")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    # Loss
    ax = axes[1]
    if loss:
        ax.plot(loss, label="train_loss")
    if val_loss:
        ax.plot(val_loss, label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.tight_layout()

    out_path = reports_dir_path / f"{run_name}_history.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"[utils] Saved training curves to: {out_path}")
    return out_path


__all__ = [
    "PROJECT_ROOT",
    "ensure_dir",
    "resolve_under_root",
    "load_config",
    "save_model",
    "save_metrics",
    "make_timestamped_name",
    "save_history_plots",
]
