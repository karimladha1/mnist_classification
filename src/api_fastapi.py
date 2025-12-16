"""
FastAPI inference API for the MNIST classifier (logits-based + temperature scaling).

Run (from repo root):
    uvicorn src.api_fastapi:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from utils import PROJECT_ROOT, resolve_under_root

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

NUM_CLASSES = 10
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))

# Blank heuristics (tune as needed for your drawing input)
DEFAULT_BLANK_MAX_THRESHOLD = float(os.getenv("BLANK_MAX_THRESHOLD", "0.05"))
DEFAULT_BLANK_MEAN_THRESHOLD = float(os.getenv("BLANK_MEAN_THRESHOLD", "0.002"))
DEFAULT_MIN_INK_PIXELS = int(os.getenv("MIN_INK_PIXELS", "12"))
DEFAULT_INK_THRESHOLD = float(os.getenv("INK_THRESHOLD", "0.08"))

# Used only if:
# - request doesn't pass temperature, AND
# - no per-model temperature file exists.
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))

# Sidecar file naming produced by training:
#   models/<run>_best.keras -> models/<run>_best.keras.temperature.json
TEMP_SUFFIX = ".temperature.json"

# Optional: include extra metadata in predict response (debugging)
DEFAULT_RETURN_DEBUG = os.getenv("RETURN_DEBUG", "0").strip().lower() in {"1", "true", "yes"}

# Warm-load default model at startup (recommended)
DEFAULT_WARM_START = os.getenv("WARM_START", "1").strip().lower() in {"1", "true", "yes"}

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(title="MNIST Inference API (logits + temperature scaling)")

# -------------------------------------------------------------------
# Thread-safe caches
# -------------------------------------------------------------------

_lock = Lock()
_model_cache: dict[str, tf.keras.Model] = {}  # key = abs model path
_model_mtime_cache: dict[str, float] = {}  # key = abs model path -> mtime
_temp_cache: dict[str, float] = {}  # key = abs model path -> temperature

# -------------------------------------------------------------------
# Utilities: model resolution + caching
# -------------------------------------------------------------------


def _find_latest_model() -> Path:
    """Prefer *_best.keras, otherwise newest *.keras."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    best = sorted(MODELS_DIR.glob("*_best.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]

    all_models = sorted(MODELS_DIR.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_models:
        raise FileNotFoundError(f"No .keras models found in {MODELS_DIR}")

    return all_models[0]


def _resolve_model_path(model_file: str | None) -> Path:
    """
    Resolve model from:
      1) explicit request model_file (filename in models/ or absolute)
      2) MODEL_PATH env var
      3) latest model in models/ (prefers *_best.keras)
    """
    if model_file:
        p = Path(model_file).expanduser()
        if not p.is_absolute():
            p = MODELS_DIR / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Requested model not found: {p}")
        return p

    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = resolve_under_root(env_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MODEL_PATH invalid: {p}")
        return p

    return _find_latest_model()


def _load_model_cached(path: Path) -> tf.keras.Model:
    """Loads model with cache. Reloads automatically if file mtime changes."""
    abs_path = str(path.resolve())
    mtime = path.stat().st_mtime

    with _lock:
        if abs_path in _model_cache:
            cached_mtime = _model_mtime_cache.get(abs_path, -1.0)
            if abs(mtime - cached_mtime) < 1e-6:
                return _model_cache[abs_path]
            # model file changed -> reload
            _model_cache.pop(abs_path, None)
            _model_mtime_cache.pop(abs_path, None)

        # compile=False is best for inference (avoids optimizer warnings/state)
        model = tf.keras.models.load_model(path, compile=False)
        _model_cache[abs_path] = model
        _model_mtime_cache[abs_path] = mtime
        return model


# -------------------------------------------------------------------
# Temperature: sidecar + cache
# -------------------------------------------------------------------


def _temperature_sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + TEMP_SUFFIX)


def _load_temperature_for_model(model_path: Path) -> float:
    """
    Load per-model temperature from sidecar JSON if present.
    Falls back to DEFAULT_TEMPERATURE.
    """
    key = str(model_path.resolve())

    with _lock:
        if key in _temp_cache:
            return _temp_cache[key]

    sidecar = _temperature_sidecar_path(model_path)
    t = DEFAULT_TEMPERATURE

    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            t_val = data.get("temperature", data.get("T", None))
            if t_val is not None:
                t = float(t_val)
        except Exception:
            t = DEFAULT_TEMPERATURE

    # Guardrails
    t = float(np.clip(t, 0.05, 10.0))

    with _lock:
        _temp_cache[key] = t

    return t


# -------------------------------------------------------------------
# Logits -> probabilities + metrics
# -------------------------------------------------------------------


def _softmax_2d(x: np.ndarray) -> np.ndarray:
    """Stable softmax for (N, C) array."""
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    s = np.sum(ex, axis=1, keepdims=True)
    return ex / np.where(s > 0, s, 1.0)


def _logits_to_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling on logits."""
    t = float(np.clip(temperature, 0.05, 10.0))
    return _softmax_2d(logits / t)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Entropy for (N, C) probabilities."""
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _topk(probs: np.ndarray, k: int = 3) -> list[list[dict[str, Any]]]:
    """Top-k per row."""
    k = int(np.clip(k, 1, probs.shape[1]))
    idx = np.argsort(-probs, axis=1)[:, :k]
    out: list[list[dict[str, Any]]] = []
    for i in range(probs.shape[0]):
        out.append([{"class": int(c), "prob": float(probs[i, c])} for c in idx[i]])
    return out


# -------------------------------------------------------------------
# Input parsing / preprocessing
# -------------------------------------------------------------------


def _to_28x28_from_pixels(pixels: list[list[float]]) -> np.ndarray:
    arr = np.array(pixels, dtype=np.float32)
    if arr.shape != (28, 28):
        raise ValueError(f"pixels must be 28x28; got {arr.shape}")
    # If user sends 0..255, normalize
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _to_28x28_from_b64(image_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception as e:
        raise ValueError(f"image_b64 is not valid base64: {e}")

    # Use TF image decode to avoid extra deps
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)  # [H,W,1], uint8
    img = tf.image.resize(img, (28, 28), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.squeeze(img, axis=-1)  # [28,28]
    arr = img.numpy().astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _blank_metrics(x28: np.ndarray) -> dict[str, float]:
    mx = float(np.max(x28))
    mean = float(np.mean(x28))
    ink_pixels = int(np.sum(x28 > DEFAULT_INK_THRESHOLD))
    return {"max": mx, "mean": mean, "ink_pixels": float(ink_pixels)}


def _is_blank(
    x28: np.ndarray,
    blank_max_threshold: float,
    blank_mean_threshold: float,
    min_ink_pixels: int,
    ink_threshold: float,
) -> bool:
    mx = float(np.max(x28))
    mean = float(np.mean(x28))
    ink_pixels = int(np.sum(x28 > float(ink_threshold)))

    # Practical rule:
    # - If there are very few ink pixels => blank
    # - Or if BOTH max and mean are very low => blank
    if ink_pixels < int(min_ink_pixels):
        return True
    if (mx < float(blank_max_threshold)) and (mean < float(blank_mean_threshold)):
        return True
    return False


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------


class PredictRequest(BaseModel):
    # Provide either pixels or image_b64
    pixels: list[list[float]] | None = Field(
        default=None,
        description="28x28 grayscale pixels as nested list. Values in [0,1] or [0,255].",
    )
    image_b64: str | None = Field(
        default=None,
        description="Base64-encoded image (PNG/JPG). Will be converted to 28x28 grayscale.",
    )

    model_file: str | None = Field(
        default=None,
        description="Model filename inside models/ (e.g. run3_..._best.keras) or absolute path.",
    )

    temperature: float | None = Field(
        default=None,
        ge=0.05,
        le=10.0,
        description="Override temperature scaling. If omitted: uses per-model sidecar or env default.",
    )

    conf_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override confidence threshold. If omitted: uses env default.",
    )

    return_debug: bool | None = Field(
        default=None, description="If true, include extra debug fields."
    )

    topk: int = Field(default=3, ge=1, le=10, description="How many top predictions to return.")

    # Optional overrides for blank heuristics (leave None to use env defaults)
    blank_max_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    blank_mean_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_ink_pixels: int | None = Field(default=None, ge=0, le=784)
    ink_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _one_input_required(self) -> PredictRequest:
        if self.pixels is None and self.image_b64 is None:
            raise ValueError("Provide either 'pixels' (28x28) or 'image_b64'.")
        return self


class PredictResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    temperature: float
    conf_threshold: float

    is_blank: bool
    accepted: bool

    pred_class: int | None = None
    confidence: float | None = None
    entropy: float | None = None
    margin: float | None = None

    topk: list[dict[str, Any]] | None = None

    debug: dict[str, Any] | None = None


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        default_model_path = _resolve_model_path(None)
        default_temp = _load_temperature_for_model(default_model_path)
    except Exception:
        default_model_path = None
        default_temp = DEFAULT_TEMPERATURE

    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    return {
        "status": "ok",
        "models_dir": str(MODELS_DIR),
        "default_model": str(default_model_path.name) if default_model_path else None,
        "default_temperature": float(default_temp),
        "default_conf_threshold": float(DEFAULT_CONF_THRESHOLD),
        "blank_thresholds": {
            "blank_max_threshold": float(DEFAULT_BLANK_MAX_THRESHOLD),
            "blank_mean_threshold": float(DEFAULT_BLANK_MEAN_THRESHOLD),
            "min_ink_pixels": int(DEFAULT_MIN_INK_PIXELS),
            "ink_threshold": float(DEFAULT_INK_THRESHOLD),
        },
        "gpus": gpus,
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    if not MODELS_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Models directory not found: {MODELS_DIR}")

    models = sorted(MODELS_DIR.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in models:
        sidecar = _temperature_sidecar_path(p)
        out.append(
            {
                "file": p.name,
                "mtime": p.stat().st_mtime,
                "has_temperature_sidecar": sidecar.exists(),
            }
        )
    return {"models": out}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Resolve model + load
    try:
        model_path = _resolve_model_path(req.model_file)
        model = _load_model_cached(model_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Temperature resolution: request -> sidecar -> default
    if req.temperature is not None:
        temperature = float(req.temperature)
    else:
        temperature = float(_load_temperature_for_model(model_path))

    conf_threshold = (
        float(req.conf_threshold)
        if req.conf_threshold is not None
        else float(DEFAULT_CONF_THRESHOLD)
    )
    return_debug = (
        bool(req.return_debug) if req.return_debug is not None else bool(DEFAULT_RETURN_DEBUG)
    )

    # Blank override thresholds
    blank_max_threshold = (
        float(req.blank_max_threshold)
        if req.blank_max_threshold is not None
        else float(DEFAULT_BLANK_MAX_THRESHOLD)
    )
    blank_mean_threshold = (
        float(req.blank_mean_threshold)
        if req.blank_mean_threshold is not None
        else float(DEFAULT_BLANK_MEAN_THRESHOLD)
    )
    min_ink_pixels = (
        int(req.min_ink_pixels) if req.min_ink_pixels is not None else int(DEFAULT_MIN_INK_PIXELS)
    )
    ink_threshold = (
        float(req.ink_threshold) if req.ink_threshold is not None else float(DEFAULT_INK_THRESHOLD)
    )

    # Parse input -> 28x28 float32 in [0,1]
    try:
        if req.pixels is not None:
            x28 = _to_28x28_from_pixels(req.pixels)
        else:
            x28 = _to_28x28_from_b64(req.image_b64 or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input image: {e}")

    # Blank decision
    blank = _is_blank(
        x28,
        blank_max_threshold=blank_max_threshold,
        blank_mean_threshold=blank_mean_threshold,
        min_ink_pixels=min_ink_pixels,
        ink_threshold=ink_threshold,
    )

    # If blank, return early
    if blank:
        dbg = None
        if return_debug:
            dbg = {
                "blank_metrics": _blank_metrics(x28),
                "blank_thresholds": {
                    "blank_max_threshold": blank_max_threshold,
                    "blank_mean_threshold": blank_mean_threshold,
                    "min_ink_pixels": min_ink_pixels,
                    "ink_threshold": ink_threshold,
                },
            }
        return PredictResponse(
            model=model_path.name,
            temperature=temperature,
            conf_threshold=conf_threshold,
            is_blank=True,
            accepted=False,
            pred_class=None,
            confidence=None,
            entropy=None,
            margin=None,
            topk=None,
            debug=dbg,
        )

    # Prepare batch (1,28,28,1)
    x = x28[None, ..., None].astype(np.float32)

    # Forward pass -> logits (1,10)
    try:
        logits = model(x, training=False).numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise HTTPException(status_code=500, detail=f"Unexpected logits shape: {logits.shape}")

    probs = _logits_to_probs(logits, temperature=temperature)
    pred = int(np.argmax(probs, axis=1)[0])
    conf = float(np.max(probs, axis=1)[0])

    # margin = top1 - top2
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = float(sorted_probs[0, 0] - sorted_probs[0, 1]) if sorted_probs.shape[1] >= 2 else 0.0

    ent = float(_entropy(probs)[0])
    accepted = conf >= conf_threshold

    topk_list = _topk(probs, k=req.topk)[0]

    dbg = None
    if return_debug:
        dbg = {
            "logits": logits[0].tolist(),
            "probs": probs[0].tolist(),
            "blank_metrics": _blank_metrics(x28),
            "blank_thresholds": {
                "blank_max_threshold": blank_max_threshold,
                "blank_mean_threshold": blank_mean_threshold,
                "min_ink_pixels": min_ink_pixels,
                "ink_threshold": ink_threshold,
            },
        }

    return PredictResponse(
        model=model_path.name,
        temperature=temperature,
        conf_threshold=conf_threshold,
        is_blank=False,
        accepted=accepted,
        pred_class=pred,
        confidence=conf,
        entropy=ent,
        margin=margin,
        topk=topk_list,
        debug=dbg,
    )


# -------------------------------------------------------------------
# Warm start (optional)
# -------------------------------------------------------------------

if DEFAULT_WARM_START:
    try:
        p = _resolve_model_path(None)
        _ = _load_model_cached(p)
        _ = _load_temperature_for_model(p)
        # (optional) tiny warm inference
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = _model_cache[str(p.resolve())](dummy, training=False).numpy()
        print(f"[warm-start] Loaded default model: {p.name}")
    except Exception as e:
        print(f"[warm-start] Skipped: {e}")
