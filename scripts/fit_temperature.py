from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def load_mnist_val(val_size: int = 10000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a validation set from MNIST train split (like Keras validation_split does),
    but done explicitly so calibration is stable/reproducible.
    """
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train[..., None]  # (N,28,28,1)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(x_train))
    rng.shuffle(idx)

    val_idx = idx[:val_size]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    return x_val, y_val


def stable_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-12, 1.0))


def nll_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    # probs shape: (N,10), y shape: (N,)
    p_true = probs[np.arange(len(y)), y]
    return float(-np.mean(stable_log(p_true)))


def apply_temperature_to_probs(probs: np.ndarray, T: float) -> np.ndarray:
    """
    Temperature scaling using probabilities:
      logits ~= log(p)
      logits' = logits / T
      p' = softmax(logits')
    """
    logits = stable_log(probs)
    T = max(1e-6, float(T))
    logits = logits / T
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature_grid(
    probs: np.ndarray, y: np.ndarray, t_min=0.5, t_max=5.0, steps=91
) -> tuple[float, float]:
    """
    Simple, robust grid search. Good enough for MNIST and avoids optimizer headaches.
    """
    Ts = np.linspace(t_min, t_max, steps)
    best_T = 1.0
    best_nll = float("inf")

    for T in Ts:
        p_cal = apply_temperature_to_probs(probs, T)
        nll = nll_from_probs(p_cal, y)
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    return best_T, best_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .h5 model file")
    ap.add_argument(
        "--out",
        default="temperature.json",
        help="Where to write temperature JSON (default: temperature.json)",
    )
    ap.add_argument("--val-size", type=int, default=10000, help="Validation size (default: 10000)")
    ap.add_argument("--seed", type=int, default=42, help="Seed for val split (default: 42)")
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=91)
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Loading MNIST validation set...")
    x_val, y_val = load_mnist_val(val_size=args.val_size, seed=args.seed)

    print("Predicting probabilities...")
    probs = model.predict(x_val, batch_size=512, verbose=1)
    if probs.shape[-1] != 10:
        raise ValueError(f"Expected 10 classes; got shape {probs.shape}")

    base_nll = nll_from_probs(probs, y_val)
    base_acc = float((np.argmax(probs, axis=1) == y_val).mean())

    print(f"Base:  acc={base_acc:.4f}  nll={base_nll:.6f}")

    best_T, best_nll = fit_temperature_grid(
        probs, y_val, t_min=args.t_min, t_max=args.t_max, steps=args.steps
    )

    probs_cal = apply_temperature_to_probs(probs, best_T)
    cal_nll = nll_from_probs(probs_cal, y_val)
    cal_acc = float((np.argmax(probs_cal, axis=1) == y_val).mean())

    print(f"Best T={best_T:.3f}")
    print(f"Cal:   acc={cal_acc:.4f}  nll={cal_nll:.6f}")

    out_path = Path(args.out).expanduser().resolve()
    payload = {
        "temperature": best_T,
        "base": {"acc": base_acc, "nll": base_nll},
        "calibrated": {"acc": cal_acc, "nll": cal_nll},
        "model": str(model_path),
        "val_size": args.val_size,
        "seed": args.seed,
        "grid": {"t_min": args.t_min, "t_max": args.t_max, "steps": args.steps},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()
