"""
Main training script for the MNIST classification project.

Key features:
- WSL/GPU friendly setup
- AMP (mixed precision) optional
- tf.data pipeline (cache/prefetch) for speed
- Warmup + cosine LR schedule optional
- EarlyStopping + ModelCheckpoint (best .keras)
- Logs learning rate to TensorBoard
- Training-time augmentation to better match draw_client drawings
- Fits temperature scaling on validation set and saves sidecar JSON:
    models/<run>_best.keras.temperature.json

IMPORTANT:
- Model outputs LOGITS (no softmax). Compile uses from_logits=True.
- Temperature calibration is fit on the BEST checkpoint (not the last epoch).
"""

from __future__ import annotations

# -------------------------------------------------------------------
# MUST be set before TF initializes (import tensorflow)
# -------------------------------------------------------------------
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # keep logs readable
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # keep your preference

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision, models
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.datasets import mnist

from utils import load_config, save_history_plots, save_metrics, save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOTUNE = tf.data.AUTOTUNE


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------------------------------------------------
# GPU setup
# -------------------------------------------------------------------
def configure_gpu(memory_growth: bool = True, xla: bool = False) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detected:", gpus)

    if gpus and memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"⚠️ Could not set memory growth for {gpu}: {e}")

    tf.config.optimizer.set_jit(bool(xla))
    print(f"XLA enabled: {bool(xla)}")


# -------------------------------------------------------------------
# Load MNIST
# -------------------------------------------------------------------
def load_and_preprocess_data() -> (
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., None]  # (N,28,28,1)
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    return (x_train, y_train), (x_test, y_test)


# -------------------------------------------------------------------
# Training augmentation to better match draw_client
# - Your draw_client does crop->square->resize->center shift, then thick strokes.
# - We simulate the "human variation" part: shifts/rotations/zoom + mild noise/contrast.
# -------------------------------------------------------------------
def make_augmentation_model(seed: int = 42) -> tf.keras.Model:
    # These ranges are intentionally modest; too strong hurts MNIST.
    aug_layers: list[layers.Layer] = [
        layers.RandomTranslation(0.18, 0.18, seed=seed),
        layers.RandomZoom(0.12, seed=seed),
        layers.RandomRotation(0.06, seed=seed),
    ]

    # Optional “stroke/noise” simulation: mild gaussian noise helps thick/rough lines.
    aug_layers.append(layers.GaussianNoise(0.06, seed=seed))

    # Optional contrast jitter (small). Works in float.
    # Keras doesn’t have RandomContrast in very old versions; TF 2.20 has it.
    try:
        aug_layers.append(layers.RandomContrast(0.12, seed=seed))  # type: ignore[attr-defined]
    except Exception:
        pass

    return tf.keras.Sequential(aug_layers, name="data_aug")


# -------------------------------------------------------------------
# Build model (UNCOMPILED) - outputs LOGITS
# -------------------------------------------------------------------
def build_model(seed: int = 42, use_augmentation: bool = True) -> tf.keras.Model:
    aug = make_augmentation_model(seed) if use_augmentation else None

    return models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            *([aug] if aug is not None else []),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation=None, dtype="float32"),  # logits
        ],
        name="mnist_logits_model",
    )


# -------------------------------------------------------------------
# Warm-up + Cosine LR schedule
# -------------------------------------------------------------------
def make_warmup_cosine_schedule(
    initial_lr: float,
    max_lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
) -> Callable[[int, float], float]:
    def schedule(epoch: int, lr: float) -> float:
        if epoch < warmup_epochs:
            if warmup_epochs <= 0:
                return max_lr
            progress = min((epoch + 1) / float(warmup_epochs), 1.0)
            return initial_lr + (max_lr - initial_lr) * progress

        remaining = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / float(remaining)
        progress = float(np.clip(progress, 0.0, 1.0))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine

    return schedule


# -------------------------------------------------------------------
# TensorBoard helper: log learning rate
# -------------------------------------------------------------------
class LRTensorBoard(Callback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self.writer is None:
            return

        lr_obj = self.model.optimizer.learning_rate  # type: ignore[attr-defined]
        try:
            lr_value = float(tf.keras.backend.get_value(lr_obj))
        except Exception:
            lr_value = float(lr_obj(epoch).numpy()) if callable(lr_obj) else float(lr_obj)

        with self.writer.as_default():
            tf.summary.scalar("learning_rate", lr_value, step=epoch)
        self.writer.flush()

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


# -------------------------------------------------------------------
# Temperature fitting on validation set
# Fits T by minimizing NLL of softmax(logits/T).
# -------------------------------------------------------------------
def fit_temperature(
    model: tf.keras.Model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    max_steps: int = 200,
    lr: float = 0.05,
) -> float:
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)

    logits_list = []
    labels_list = []
    for xb, yb in val_ds:
        logits = model(xb, training=False)  # logits
        logits_list.append(tf.cast(logits, tf.float32))
        labels_list.append(tf.cast(yb, tf.int32))

    logits_all = tf.concat(logits_list, axis=0)
    labels_all = tf.concat(labels_list, axis=0)

    log_T = tf.Variable(0.0, dtype=tf.float32)  # T = exp(log_T) starts at 1.0
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            T = tf.exp(log_T) + 1e-6
            scaled_logits = logits_all / T
            nll = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_all, logits=scaled_logits
                )
            )
        grads = tape.gradient(nll, [log_T])
        opt.apply_gradients(zip(grads, [log_T]))
        return nll, T

    best_nll = float("inf")
    best_T = 1.0

    for _ in range(max_steps):
        nll, T = step()
        nll_v = float(nll.numpy())
        T_v = float(T.numpy())
        if nll_v < best_nll:
            best_nll = nll_v
            best_T = T_v

    best_T = float(np.clip(best_T, 0.05, 10.0))
    print(f"✅ Temperature fit complete: T={best_T:.4f} (best val NLL={best_nll:.4f})")
    return best_T


def save_temperature_sidecar(best_model_path: Path, temperature: float) -> Path:
    sidecar = best_model_path.with_suffix(best_model_path.suffix + ".temperature.json")
    payload = {
        "temperature": float(temperature),
        "model_file": best_model_path.name,
        "timestamp": datetime.now().isoformat(),
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"✅ Saved temperature sidecar: {sidecar}")
    return sidecar


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_model(cfg: dict) -> None:
    train_cfg = cfg["train"]
    paths = cfg["paths"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    configure_gpu(
        memory_growth=bool(train_cfg.get("gpu_memory_growth", True)),
        xla=bool(train_cfg.get("xla", False)),
    )

    use_amp = bool(train_cfg.get("amp", True))
    if use_amp:
        mixed_precision.set_global_policy("mixed_float16")
        print("✅ AMP enabled: mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")
        print("ℹ️ AMP disabled: float32")
    print("Policy:", mixed_precision.global_policy())

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    base_lr = float(train_cfg.get("learning_rate", 1e-3))
    lr_mode = str(train_cfg.get("schedule", "none")).lower()

    val_split = float(train_cfg.get("validation_split", 0.1))
    if not (0.0 < val_split < 0.5):
        raise ValueError(f"validation_split must be between 0 and 0.5, got {val_split}")

    run_name = f"{train_cfg.get('run_name', 'run')}_{datetime.now():%Y%m%d-%H%M%S}"
    print(f"\n▶ Run name: {run_name}")

    print("Loading data...")
    (x_all, y_all), (x_test, y_test) = load_and_preprocess_data()

    # Deterministic split so x_val/y_val is reusable for temperature fitting
    n = x_all.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(x_train), 50_000), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    print("Building model...")
    use_aug = bool(train_cfg.get("augmentation", True))
    model = build_model(seed=seed, use_augmentation=use_aug)

    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

    # ✅ FIX: logits compile
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    model_dir = Path(paths["model_dir"])
    metrics_dir = Path(paths["metrics_dir"])
    reports_dir = Path(paths["reports_dir"])
    logs_root = Path(paths["logs_dir"])
    for d in (model_dir, metrics_dir, reports_dir, logs_root):
        d.mkdir(parents=True, exist_ok=True)

    run_logs_dir = logs_root / run_name

    early_monitor = str(train_cfg.get("early_stop_monitor", "val_accuracy"))
    ckpt_monitor = str(train_cfg.get("checkpoint_monitor", early_monitor))
    early_patience = int(train_cfg.get("early_stop_patience", 3))

    best_path = model_dir / f"{run_name}_best.keras"

    callbacks: list[Callback] = [
        TensorBoard(log_dir=str(run_logs_dir), histogram_freq=1),
        LRTensorBoard(str(run_logs_dir)),
        EarlyStopping(
            monitor=early_monitor,
            patience=early_patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(best_path),
            monitor=ckpt_monitor,
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    if lr_mode in {"cosine", "warmup_cosine"}:
        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        min_factor = float(train_cfg.get("cosine_min_factor", 0.01))
        initial_factor = float(train_cfg.get("warmup_initial_factor", 0.1))

        schedule_fn = make_warmup_cosine_schedule(
            initial_lr=base_lr * initial_factor if warmup_epochs > 0 else base_lr,
            max_lr=base_lr,
            min_lr=base_lr * min_factor,
            warmup_epochs=warmup_epochs if lr_mode == "warmup_cosine" else 0,
            total_epochs=epochs,
        )
        callbacks.append(LearningRateScheduler(schedule_fn, verbose=1))

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    save_history_plots(history, reports_dir, run_name)

    final_path = model_dir / f"{run_name}.keras"
    save_model(model, model_dir, final_path.name)

    # -------------------------------------------------------------------
    # ✅ Temperature calibration SNIPPET (exact spot)
    # -------------------------------------------------------------------
    temperature_sidecar = None
    fitted_temperature = None

    fit_temp = bool(train_cfg.get("fit_temperature", True))
    if fit_temp and best_path.exists():
        try:
            best_model = tf.keras.models.load_model(best_path)
            fitted_temperature = fit_temperature(
                best_model,
                x_val=x_val,
                y_val=y_val,
                batch_size=int(train_cfg.get("temp_batch_size", 256)),
                max_steps=int(train_cfg.get("temp_max_steps", 200)),
                lr=float(train_cfg.get("temp_lr", 0.05)),
            )
            temperature_sidecar = save_temperature_sidecar(best_path, fitted_temperature)

            writer = tf.summary.create_file_writer(str(run_logs_dir))
            with writer.as_default():
                tf.summary.scalar("calibration/temperature", fitted_temperature, step=0)
            writer.flush()
            writer.close()
        except Exception as e:
            print(f"⚠️ Temperature fitting failed (continuing): {e}")

    metrics = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": base_lr,
        "schedule": lr_mode,
        "amp_enabled": use_amp,
        "policy": str(mixed_precision.global_policy()),
        "gpu_memory_growth": bool(train_cfg.get("gpu_memory_growth", True)),
        "xla": bool(train_cfg.get("xla", False)),
        "validation_split": val_split,
        "augmentation": use_aug,
        "final_model": str(final_path),
        "best_model": str(best_path) if best_path.exists() else None,
        "temperature": float(fitted_temperature) if fitted_temperature is not None else None,
        "temperature_sidecar": str(temperature_sidecar) if temperature_sidecar else None,
        "tensorflow_version": tf.__version__,
        "gpus": [str(g) for g in tf.config.list_physical_devices("GPU")],
        "monitors": {
            "early_stop_monitor": early_monitor,
            "checkpoint_monitor": ckpt_monitor,
            "early_stop_patience": early_patience,
        },
        "config_used": train_cfg,
    }

    save_metrics(metrics, metrics_dir, f"{run_name}.json")

    print("✅ Training complete")
    print(f"Final model: {final_path}")
    if best_path.exists():
        print(f"Best model:  {best_path}")
    print(f"Logs:        {run_logs_dir}")
    if fitted_temperature is not None:
        print(f"Temperature: {fitted_temperature:.4f} (saved next to best model)")


if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    train_model(cfg)
