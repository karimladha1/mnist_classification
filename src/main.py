import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
from .utils import load_config, save_model, save_metrics, make_timestamped_name


# ------------------------------------------------
# Load + Preprocess MNIST
# ------------------------------------------------
def load_and_preprocess_data():
    """Load MNIST dataset and normalize pixel values."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images (0–255 → 0–1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for CNNs (28,28 → 28,28,1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


# ------------------------------------------------
# Build the MNIST model
# ------------------------------------------------
def build_model():
    """Create a simple MNIST classifier."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------
# Training Function
# ------------------------------------------------
def train_model(config):
    """Train the MNIST model using settings from config.yaml."""
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print("Building model...")
    model = build_model()

    print("Training...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=config["train"]["epochs"],
        batch_size=config["train"]["batch_size"],
        verbose=1
    )

    print("Evaluating...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # ----- Save model with timestamp -----
    model_filename = make_timestamped_name("mnist_model", "h5")
    save_model(model, config["paths"]["model_dir"], model_filename)

    # ----- Save metrics with timestamp (based on run_name) -----
    metrics = {
        "accuracy": test_acc,
        "loss": float(test_loss),
        "epochs": config["train"]["epochs"],
    }

    metrics_filename = make_timestamped_name(config["run_name"], "json")
    save_metrics(metrics, config["paths"]["metrics_dir"], metrics_filename)


    print("Training complete.")
    print("Model saved to:", config["paths"]["model_dir"])
    print("Metrics saved to:", config["paths"]["metrics_dir"])


# ------------------------------------------------
# MAIN ENTRYPOINT
# ------------------------------------------------
if __name__ == "__main__":
    print("Loading config.yaml...")
    config = load_config("config.yaml")

    train_model(config)

