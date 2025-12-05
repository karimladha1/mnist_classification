import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np


def load_and_preprocess_data():
    """Load MNIST dataset and normalize pixel values."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize (0–255 → 0–1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Define a simple neural network for MNIST classification."""
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
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


def train_model(model, x_train, y_train, epochs=5):
    """Train the model on MNIST data."""
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    return history


def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model on test data."""
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}")
    return acc


def make_predictions(model, x_test, y_test, num_samples=5):
    """Print predictions for a few example images."""
    sample_images = x_test[:num_samples]
    sample_labels = y_test[:num_samples]

    preds = model.predict(sample_images)
    pred_labels = np.argmax(preds, axis=1)

    print("\nSample Predictions:")
    print("Predicted:", pred_labels)
    print("Actual:   ", sample_labels)


def check_gpu():
    """Display whether GPU is available."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU Detected: {gpus}")
    else:
        print("⚠ Using CPU (no GPU detected)")


def main():
    print("\n=== MNIST Classification (TensorFlow) ===\n")

    check_gpu()
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model()
    model.summary()

    print("\nTraining model...\n")
    train_model(model, x_train, y_train, epochs=5)

    print("\nEvaluating model...\n")
    evaluate_model(model, x_test, y_test)

    make_predictions(model, x_test, y_test)


if __name__ == "__main__":
    main()
