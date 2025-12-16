from pathlib import Path

import tensorflow as tf

m = sorted(Path("models").glob("*_best.keras"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
print("Model:", m.name)
model = tf.keras.models.load_model(m, compile=False)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = x_test[:16].astype("float32") / 255.0
x = x[..., None]  # (N,28,28,1)

logits = model(x, training=False).numpy()
pred = logits.argmax(axis=1)

print("y_true:", y_test[:16].tolist())
print("y_pred:", pred.tolist())
print("acc@16:", float((pred == y_test[:16]).mean()))
