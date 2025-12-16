from pathlib import Path

import tensorflow as tf

m = sorted(Path("models").glob("*_best.keras"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
print("Model:", m.name)
model = tf.keras.models.load_model(m, compile=False)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = (x_test.astype("float32") / 255.0)[..., None]

logits = model(x, training=False).numpy()
pred = logits.argmax(axis=1)

acc = (pred == y_test).mean()
print("MNIST test accuracy:", float(acc))
