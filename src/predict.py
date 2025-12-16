from pathlib import Path

import numpy as np
import tensorflow as tf

# Locate latest *_best.keras
models_dir = Path("models")
model_path = sorted(models_dir.glob("*_best.keras"), key=lambda p: p.stat().st_mtime, reverse=True)[
    0
]

print("Loading:", model_path.name)

model = tf.keras.models.load_model(model_path, compile=False)

# Test on a blank MNIST-like input
x = np.zeros((1, 28, 28, 1), dtype=np.float32)
logits = model(x, training=False).numpy()

print("logits shape:", logits.shape)
print("top class:", logits.argmax(axis=1)[0])
print("max logit:", logits.max())
