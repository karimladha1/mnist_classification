import requests
import tensorflow as tf

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

i = 0
img = x_test[i].astype("float32") / 255.0
img = img * 0.3  # fade it

payload = {"pixels": img.tolist()}
r = requests.post("http://127.0.0.1:8000/predict", json=payload)

print("true:", int(y_test[i]))
print(r.json())
