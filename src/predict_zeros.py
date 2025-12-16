import requests
import tensorflow as tf

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
img = (x_test[0].astype("float32") / 255.0).tolist()
label = int(y_test[0])

r = requests.post("http://127.0.0.1:8000/predict", json={"pixels": img, "return_debug": False})
print("true:", label)
print(r.json())
