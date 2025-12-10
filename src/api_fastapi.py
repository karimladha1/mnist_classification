from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

# ----------------------------------------------------
# Load model once at startup
# ----------------------------------------------------
MODEL_PATH = os.path.join("models", "mnist_model.h5")

print(f"[api] Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("[api] Model loaded.")


# ----------------------------------------------------
# Pydantic models
# ----------------------------------------------------
class PixelsPayload(BaseModel):
    pixels: list[float]  # 784 values


class CanvasPayload(BaseModel):
    image_base64: str  # "data:image/png;base64,...."


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def preprocess_pixels(pixels: list[float]) -> np.ndarray:
    """Convert a flat 784-element list into a (1, 28, 28, 1) tensor."""
    arr = np.array(pixels, dtype="float32")
    arr = arr.reshape(28, 28)
    arr = arr / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)
    return arr


def preprocess_canvas(base64_str: str) -> np.ndarray:
    """
    Convert base64 PNG from <canvas> to (1, 28, 28, 1) tensor.
    We convert to grayscale, resize to 28x28, invert colors
    (black background, white digit), then normalize.
    """
    # Strip header: "data:image/png;base64,...."
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img).astype("float32")

    # Invert colors: canvas is usually black bg, white drawing
    # If your drawing looks inverted, comment this line out.
    arr = 255.0 - arr

    arr = arr / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)
    return arr


def predict_from_tensor(tensor: np.ndarray) -> dict:
    """Run model prediction and return digit + confidence."""
    probs = model.predict(tensor, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return {"prediction": digit, "confidence": confidence}


# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="MNIST Classification API")


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "message": "MNIST API is up. POST to /predict with 784 pixels or use /draw to draw a digit."
    }


# --------- JSON pixels endpoint (existing style) --------------
@app.post("/predict", response_class=JSONResponse)
def predict(payload: PixelsPayload):
    if len(payload.pixels) != 784:
        return JSONResponse(
            status_code=400,
            content={"error": "Expected 784 pixel values (28x28)."},
        )

    tensor = preprocess_pixels(payload.pixels)
    result = predict_from_tensor(tensor)
    return result


# --------- Canvas image endpoint (for draw UI) ---------------
@app.post("/predict_canvas", response_class=JSONResponse)
def predict_canvas(payload: CanvasPayload):
    try:
        tensor = preprocess_canvas(payload.image_base64)
        result = predict_from_tensor(tensor)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process image: {e}"},
        )


# --------- Draw UI page --------------------------------------
@app.get("/draw", response_class=HTMLResponse)
def draw_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNIST Draw Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #111;
                color: #eee;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }
            h1 {
                margin-bottom: 5px;
            }
            #canvas-container {
                margin-top: 20px;
            }
            canvas {
                border: 2px solid #fff;
                background: #000;
                cursor: crosshair;
            }
            .controls {
                margin-top: 15px;
            }
            button {
                margin: 0 8px;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
                font-size: 14px;
                cursor: pointer;
            }
            #predict-btn { background: #4CAF50; color: white; }
            #clear-btn { background: #f44336; color: white; }
            #result {
                margin-top: 20px;
                font-size: 20px;
            }
        </style>
    </head>
    <body>
        <h1>MNIST Draw Demo</h1>
        <p>Draw a digit (0–9) below, then click <strong>Predict</strong>.</p>

        <div id="canvas-container">
            <canvas id="canvas" width="280" height="280"></canvas>
        </div>

        <div class="controls">
            <button id="clear-btn">Clear</button>
            <button id="predict-btn">Predict</button>
        </div>

        <div id="result"></div>

        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const clearBtn = document.getElementById('clear-btn');
            const predictBtn = document.getElementById('predict-btn');
            const resultDiv = document.getElementById('result');

            // Canvas setup
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 18;
            ctx.lineCap = 'round';

            let drawing = false;

            function getPos(e) {
                const rect = canvas.getBoundingClientRect();
                if (e.touches) {
                    e = e.touches[0];
                }
                return {
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top
                };
            }

            canvas.addEventListener('mousedown', (e) => {
                drawing = true;
                const pos = getPos(e);
                ctx.beginPath();
                ctx.moveTo(pos.x, pos.y);
            });

            canvas.addEventListener('mousemove', (e) => {
                if (!drawing) return;
                const pos = getPos(e);
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
            });

            canvas.addEventListener('mouseup', () => drawing = false);
            canvas.addEventListener('mouseleave', () => drawing = false);

            // Touch support
            canvas.addEventListener('touchstart', (e) => {
                e.preventDefault();
                drawing = true;
                const pos = getPos(e);
                ctx.beginPath();
                ctx.moveTo(pos.x, pos.y);
            });

            canvas.addEventListener('touchmove', (e) => {
                e.preventDefault();
                if (!drawing) return;
                const pos = getPos(e);
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
            });

            canvas.addEventListener('touchend', (e) => {
                e.preventDefault();
                drawing = false;
            });

            clearBtn.addEventListener('click', () => {
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                resultDiv.textContent = '';
            });

            predictBtn.addEventListener('click', async () => {
                const dataURL = canvas.toDataURL('image/png');

                resultDiv.textContent = 'Predicting...';

                try {
                    const response = await fetch('/predict_canvas', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_base64: dataURL })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        const digit = data.prediction;
                        const conf = (data.confidence * 100).toFixed(2);
                        resultDiv.textContent = `Prediction: ${digit} (confidence: ${conf}%)`;
                    } else {
                        resultDiv.textContent = 'Error: ' + (data.error || 'Unknown error');
                    }
                } catch (err) {
                    resultDiv.textContent = 'Request failed: ' + err;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
