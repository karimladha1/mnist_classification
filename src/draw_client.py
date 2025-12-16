from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox

import numpy as np
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageOps, ImageTk

API_URL = "http://127.0.0.1:8000/predict"

# --- knobs (client defaults; server also has defaults) ---
CONF_THRESHOLD = 0.60  # below this => UNKNOWN (server enforces too)
BLANK_MAX_THRESHOLD = 0.03  # blank/too faint rejection (server enforces too)

# Optional: let client override temperature (normally leave None and use server sidecar)
# Set to a float (e.g., 1.15) to force it, or None to use server auto temperature.
CLIENT_TEMPERATURE_OVERRIDE: float | None = None


def entropy(probs: list[float]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def margin_top2(probs: list[float]) -> tuple[float, int, float, int, float]:
    p = np.asarray(probs, dtype=np.float64)
    idx = np.argsort(-p)
    a, b = int(idx[0]), int(idx[1])
    pa, pb = float(p[a]), float(p[b])
    return pa - pb, a, pa, b, pb


def center_of_mass_shift(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    total = arr.sum()
    if total <= 1e-6:
        return img

    ys, xs = np.indices(arr.shape)
    cy = float((ys * arr).sum() / total)
    cx = float((xs * arr).sum() / total)

    target_x = arr.shape[1] / 2.0
    target_y = arr.shape[0] / 2.0
    dx = target_x - cx
    dy = target_y - cy

    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def mnist_preprocess(pil_img: Image.Image) -> tuple[np.ndarray | None, dict, Image.Image | None]:
    """
    Returns:
      flat784 (or None),
      debug_info,
      preview_28 (PIL Image, 28x28) or None
    """
    debug: dict = {}

    img = pil_img.convert("L")
    arr0 = np.array(img, dtype=np.float32) / 255.0
    debug["raw_min"] = float(arr0.min())
    debug["raw_max"] = float(arr0.max())
    debug["raw_mean"] = float(arr0.mean())

    # fast blank check (client-side)
    if float(arr0.max()) < BLANK_MAX_THRESHOLD:
        return None, debug, None

    # crop bounding box using simple threshold
    thresh = 0.10
    mask = arr0 > thresh
    if not mask.any():
        return None, debug, None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = img.crop((x0, y0, x1, y1))

    # pad to square
    w, h = cropped.size
    side = max(w, h)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    squared = ImageOps.expand(
        cropped,
        border=(pad_x, pad_y, side - w - pad_x, side - h - pad_y),
        fill=0,
    )

    # resize to 20x20 then paste into 28x28 with 4px margin (classic MNIST-like)
    resized20 = squared.resize((20, 20), Image.Resampling.LANCZOS)
    canvas28 = Image.new("L", (28, 28), 0)
    canvas28.paste(resized20, (4, 4))

    centered = center_of_mass_shift(canvas28)

    arr = np.array(centered, dtype=np.float32) / 255.0
    debug["proc_min"] = float(arr.min())
    debug["proc_max"] = float(arr.max())
    debug["proc_mean"] = float(arr.mean())

    return arr.reshape(-1), debug, centered


class DigitClient:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST Digit Drawing Client")

        # --- layout frames ---
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        left = tk.Frame(top)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        right = tk.Frame(top)
        right.pack(side=tk.LEFT, padx=10, pady=10)

        # --- drawing canvas ---
        self.canvas_size = 280
        self.brush_size = 18

        self.canvas = tk.Canvas(
            left, width=self.canvas_size, height=self.canvas_size, bg="black", cursor="cross"
        )
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_img = ImageDraw.Draw(self.image)

        # --- preview (28x28) ---
        tk.Label(right, text="Model input preview (28×28)", font=("Arial", 11)).pack(pady=(0, 6))
        self.preview_label = tk.Label(right)
        self.preview_label.pack()
        self._preview_tk = None  # keep ref

        # --- buttons ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, pady=(0, 5))

        tk.Button(btn_frame, text="Clear", width=12, command=self.clear_canvas).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="Predict", width=12, command=self.predict_digit).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="Quit", width=12, command=root.destroy).pack(side=tk.LEFT, padx=5)

        # --- labels ---
        self.pred_label = tk.Label(root, text="Prediction: ?", font=("Arial", 14))
        self.pred_label.pack(pady=2)

        self.conf_label = tk.Label(
            root, text=f"Confidence: ? (threshold {CONF_THRESHOLD:.2f})", font=("Arial", 12)
        )
        self.conf_label.pack(pady=2)

        self.unc_label = tk.Label(root, text="Uncertainty: ?", font=("Arial", 11))
        self.unc_label.pack(pady=(0, 6))

        self.meta_label = tk.Label(root, text="Model: ? | Temp: ?", font=("Arial", 10))
        self.meta_label.pack(pady=(0, 8))

        # --- probability chart ---
        self.fig = Figure(figsize=(5.2, 2.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Class probabilities (from API)")
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels([str(i) for i in range(10)])
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_ylabel("p(class)")
        self.ax.set_xlabel("digit")
        self.bars = self.ax.bar(range(10), [0.0] * 10)

        self.prob_canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.prob_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=False, padx=10, pady=(0, 10)
        )

    def draw(self, event):
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw_img.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_img = ImageDraw.Draw(self.image)

        self.pred_label.config(text="Prediction: ?")
        self.conf_label.config(text=f"Confidence: ? (threshold {CONF_THRESHOLD:.2f})")
        self.unc_label.config(text="Uncertainty: ?")
        self.meta_label.config(text="Model: ? | Temp: ?")
        self.update_prob_chart([0.0] * 10)
        self.update_preview(None)

    def update_preview(self, img28: Image.Image | None):
        if img28 is None:
            self.preview_label.config(image="")
            self._preview_tk = None
            return

        big = img28.resize((140, 140), Image.Resampling.NEAREST)
        big = ImageOps.autocontrast(big)

        self._preview_tk = ImageTk.PhotoImage(big)
        self.preview_label.config(image=self._preview_tk)

    def predict_digit(self):
        pixels, dbg, preview28 = mnist_preprocess(self.image)

        print(
            "[client] raw(min,max,mean)=",
            dbg.get("raw_min"),
            dbg.get("raw_max"),
            dbg.get("raw_mean"),
            "| proc(min,max,mean)=",
            dbg.get("proc_min"),
            dbg.get("proc_max"),
            dbg.get("proc_mean"),
        )

        self.update_preview(preview28)

        if pixels is None:
            messagebox.showwarning(
                "No digit", "Canvas is blank (or too faint). Draw a digit more boldly."
            )
            return

        payload = {
            "pixels": pixels.tolist(),
            "conf_threshold": CONF_THRESHOLD,
            "blank_max_threshold": BLANK_MAX_THRESHOLD,
        }
        if CLIENT_TEMPERATURE_OVERRIDE is not None:
            payload["temperature"] = float(CLIENT_TEMPERATURE_OVERRIDE)

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
        except Exception as e:
            messagebox.showerror("Error", f"Could not reach API:\n{e}")
            return

        if resp.status_code != 200:
            messagebox.showerror("API error", f"Status {resp.status_code}:\n{resp.text}")
            return

        data = resp.json()

        probs = data.get("probabilities")
        pred = data.get("predicted_class")
        conf = float(data.get("confidence", 0.0))
        reason = data.get("reason")
        model_file = data.get("model_file", "?")
        temp_used = data.get("temperature_used", None)

        if probs is None or len(probs) != 10:
            messagebox.showerror("API error", "Response missing a valid 'probabilities' array.")
            return

        # uncertainty metrics on API-provided probs (already calibrated if your API is)
        ent = entropy(probs)
        margin, a, pa, b, pb = margin_top2(probs)

        if pred is None:
            # server decided blank/low confidence
            self.pred_label.config(text=f"Prediction: UNKNOWN ({reason})")
        else:
            self.pred_label.config(text=f"Prediction: {pred}")

        self.conf_label.config(text=f"Confidence: {conf*100:.2f}% (threshold {CONF_THRESHOLD:.2f})")
        self.unc_label.config(
            text=(
                f"Uncertainty: entropy={ent:.3f} nats (max~{math.log(10):.3f}), "
                f"top2={a}:{pa:.3f} vs {b}:{pb:.3f}, margin={margin:.3f}"
            )
        )
        self.meta_label.config(text=f"Model: {model_file} | Temp used: {temp_used}")

        self.update_prob_chart(probs)

        print(
            f"[client] pred={pred} conf={conf:.4f} reason={reason} "
            f"temp_used={temp_used} ent={ent:.4f} margin={margin:.4f}"
        )

    def update_prob_chart(self, probs):
        for rect, p in zip(self.bars, probs):
            rect.set_height(p)
        self.ax.set_ylim(0.0, 1.0)
        self.prob_canvas.draw_idle()


def main():
    root = tk.Tk()
    DigitClient(root)
    root.mainloop()


if __name__ == "__main__":
    main()
