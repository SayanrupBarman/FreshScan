import os
import json
import numpy as np
import requests
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from flask_compress import Compress
from PIL import Image
import io

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__)
Compress(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.onnx")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")

IMG_SIZE = (256, 256)
FRUIT_NAMES = ["Apple", "Banana", "Orange"]

# ThingSpeak config (read-only key)
THINGSPEAK_CHANNEL = os.environ.get("THINGSPEAK_CHANNEL", "3147420")
THINGSPEAK_API_KEY = os.environ.get("THINGSPEAK_API_KEY", "6S9MVJI76UCQYOKT")

# ── Load ONNX model & threshold once at startup ─────────────
print("Loading ONNX model …")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"Model loaded. Input: {input_name}, Outputs: {output_names}")

with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["best_threshold"]
print(f"Threshold: {THRESHOLD}")


# ── Helpers ──────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Match training pipeline: decode → resize 256×256 → scale [0,1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 256, 256, 3)


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    return app.send_static_file("sw.js")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()
    img = preprocess_image(image_bytes)
    preds = session.run(output_names, {input_name: img})

    # preds[0] = freshness (1,1), preds[1] = fruit (1,3)
    freshness_prob = float(preds[0][0][0])
    fruit_idx = int(np.argmax(preds[1][0]))
    fruit_name = FRUIT_NAMES[fruit_idx]
    freshness_label = "Fresh" if freshness_prob > THRESHOLD else "Rotten"

    return jsonify({
        "fruit": fruit_name,
        "freshness_prob": round(freshness_prob * 100, 2),
        "freshness_label": freshness_label,
        "threshold": round(THRESHOLD * 100, 2),
    })


@app.route("/sensor", methods=["GET"])
def sensor():
    fruit = request.args.get("fruit", "").lower()
    if fruit not in ("apple", "banana", "orange"):
        return jsonify({"error": "Invalid or missing fruit parameter"}), 400

    url = (
        f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL}"
        f"/feeds/last.json?api_key={THINGSPEAK_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return jsonify({"error": f"ThingSpeak request failed: {e}"}), 502

    sensor_data = {
        "temperature": float(data.get("field1", 0)),
        "humidity": float(data.get("field2", 0)),
        "air_quality": float(data.get("field3", 0)),
        "banana_score": float(data.get("field4", 0)),
        "apple_score": float(data.get("field5", 0)),
        "orange_score": float(data.get("field6", 0)),
        "recorded_at": data.get("created_at", ""),
    }

    score_key = f"{fruit}_score"
    sensor_score = sensor_data.get(score_key, 0)

    return jsonify({**sensor_data, "sensor_score": sensor_score})


@app.route("/combine", methods=["POST"])
def combine():
    body = request.get_json(silent=True) or {}
    ml_score = body.get("ml_score")
    sensor_score = body.get("sensor_score")

    if ml_score is None or sensor_score is None:
        return jsonify({"error": "ml_score and sensor_score required"}), 400

    final_score = 0.6 * float(ml_score) + 0.4 * float(sensor_score)
    return jsonify({"final_score": round(final_score, 2)})


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
