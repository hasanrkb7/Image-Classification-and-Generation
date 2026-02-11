# ============================================================
# Task 4-4 — Multi-Label Prediction Flask API
# ============================================================

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ---- Config ----
BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-4_MultiLabel_Service"
MODEL_PATH = os.path.join(BASE_DIR, "model", "vehicle_multilabel.h5")
IMG_SIZE = 224
LABELS = ["bike", "bus", "car"]
THRESHOLD = 0.5

# ---- Load model ----
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)
CORS(app)

def preprocess_image(file_storage):
    img = Image.open(file_storage).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use form-data key 'file'."}), 400

    file = request.files["file"]
    x = preprocess_image(file)

    probs = model.predict(x)[0].tolist()
    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    predicted = [LABELS[i] for i in range(len(LABELS)) if probs[i] >= THRESHOLD]

    return jsonify({
        "threshold": THRESHOLD,
        "predicted_labels": predicted,
        "probabilities": prob_dict
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Task 4-4 Multi-Label API running. Use POST /predict"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)