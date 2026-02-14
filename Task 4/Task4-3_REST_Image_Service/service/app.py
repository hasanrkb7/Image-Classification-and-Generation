# Flask REST API (Image Upload → Prediction)

import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Paths
BASE_DIR = r"E:\JYU\Deep-Learning for Cognitive Computing for Developers\Task 4\Task4-3_REST_Image_Service"
MODEL_PATH = os.path.join(BASE_DIR, "model", "vehicle_classifier_mobilenetv2_patched.h5")

# Class labels (must match training class_indices)
CLASS_NAMES = ["bike", "bus", "car"]

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)
CORS(app)

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr

@app.get("/")
def home():
    return jsonify({
        "message": "Vehicle classifier API is running",
        "endpoint": "/predict (POST image file with key='file')"
    })

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data key: file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        x = preprocess_image(file)
        preds = model.predict(x, verbose=0)[0]
        pred_index = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_index]
        confidence = float(preds[pred_index])

        return jsonify({
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(len(CLASS_NAMES))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)