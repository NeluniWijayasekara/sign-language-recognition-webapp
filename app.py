from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

from mediapipe_utils import extract_landmarks
from preprocessing import pad_or_truncate

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load trained model & encoder
# -----------------------------
model = load_model("final_cnn_bilstm_model.h5")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

SEQUENCE_LENGTH = 40
TOTAL_FEATURES = 201

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"})

    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_landmarks(frame)
        sequence.append(keypoints)

    cap.release()
    os.remove(video_path)

    if len(sequence) == 0:
        return jsonify({"error": "No landmarks detected"})

    sequence = pad_or_truncate(sequence, SEQUENCE_LENGTH)
    sequence = np.expand_dims(sequence, axis=0)  # (1, 40, 201)

    prediction = model.predict(sequence, verbose=0)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    label = label_encoder.inverse_transform([class_id])[0]

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
