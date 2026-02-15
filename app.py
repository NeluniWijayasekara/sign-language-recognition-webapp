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
print("Loading model...")
model = load_model("final_cnn_bilstm_model.h5")
print("Model loaded successfully!")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
print("Label encoder loaded successfully!")

SEQUENCE_LENGTH = 40
TOTAL_FEATURES = 201

# English to Sinhala mapping
SINHALA_LABELS = {
    "Allergic": "අසාත්මිකතා",
    "Broken Hand": "අත් කැඩිලා",
    "Chest Pain": "පපුව රිදෙනවා",
    "Difficulty Breathing": "හුස්ම ගැනීමට අපහසුයි",
    "Fainting": "ක්ලාන්තය",
    "Fever": "උණ",
    "Headache": "ඔළුව රිදෙනවා",
    "Sore Throat": "උගුර රිදෙනවා",
    "Stomach Pain": "බඩ රිදෙනවා",
    "Vomit": "වමනය"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"})

    video = request.files["video"]
    
    # Save video temporarily
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    try:
        cap = cv2.VideoCapture(video_path)
        sequence = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 2nd frame to speed up
            if frame_count % 2 != 0:
                continue

            keypoints = extract_landmarks(frame)
            sequence.append(keypoints)

        cap.release()

        if len(sequence) == 0:
            return jsonify({"error": "No landmarks detected in video"})

        # Preprocess sequence
        sequence = pad_or_truncate(sequence, SEQUENCE_LENGTH)
        sequence = np.expand_dims(sequence, axis=0)  # (1, 40, 201)

        # Make prediction
        prediction = model.predict(sequence, verbose=0)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        english_label = label_encoder.inverse_transform([class_id])[0]
        sinhala_label = SINHALA_LABELS.get(english_label, english_label)

        return jsonify({
            "prediction": sinhala_label,
            "confidence": round(confidence * 100, 2),
            "english_label": english_label
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})
    
    finally:
        # Clean up - delete uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)