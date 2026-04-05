import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import tensorflow as tf
import pickle

# Fix import path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extractor import extract_features

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/gesture_model_best.h5")
    scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))
    encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
    return model, scaler, encoder

model, scaler, encoder = load_model()

# ===== WEBCAM PROCESSOR =====
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # ===== MODEL PREDICTION =====
        features = extract_features(img)

        if features is not None:
            features = scaler.transform([features])
            prediction = model.predict(features)

            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            label = encoder.inverse_transform([class_id])[0]

            cv2.putText(img, f"{label} ({confidence:.2f})",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        return img

# ===== STREAMLIT UI =====
st.title("🤖 Real-Time Hand Gesture Recognition")

webrtc_streamer(
    key="gesture",
    video_processor_factory=VideoProcessor,
)
