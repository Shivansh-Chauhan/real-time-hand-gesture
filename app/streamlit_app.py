import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# OPTIONAL: import your model
# from src.feature_extractor import extract_features
# import tensorflow as tf
# import pickle

st.title("🤖 Real-Time Hand Gesture Recognition")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        pass

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Flip for mirror effect
        img = cv2.flip(img, 1)

        # ===== YOUR MODEL LOGIC HERE =====
        # Example:
        # features = extract_features(img)
        # prediction = model.predict(...)
        # cv2.putText(img, "Gesture: A", (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img


webrtc_streamer(
    key="gesture",
    video_processor_factory=VideoProcessor,
)
