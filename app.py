import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load model safely
@st.cache_resource
def load_cnn_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "cnn_model.h5")
    return load_model(model_path)

model = load_cnn_model()

st.title("Handwritten Digit Recognition")

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data

        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        digit = np.argmax(pred)

        st.success(f"Predicted Digit: {digit}")
