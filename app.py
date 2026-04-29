import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# Load Model (SAFE PATH)
# -------------------------------
@st.cache_resource
def load_cnn_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "cnn_model.h5")
    model = load_model(model_path)
    return model

model = load_cnn_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("✍️ Handwritten Digit Recognition (CNN)")
st.write("Draw a digit (0–9) below and click Predict")

# Canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data

        # Convert to grayscale
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)

        # Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # Invert colors (white digit on black background)
        img = 255 - img

        # Normalize
        img = img / 255.0

        # Reshape for CNN (IMPORTANT)
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first!")
