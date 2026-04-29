import streamlit as st
import numpy as np
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# App Title
st.title("Handwritten Digit Recognition")
st.write("Enter pixel values to predict digit")

# Input (example: 784 inputs for MNIST)
input_data = []

for i in range(28):
    row = st.text_input(f"Row {i+1} (comma-separated 28 values)", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    row_values = list(map(int, row.split(",")))
    input_data.extend(row_values)

# Convert to numpy
input_array = np.array(input_data).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_array)
    st.success(f"Predicted Digit: {prediction[0]}")
