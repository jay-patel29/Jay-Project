from tensorflow.keras.models import load_model
import joblib

# Load your trained model
model = load_model("cnn_model.h5")

# Save as .pkl
joblib.dump(model, "model.pkl")

print("Model converted successfully!")
