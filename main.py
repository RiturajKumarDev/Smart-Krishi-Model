from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
interpreter = tf.lite.Interpreter(model_path="SoilSuitabilityModel.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels
with open("crop_labels.json") as f:
    crop_labels = json.load(f)

# Scaler
with open("soil_scaler.json") as f:
    scaler = json.load(f)

mean = np.array(scaler["mean"])
std = np.array(scaler["std"])


@app.get("/")
def home():
    return {"message": "Smart Krishi API Running"}


@app.post("/predict")
async def predict(data: dict):

    features = [
        data["nitrogen"],
        data["phosphorus"],
        data["potassium"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"],
    ]

    scaled = (np.array(features) - mean) / std
    input_data = np.array([scaled], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    predicted_index = int(np.argmax(prediction[0]))
    crop_name = crop_labels[predicted_index]

    return {
        "recommended_crop": crop_name,
        "confidence": float(np.max(prediction[0])),
    }
