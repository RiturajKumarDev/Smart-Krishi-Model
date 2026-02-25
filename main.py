from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tflite_runtime.interpreter as tflite
import json
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= MODEL LOADING WITH ERROR HANDLING =============
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Model load karo
model_path = os.path.join(current_dir, "SoilSuitabilityModel.tflite")
print(f"🔍 Loading model from: {model_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()  # 🔥 IMPORTANT: Ye line missing thi!

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded successfully!")
print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# 2. Labels load karo
labels_path = os.path.join(current_dir, "crop_labels.json")
print(f"🔍 Loading labels from: {labels_path}")

if not os.path.exists(labels_path):
    raise FileNotFoundError(f"❌ Labels file not found at: {labels_path}")

with open(labels_path, "r") as f:
    crop_labels = json.load(f)

print(f"✅ Loaded {len(crop_labels)} crop labels")

# 3. Scaler load karo
scaler_path = os.path.join(current_dir, "soil_scaler.json")
print(f"🔍 Loading scaler from: {scaler_path}")

if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ Scaler file not found at: {scaler_path}")

with open(scaler_path, "r") as f:
    scaler = json.load(f)

mean = np.array(scaler["mean"], dtype=np.float32)
std = np.array(scaler["std"], dtype=np.float32)

print(f"✅ Scaler loaded successfully!")

@app.get("/")
def home():
    return {
        "message": "Smart Krishi API Running",
        "status": "active",
        "model_loaded": True,
        "num_crops": len(crop_labels)
    }

@app.post("/predict")
async def predict(data: dict):
    try:
        # Input validation
        required_fields = ["nitrogen", "phosphorus", "potassium", 
                          "temperature", "humidity", "ph", "rainfall"]
        
        for field in required_fields:
            if field not in data:
                return {"error": f"Missing field: {field}"}
        
        # Features extract karo
        features = [
            float(data["nitrogen"]),
            float(data["phosphorus"]),
            float(data["potassium"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
        ]
        
        # Scale karo
        features_array = np.array(features, dtype=np.float32)
        scaled = (features_array - mean) / std
        input_data = np.expand_dims(scaled, axis=0)  # (7,) -> (1, 7)
        
        # Model inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])
        
        # Results
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        crop_name = crop_labels[predicted_index]
        
        return {
            "success": True,
            "recommended_crop": crop_name,
            "confidence": round(confidence, 4),
            "crop_index": predicted_index
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded",
        "files": {
            "model": os.path.exists(model_path),
            "labels": os.path.exists(labels_path),
            "scaler": os.path.exists(scaler_path)
        }
    }