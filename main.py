from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os
import sys

# Try importing tflite_runtime, fallback to tensorflow if needed
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("✅ Using TensorFlow Lite")

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

def load_model_safely():
    """Safely load TFLite model with multiple path attempts"""
    
    # Try different possible paths
    possible_paths = [
        os.path.join(current_dir, "SoilSuitabilityModel.tflite"),
        os.path.join(os.getcwd(), "SoilSuitabilityModel.tflite"),
        "/opt/render/project/src/SoilSuitabilityModel.tflite"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found model at: {model_path}")
            break
    
    if model_path is None:
        print("❌ Model file not found in any location!")
        print(f"Current directory: {current_dir}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Files in current dir: {os.listdir(current_dir)}")
        raise FileNotFoundError("Model file not found!")
    
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✅ Model loaded successfully!")
        return interpreter, model_path
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Load model with error handling
try:
    interpreter, model_path = load_model_safely()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    interpreter = None
    input_details = None
    output_details = None

# 2. Labels load karo
labels_path = os.path.join(current_dir, "crop_labels.json")
try:
    with open(labels_path, "r") as f:
        crop_labels = json.load(f)
    print(f"✅ Loaded {len(crop_labels)} crop labels")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    crop_labels = []

# 3. Scaler load karo
scaler_path = os.path.join(current_dir, "soil_scaler.json")
try:
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    print(f"✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    mean = None
    std = None

@app.get("/")
def home():
    return {
        "message": "Smart Krishi API Running",
        "status": "active",
        "model_loaded": interpreter is not None,
        "num_crops": len(crop_labels),
        "environment": {
            "python_version": sys.version,
            "current_dir": current_dir,
            "files_present": os.listdir(current_dir)
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if interpreter is not None else "degraded",
        "model": "loaded" if interpreter is not None else "failed",
        "files": {
            "model": os.path.exists(model_path) if 'model_path' in locals() else False,
            "labels": os.path.exists(labels_path),
            "scaler": os.path.exists(scaler_path)
        },
        "debug": {
            "current_dir": current_dir,
            "working_dir": os.getcwd(),
            "files": os.listdir(current_dir)[:10]  # First 10 files
        }
    }

@app.post("/predict")
async def predict(data: dict):
    try:
        # Check if model is loaded
        if interpreter is None:
            return {
                "success": False,
                "error": "Model not loaded. Please check /health endpoint"
            }
        
        # Input validation
        required_fields = ["nitrogen", "phosphorus", "potassium", 
                          "temperature", "humidity", "ph", "rainfall"]
        
        for field in required_fields:
            if field not in data:
                return {"success": False, "error": f"Missing field: {field}"}
        
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
        
        # Verify input shape
        expected_shape = input_details[0]['shape']
        if input_data.shape != tuple(expected_shape):
            return {
                "success": False,
                "error": f"Input shape mismatch. Expected {expected_shape}, got {input_data.shape}"
            }
        
        # Model inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])
        
        # Results
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        if predicted_index < len(crop_labels):
            crop_name = crop_labels[predicted_index]
        else:
            crop_name = f"Unknown (index {predicted_index})"
        
        return {
            "success": True,
            "recommended_crop": crop_name,
            "confidence": round(confidence, 4),
            "crop_index": predicted_index
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# For debugging - print all files at startup
print("\n" + "="*50)
print("📁 FILES IN CURRENT DIRECTORY:")
for file in os.listdir(current_dir):
    size = os.path.getsize(os.path.join(current_dir, file)) if os.path.isfile(os.path.join(current_dir, file)) else 0
    print(f"   - {file} ({size} bytes)")
print("="*50 + "\n")