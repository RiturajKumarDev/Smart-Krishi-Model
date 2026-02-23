from flask import Flask, request, render_template
import numpy as np
import tflite_runtime.interpreter as tflite
import json

# Load TFLite model
interpreter = tflite.Interpreter(
    model_path="SoilSuitabilityModel.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load crop labels
with open("crop_labels.json") as f:
    crop_labels = json.load(f)

# Load scaler
with open("soil_scaler.json") as f:
    scaler = json.load(f)

mean = np.array(scaler["mean"])
std = np.array(scaler["std"])

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]

    # 🔥 APPLY NORMALIZATION
    scaled = (np.array(features) - mean) / std
    input_data = np.array([scaled], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    predicted_index = int(np.argmax(prediction[0]))
    crop_name = crop_labels[predicted_index]

    return render_template(
        "index.html", prediction_text=f"Recommended Crop: {crop_name}"
    )


if __name__ == "__main__":
    app.run(debug=True)
