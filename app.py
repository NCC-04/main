import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Potato__Early_Blight', 'Potato__Late_Blight', 'Potato__Healthy']

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)  # model already rescales
    return image

def predict_image(image):
    image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = float(np.max(output_data) * 100)
    return class_names[predicted_class], confidence

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            img = Image.open(request.files["file"].stream).convert("RGB")
        elif "image" in request.files:
            img = Image.open(request.files["image"].stream).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400

        label, confidence = predict_image(img)
        return jsonify({"label": label, "confidence": f"{confidence:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
