from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# 모델 로딩
interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 라벨 로딩 (옵션)
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# 응답 형식
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).resize((224, 224))  # Teachable Machine 기본 입력 크기
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))

# 응답 형식 지정
    return jsonify({
        "prediction": labels[predicted_index] if labels else predicted_index,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
