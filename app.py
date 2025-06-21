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

# 라벨 로딩 (선택)
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Spring → Flask 이미지 검증 요청 처리
@app.route("/model/events", methods=["POST"])
def validate_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))

    # 'Class 2'일 때만 유효한 이미지라고 판단하고 1 반환
    VALID_LABEL = "Class 2"
    is_valid = labels[predicted_index] == VALID_LABEL and confidence > 0.7 #confidence 수치 0.7 이상일 때 성공

    return jsonify({"valid": is_valid})

if __name__ == "__main__":
    app.run(debug=True)
