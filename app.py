from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model once at startup
print("Loading model...")
model = tf.keras.models.load_model('Resnet50_best_final_model.h5')

print("Model loaded successfully!")

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Preprocess image exactly as during training
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)

    # Run inference
    preds = model.predict(arr)[0]  # shape: (7,)

    # Return probabilities as JSON
    result = {cls: float(round(float(p), 6)) for cls, p in zip(CLASSES, preds)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
