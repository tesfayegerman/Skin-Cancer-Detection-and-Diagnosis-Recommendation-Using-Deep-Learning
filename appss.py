from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('best_model_balanced.keras', compile=False)
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
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    result = {cls: float(round(float(p), 6)) for cls, p in zip(CLASSES, preds)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)