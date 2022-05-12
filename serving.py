import cv2
import uuid
import os
from flask import Flask, request, jsonify
from src.classifier import Classifier
app = Flask(__name__)


classifier = Classifier.from_pretrained('')
cache_dir = '.cache_image'
os.makedirs(cache_dir, exist_ok=True)


@app.route('/antigen-images', methods=['POST'])
def hello():
    image = request.files['antigen']
    file = os.path.join(cache_dir, f'{str(uuid.uuid4())}-{image.filename}')
    image.save(file)
    result = classifier.predict(file)
    return jsonify(dict(code=200, data=result))

app.run(host='0.0.0.0', port=8089)
