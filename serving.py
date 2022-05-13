import cv2
import uuid
import os
from flask import Flask, request, jsonify
#from src.classifier import Classifier
from src.antigener_detection.antigener_detector import antigener_classification_update_1, Searcher
app = Flask(__name__)


#classifier = Classifier.from_pretrained('')
cache_dir = '.cache_image'
os.makedirs(cache_dir, exist_ok=True)

searcher, id_map = Searcher()


def map_result(result):
    new_result = {
        "positive": [],
        "negative": []
    }
    for item in result.get('positive', []):
        new_result['positive'].append(float(item[-1]))
    for item in result.get('negative', []):
        new_result['negative'].append(float(item[-1]))
    return new_result 


@app.route('/antigen-images', methods=['POST'])
def hello():
    image = request.files['antigen']
    file = os.path.join(cache_dir, f'{str(uuid.uuid4())}-{image.filename}')
    image.save(file)
    #result = classifier.predict(file)
    img = cv2.imread(file)
    result = antigener_classification_update_1(img, searcher, id_map)
    result = map_result(result)
    return jsonify(dict(code=200, data=result))

app.run(host='0.0.0.0', port=8089)
