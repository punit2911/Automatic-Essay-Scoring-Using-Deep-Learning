from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
import keras.backend as K
import pytesseract
from PIL import Image
import io

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Ensure you set the path to the Tesseract executable if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Download NLTK resources if not done already
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing functions
def sent2word(x):
    stop_words = set(stopwords.words('english')) 
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    filtered_sentence = [] 
    words = x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = []
    for i in raw:
        if len(i) > 0:
            final_words.append(sent2word(i))
    return final_words

def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)  # Updated for newer Gensim versions
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[i])        
    if noOfWords > 0:  # Avoid division by zero
        vec = np.divide(vec, noOfWords)
    return vec

def getVecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for c, i in enumerate(essays):
        essay_vecs[c] = makeVec(i, model, num_features)
    return essay_vecs

def convertToVec(text):
    content = text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("best_model_for_deployment.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))

# Route to render the mainpage.html
@app.route('/')
def main_page():
    return render_template('mainpage.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def score_essay():  # Renamed to avoid endpoint conflict
    K.clear_session()
    final_text = request.get_json()["text"]  # Get text from JSON
    score = convertToVec(final_text)  # Call the conversion function
    K.clear_session()
    return jsonify({'score': score}), 201

# Route to handle image uploads
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check for valid image formats
    if file and (file.filename.endswith('.png') or file.filename.endswith('.jpg') or file.filename.endswith('.jpeg')):
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image)
        return jsonify({'text': text}), 200

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)

