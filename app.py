from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the modified VGG16 model
vgg16 = VGG16()
vgg16 = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)

model = tf.keras.models.load_model('image_captioning_model.h5')  # Replace with your model path

# Load the tokenizer
tokenizer_path = 'features.pkl'  # Replace with your tokenizer path
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 33  # Replace with your max_length

def preprocess_img(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def generate_caption(img_path):
    processed_img = preprocess_img(img_path)
    features = vgg16.predict(processed_img)
    caption = predict_caption(model, features, tokenizer, max_length)
    return caption

def predict_caption(model, photo, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(img_path)
            caption = generate_caption(img_path)
            return render_template('result.html', image=img_path, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
