import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

max_length = 34

# load tokenizer
tokenizer = pickle.load(open("tokenizer.p","rb"))

# load caption model
model = load_model("caption_model.keras")

# load feature extractor
xception_model = Xception(weights='imagenet', include_top=False, pooling='avg')


def extract_features(filename):
    
    image = load_img(filename, target_size=(299,299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = xception_model.predict(image, verbose=0)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(photo):
    
    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    caption = in_text.replace("startseq","").replace("endseq","")
    return caption