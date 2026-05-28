import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# -----------------------------
# Load Tokenizer
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

tokenizer = load_tokenizer()

# -----------------------------
# Load Caption Model
# -----------------------------
@st.cache_resource
def load_caption_model():
    model = load_model("best_model.keras")
    return model

model = load_caption_model()

# -----------------------------
# Load VGG16 Feature Extractor
# -----------------------------
@st.cache_resource
def load_feature_extractor():

    base_model = VGG16(weights='imagenet')

    feature_extractor = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )

    return feature_extractor

feature_extractor = load_feature_extractor()

# -----------------------------
# Max Caption Length
# -----------------------------
max_length = 34

# -----------------------------
# Index to Word
# -----------------------------
def idx_to_word(integer, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == integer:
            return word

    return None

# -----------------------------
# Extract Features
# -----------------------------
def extract_features(image):

    image = image.resize((224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = feature_extractor.predict(image, verbose=0)

    return feature

# -----------------------------
# Generate Caption
# -----------------------------
def predict_caption(model, image_feature, tokenizer, max_length):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding='post'
        )

        yhat = model.predict(
            [image_feature, sequence],
            verbose=0
        )

        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    final_caption = in_text.replace('startseq', '')
    final_caption = final_caption.replace('endseq', '')

    return final_caption.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ Image Caption Generator")

st.write(
    "Upload an image and AI will generate a caption."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Generating caption..."):

        feature = extract_features(image)

        caption = predict_caption(
            model,
            feature,
            tokenizer,
            max_length
        )

    st.success(f"Caption: {caption}")
