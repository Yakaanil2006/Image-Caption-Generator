# 🖼️ Image Caption Generator

An AI-powered Image Caption Generator built using **Deep Learning**, **Computer Vision**, and **Natural Language Processing (NLP)**.

> **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,000 images each paired with 5 human-written captions (40,000 captions total).

This project uses:
- **CNN (VGG16)** for image feature extraction
- **LSTM** for sequence-based caption generation
- **TensorFlow / Keras** for model training
- **Streamlit** for web deployment

---

## 🚀 Live Demo

🌐 https://image-caption-generator-2026.streamlit.app/

---

## 📌 Project Overview

The goal is to automatically generate meaningful natural-language captions for any uploaded image.

| Input Image | Generated Caption |
|---|---|
| Dog running in grass | `"a dog is running through the grass"` |

The model jointly learns:
- **Image understanding** — what objects and scenes are present
- **Sentence generation** — how to describe them in natural language
- **Word prediction** — building captions word-by-word

---

## 🧠 Technologies Used

| Technology | Role |
|---|---|
| Python 3 | Core language |
| TensorFlow / Keras | Model training & inference |
| CNN — VGG16 | Image feature extraction (transfer learning) |
| LSTM | Sequential caption generation |
| Streamlit | Web app deployment |
| NumPy | Numerical operations |
| NLTK | BLEU score evaluation |
| Pickle | Serializing features, tokenizer, mappings |

---

## 📂 Project Structure

```
Image-Caption-Generator/
│
├── Images/                         # Sample images
├── Image_Caption_Generator.ipynb   # Training notebook
├── app.py                          # Streamlit web application
├── best_model.keras                # Saved trained model
├── captions.txt                    # Flickr8k captions dataset
├── features.pkl                    # Pre-extracted VGG16 features
├── tokenizer.pkl                   # Fitted Keras tokenizer
├── mapping.pkl                     # image_id → captions mapping
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🔍 Step-by-Step Code Explanation

---

### 1️⃣ Import Required Libraries

```python
import os, re, pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
```

---

### 2️⃣ Load Flickr8k Captions Dataset

The Flickr8k `captions.txt` file maps each image filename to five human-written captions:

```
1000268201.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201.jpg,A girl going into a wooden building .
```

Parsed into:
```python
mapping = {
  "1000268201": ["A child in a pink...", "A girl going into...", ...]
}
```

---

### 3️⃣ Clean Captions

```python
import re

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()                          # lowercase
            caption = re.sub(r'[^a-zA-Z ]', '', caption)      # remove punctuation & digits
            caption = re.sub(r'\s+', ' ', caption)             # collapse whitespace
            caption = " ".join([w for w in caption.split() if len(w) > 1])  # drop single chars
            caption = 'startseq ' + caption + ' endseq'       # add boundary tokens
            captions[i] = caption
```

> **Note:** `re.sub()` is used (not `str.replace`) so the regex patterns `[^a-zA-Z ]` and `\s+` work correctly.

**Result:**
```
"A child in a pink dress..." → "startseq child in pink dress is climbing up set of stairs in entry way endseq"
```

---

### 4️⃣ Extract Features using VGG16

VGG16 is a pre-trained CNN from ImageNet. We remove its final classification layer to get a **4096-dimensional feature vector** per image.

```python
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# Output shape: (1, 4096)
```

Features are saved to `features.pkl` to avoid re-extracting on every run.

---

### 5️⃣ Tokenization

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
```

| Word | Token |
|---|---|
| `dog` | 25 |
| `running` | 67 |

Saved as `tokenizer.pkl`.

---

### 6️⃣ Create Training Sequences (Teacher Forcing)

For a caption `"startseq a dog is running endseq"` the model learns:

| Image Feature | Partial Caption | Predict |
|---|---|---|
| 4096-vec | `[startseq]` | `a` |
| 4096-vec | `[startseq, a]` | `dog` |
| 4096-vec | `[startseq, a, dog]` | `is` |

---

### 7️⃣ CNN + LSTM Model Architecture

```python
# Image branch
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Text branch
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Merge + decode
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs  = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

| Branch | Layer | Purpose |
|---|---|---|
| Image | `Dropout(0.4)` | Reduce overfitting |
| Image | `Dense(256)` | Compress 4096 → 256 |
| Text | `Embedding` | Word → 256-dim vector |
| Text | `LSTM(256)` | Learn word sequences |
| Merge | `add([...])` | Fuse visual + language |
| Output | `Dense(softmax)` | Predict next word |

---

### 8️⃣ Training

```python
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
```

---

### 9️⃣ Save Model

```python
model.save('best_model.keras')
```

---

### 🔟 Caption Prediction

```python
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text
```

Generation stops when `endseq` is predicted or `max_length` is reached.

---

## 📊 Evaluation — BLEU Scores

The model is evaluated using **BLEU (Bilingual Evaluation Understudy)** scores on a held-out test set (10% of Flickr8k).

```python
from nltk.translate.bleu_score import corpus_bleu

print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print("BLEU-3:", corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
print("BLEU-4:", corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
```

| Metric | Score | What it measures |
|---|---|---|
| BLEU-1 | unigram precision | Individual word overlap |
| BLEU-2 | bigram precision | Two-word phrase overlap |
| BLEU-3 | trigram precision | Three-word phrase overlap |
| BLEU-4 | 4-gram precision | Overall fluency & accuracy |

> BLEU-1 scores around **0.50–0.60** and BLEU-4 around **0.10–0.15** are typical for a VGG16+LSTM model trained on Flickr8k.

---

## 🌐 Streamlit Web App (`app.py`)

### Features
- ✅ Upload any JPG / JPEG / PNG image
- ✅ Display the uploaded image
- ✅ Generate and display an AI caption instantly
- ✅ Graceful error handling if model files are missing

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Yakaanil2006/Image-Caption-Generator.git
cd Image-Caption-Generator

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow
streamlit
numpy
Pillow
nltk
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 💼 Skills Demonstrated

- Deep Learning (CNN + LSTM encoder–decoder)
- Transfer Learning (VGG16 pretrained on ImageNet)
- Natural Language Processing (tokenization, sequence modeling)
- Computer Vision (image preprocessing, feature extraction)
- Model Evaluation (BLEU-1 through BLEU-4)
- TensorFlow / Keras
- Streamlit Deployment

---

## 🎯 Recruiter Highlights

✅ End-to-end AI pipeline — data → training → evaluation → deployment  
✅ Transfer learning with a production-grade CNN (VGG16)  
✅ Quantitative model evaluation with BLEU scores  
✅ Computer Vision + NLP integration  
✅ Real-world deployment via Streamlit Cloud  
✅ Clean, well-commented code in both notebook and app  

---

## 📚 Dataset Reference

**Flickr8k** — M. Hodosh, P. Young and J. Hockenmaier (2013).  
*"Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics."*  
Journal of Artificial Intelligence Research, Volume 47, pages 853–899.  
🔗 https://www.kaggle.com/datasets/adityajn105/flickr8k

---

## 🔗 Links

- **GitHub:** https://github.com/Yakaanil2006/Image-Caption-Generator  
- **Live Demo:** https://image-caption-generator-2026.streamlit.app/
