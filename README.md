# 🖼️ Image Caption Generator

An AI-powered Image Caption Generator built using **Deep Learning**, **Computer Vision**, and **Natural Language Processing (NLP)**.

This project uses:

* **CNN (VGG16)** for image feature extraction
* **LSTM** for caption generation
* **TensorFlow/Keras**
* **Streamlit** for deployment

---

# 🚀 Live Demo

🌐 https://image-caption-generator-2026.streamlit.app/

---

# 📌 Project Overview

The goal of this project is to generate meaningful captions for images automatically.

Example:

| Input Image          | Generated Caption                    |
| -------------------- | ------------------------------------ |
| Dog running in grass | "a dog is running through the grass" |

The model learns:

* image understanding
* object detection
* sentence generation
* word prediction

---

# 🧠 Technologies Used

* Python
* TensorFlow / Keras
* CNN (VGG16)
* LSTM
* Streamlit
* NumPy
* Pickle

---

# 📂 Project Structure

```bash
Image-Caption-Generator/
│
├── Images/
├── Image_Caption_Generator.ipynb
├── app.py
├── best_model.keras
├── captions.txt
├── features.pkl
├── tokenizer.pkl
├── mapping.pkl
├── requirements.txt
└── README.md
```

---

# 🔍 Step-by-Step Code Explanation

---

# 1️⃣ Import Required Libraries

The project starts by importing necessary libraries.

```python
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
```

### Purpose

| Library          | Usage                   |
| ---------------- | ----------------------- |
| TensorFlow/Keras | Deep learning           |
| NumPy            | Numerical operations    |
| Pickle           | Save tokenizer/features |
| VGG16            | Feature extraction      |

---

# 2️⃣ Load Captions Dataset

The dataset contains image names and captions.

Example:

```text
1000268201.jpg,A child is playing outside
```

The code maps:

```python
image_name → captions
```

This helps connect images with their descriptions.

---

# 3️⃣ Clean Captions

The captions are cleaned before training.

## Code

```python
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]

            caption = caption.lower()

            caption = caption.replace('[^A-Za-z]', '')

            caption = caption.replace('\s+', ' ')

            caption = 'startseq ' + caption + ' endseq'

            captions[i] = caption
```

---

# 🔍 What This Function Does

### Convert to Lowercase

```python
caption.lower()
```

Example:

```text
Dog → dog
```

---

### Remove Special Characters

```python
caption.replace('[^A-Za-z]', '')
```

Removes:

* punctuation
* symbols
* numbers

---

### Add Start and End Tokens

```python
startseq
endseq
```

Example:

```text
startseq a dog running endseq
```

These tokens help the model understand:

* sentence start
* sentence end

---

# 4️⃣ Extract Features using VGG16

The project uses **VGG16**, a pre-trained CNN model.

## Load VGG16

```python
model = VGG16()
```

---

# Remove Last Layer

```python
model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)
```

### Why?

The final classification layer is removed because we only need:

✅ image features
❌ image classification

---

# Feature Extraction

Each image becomes a:

```text
4096-dimensional vector
```

These features are saved in:

```text
features.pkl
```

This speeds up training.

---

# 5️⃣ Tokenization

The tokenizer converts words into numbers.

## Example

| Word    | Token |
| ------- | ----- |
| dog     | 25    |
| running | 67    |

## Code

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
```

The tokenizer is saved as:

```python
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

---

# 6️⃣ Create Input Sequences

The model learns next-word prediction.

## Example Caption

```text
startseq a dog running endseq
```

Training sequences:

| Input          | Output  |
| -------------- | ------- |
| startseq       | a       |
| startseq a     | dog     |
| startseq a dog | running |

This helps the model learn sentence generation.

---

# 7️⃣ Build CNN + LSTM Model

The architecture contains two parts.

---

# 🖼️ Image Model

Processes image features.

```python
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
```

### Purpose

* Reduce overfitting
* Learn important visual patterns

---

# 📝 Text Model

Processes caption sequences.

```python
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256)(inputs2)
se2 = LSTM(256)(se1)
```

### Purpose

| Layer     | Usage                       |
| --------- | --------------------------- |
| Embedding | Converts words into vectors |
| LSTM      | Learns sentence sequence    |

---

# 8️⃣ Combine Both Models

The outputs are merged together.

```python
decoder1 = add([fe2, se2])
```

This combines:

* image understanding
* language understanding

---

# Final Prediction Layer

```python
outputs = Dense(vocab_size, activation='softmax')(decoder2)
```

The model predicts the next word.

---

# 9️⃣ Compile and Train Model

## Compile

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)
```

---

## Train

```python
model.fit()
```

The model learns:

* object relationships
* sentence patterns
* caption generation

---

# 🔟 Save Trained Model

```python
model.save('best_model.keras')
```

This saves the trained model for future use.

---

# 🧪 Caption Prediction Process

When a user uploads an image:

---

## Step 1

Image is resized.

---

## Step 2

VGG16 extracts features.

---

## Step 3

Caption generation starts with:

```text
startseq
```

---

## Step 4

Model predicts words one-by-one.

Example:

```text
startseq
→ a
→ dog
→ running
→ outside
```

---

## Step 5

Generation stops when:

```text
endseq
```

is predicted.

---

# 🌐 Streamlit Web App

The project includes a Streamlit interface.

## Features

✅ Upload image
✅ Display uploaded image
✅ Generate captions instantly

---

# ▶️ Run the Project

## Clone Repository

```bash
git clone https://github.com/Yakaanil2006/Image-Caption-Generator.git
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 💼 Skills Demonstrated

* Deep Learning
* CNN & LSTM
* Computer Vision
* NLP
* TensorFlow/Keras
* Streamlit Deployment

---

# 🎯 Recruiter Highlights

This project demonstrates:

✅ End-to-end AI pipeline
✅ Deep Learning implementation
✅ Transfer Learning
✅ Computer Vision + NLP integration
✅ Real-world deployment using Streamlit
---

# 🔗 GitHub Repository

https://github.com/Yakaanil2006/Image-Caption-Generator
