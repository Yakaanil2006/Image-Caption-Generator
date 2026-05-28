# 🖼️ Image Caption Generator using CNN + LSTM

An end-to-end Deep Learning project that automatically generates captions for images using **Computer Vision** and **Natural Language Processing (NLP)** techniques.

This project combines:

* **CNN (VGG16)** for image feature extraction
* **LSTM** for sequential text generation
* **Transfer Learning**
* **Streamlit** for deployment

The model is trained on the **Flickr8k Dataset**, where each image contains multiple human-written captions.

---

# 📌 Project Objective

The goal of this project is to build an AI system capable of understanding image content and generating meaningful natural language descriptions automatically.

This project demonstrates practical implementation of:

* Deep Learning
* Computer Vision
* NLP
* Transfer Learning
* Sequence Modeling
* AI Deployment

---

# 🚀 Features

✅ Upload any image
✅ Generate captions automatically
✅ CNN + LSTM hybrid architecture
✅ Pre-trained VGG16 feature extraction
✅ Streamlit web application
✅ Saved trained model support
✅ Real-time caption generation

---

# 🧠 Deep Learning Architecture

```text
Input Image
     ↓
VGG16 CNN Feature Extractor
     ↓
4096-Dimensional Feature Vector
     ↓
LSTM Caption Generator
     ↓
Word-by-Word Prediction
     ↓
Generated Caption
```

---

# 📂 Project Structure

```bash
Image-Caption-Generator/
│
├── Images/                         # Dataset images
├── Image_Caption_Generator.ipynb  # Training notebook
├── app.py                          # Streamlit web app
├── best_model.keras                # Trained model
├── captions.txt                    # Dataset captions
├── features.pkl                    # Extracted image features
├── mapping.pkl                     # Image-caption mapping
├── tokenizer.pkl                   # Saved tokenizer
├── requirements.txt                # Required libraries
├── README.md
└── .gitattributes
```

---

# 📚 Dataset Used

## Flickr8k Dataset

The Flickr8k dataset contains:

* 8,000 images
* 5 captions per image
* Human-annotated image descriptions

Example:

```text
A dog is running through the grass.
A brown dog is playing outside.
A dog runs in a grassy field.
```

This dataset helps the model learn both:

* Visual understanding
* Language generation

---

# 🔍 Step-by-Step Code Explanation

---

# 1️⃣ Importing Required Libraries

The project starts by importing essential libraries.

```python
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

---

## Why These Libraries?

| Library          | Purpose                   |
| ---------------- | ------------------------- |
| NumPy            | Numerical operations      |
| Pickle           | Saving tokenizer/features |
| TensorFlow/Keras | Deep learning             |
| VGG16            | Feature extraction        |
| Tokenizer        | Text preprocessing        |

---

# 2️⃣ Loading Captions Dataset

The captions file contains image names and their corresponding captions.

Example:

```text
1000268201_693b08cb0e.jpg,A child climbing stairs.
```

The code creates a mapping between:

```python
image_name → list_of_captions
```

This helps associate multiple captions with one image.

---

# 3️⃣ Caption Cleaning Function

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

# 🔍 Explanation of Each Step

---

## Convert to Lowercase

```python
caption.lower()
```

Purpose:

* Removes case differences
* Helps model treat:

  * Dog
  * dog

as same word.

---

## Remove Special Characters

```python
caption.replace('[^A-Za-z]', '')
```

Removes:

* punctuation
* symbols
* numbers

Example:

```text
Dog!!!
↓
Dog
```

---

## Remove Extra Spaces

```python
caption.replace('\s+', ' ')
```

Removes unnecessary spaces.

---

## Add Sequence Tokens

```python
'startseq ' + caption + ' endseq'
```

These tokens help the model learn:

* Sentence beginning
* Sentence ending

Example:

```text
startseq a dog running endseq
```

---

# 4️⃣ Feature Extraction using VGG16

---

# Why VGG16?

VGG16 is a pre-trained CNN model trained on millions of images from ImageNet.

Advantages:

✅ Powerful image understanding
✅ Transfer learning
✅ Faster training
✅ Better feature extraction

---

# Loading VGG16

```python
model = VGG16()
```

---

# Removing Final Classification Layer

```python
model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)
```

---

# Why Remove Last Layer?

The last layer predicts image classes.

For caption generation, we only need:

✅ image features
❌ image classification

---

# Feature Vector Extraction

Each image becomes:

```text
4096-dimensional vector
```

These features are saved in:

```text
features.pkl
```

This improves efficiency because features don't need to be extracted repeatedly.

---

# 5️⃣ Tokenization

Deep learning models cannot understand words directly.

Tokenizer converts words into numbers.

---

# Example

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
```

---

# Word-to-Integer Mapping

| Word    | Token |
| ------- | ----- |
| dog     | 25    |
| running | 67    |
| grass   | 90    |

---

# Why Tokenization?

Neural networks only process numerical data.

The tokenizer is saved using:

```python
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

---

# 6️⃣ Sequence Generation

The model learns next-word prediction.

---

# Example Caption

```text
startseq a dog running endseq
```

Training sequences become:

| Input          | Output  |
| -------------- | ------- |
| startseq       | a       |
| startseq a     | dog     |
| startseq a dog | running |

---

# Sequence Padding

Sentences have different lengths.

To make input size equal:

```python
pad_sequences(sequence, maxlen=max_length)
```

Padding ensures:

✅ Fixed-length input
✅ Efficient batch training

---

# 7️⃣ Building the Deep Learning Model

The architecture contains two separate models.

---

# 🖼️ Image Model

Processes image feature vectors.

```python
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
```

---

# Explanation

| Layer   | Purpose                        |
| ------- | ------------------------------ |
| Dropout | Prevent overfitting            |
| Dense   | Learn important image patterns |

---

# 📝 Text Model

Processes caption sequences.

```python
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)
```

---

# Explanation

---

## Embedding Layer

Converts tokens into dense vectors.

Example:

```text
dog → [0.21, 0.66, 0.92]
```

---

## LSTM Layer

LSTM remembers previous words while predicting the next word.

This helps generate grammatically meaningful captions.

---

# 8️⃣ Combining Both Models

The outputs are merged together.

```python
decoder1 = add([fe2, se3])
```

This combines:

* Image understanding
* Language understanding

---

# Final Prediction Layer

```python
outputs = Dense(vocab_size, activation='softmax')(decoder2)
```

Purpose:

Predict the most probable next word.

---

# 9️⃣ Model Compilation

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)
```

---

# Why These Choices?

| Component         | Reason                     |
| ----------------- | -------------------------- |
| Adam Optimizer    | Fast convergence           |
| Crossentropy Loss | Multi-class classification |

---

# 🔟 Model Training

```python
model.fit()
```

During training, the model learns:

✅ Image context
✅ Sentence patterns
✅ Word relationships
✅ Caption generation

---

# 💾 Saving the Model

```python
model.save('best_model.keras')
```

This allows reuse without retraining.

---

# 🧪 Caption Prediction Process

When user uploads an image:

---

## Step 1

Resize image for VGG16.

---

## Step 2

Extract image features.

---

## Step 3

Start caption with:

```text
startseq
```

---

## Step 4

Predict next word repeatedly.

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

Stop when:

```text
endseq
```

is predicted.

---

# 🌐 Streamlit Web Application

The project includes a Streamlit interface for real-time interaction.

---

# app.py Workflow

---

## Load Model

```python
model = load_model('best_model.keras')
```

---

## Load Tokenizer

```python
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
```

---

## Upload Image

```python
uploaded_file = st.file_uploader()
```

---

## Generate Caption

The uploaded image is processed and passed into the model.

Predicted caption is displayed instantly.

---

# ▶️ How to Run the Project

---

# Step 1: Clone Repository

```bash
git clone https://github.com/Yakaanil2006/Image-Caption-Generator.git
```

---

# Step 2: Open Project Folder

```bash
cd Image-Caption-Generator
```

---

# Step 3: Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

# Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Step 5: Run Streamlit App

```bash
streamlit run app.py
```

---

# Step 6: Open Browser

```text
http://localhost:8501
```

---

# 📊 Skills Demonstrated

This project demonstrates:

---

## Machine Learning Skills

* Deep Learning
* CNN
* LSTM
* NLP
* Computer Vision
* Transfer Learning

---

## Engineering Skills

* Data preprocessing
* Feature engineering
* Model optimization
* Sequence modeling
* AI deployment

---

## Development Skills

* Python
* TensorFlow/Keras
* Streamlit
* GitHub project management

---

# 🎯 Recruiter Highlights

This project showcases the ability to:

✅ Build complete AI systems
✅ Combine Computer Vision + NLP
✅ Implement Transfer Learning
✅ Develop deployable ML applications
✅ Work with real-world datasets
✅ Build end-to-end Deep Learning pipelines

---

# 🔮 Future Improvements

Possible enhancements:

* Attention Mechanism
* Beam Search
* Transformer Models
* Larger datasets (MSCOCO)
* Multilingual captions
* Cloud deployment

---

# 👨‍💻 Author

## Anil Kumar

AI/ML Enthusiast passionate about:

* Deep Learning
* Computer Vision
* NLP
* Generative AI

---

# ⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork the project
📢 Share with others

---

# 🔗 GitHub Repository

https://github.com/Yakaanil2006/Image-Caption-Generator
