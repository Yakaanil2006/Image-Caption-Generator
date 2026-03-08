import streamlit as st
from PIL import Image
import tempfile
from utils import extract_features, generate_caption

# ---------------- Page Configuration ----------------

st.set_page_config(
page_title="AI Image Caption Generator",
page_icon="🧠",
layout="wide"
)

# ---------------- CSS Styling ----------------

st.markdown("""

<style>

/* Background */
.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
font-family: 'Segoe UI', sans-serif;
}

/* Main container width */
.block-container{
max-width:900px;
padding-top:30px;
}

/* Title */
.title{
font-size:42px;
font-weight:700;
color:#22c55e;
text-align:center;
margin-bottom:5px;
}

/* Subtitle */
.subtitle{
text-align:center;
font-size:18px;
color:#cbd5f5;
margin-bottom:35px;
}

/* Upload Box */
.upload-card{
background:#1e293b;
padding:25px;
border-radius:12px;
border:1px solid #334155;
box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

/* Caption result */
.caption-box{
background:#22c55e;
padding:18px;
border-radius:10px;
font-size:20px;
text-align:center;
font-weight:600;
margin-top:20px;
}

/* Buttons */
div.stButton > button{
background:#22c55e;
color:white;
font-size:16px;
border-radius:8px;
padding:10px 25px;
border:none;
transition:0.25s;
}

div.stButton > button:hover{
background:#16a34a;
transform:scale(1.03);
}

/* Image style */
img{
border-radius:10px;
margin-top:10px;
}

/* Hide footer */
footer{
visibility:hidden;
}

</style>

""", unsafe_allow_html=True)

# ---------------- Sidebar Navigation ----------------

st.sidebar.title("🧠 Navigation")

page = st.sidebar.radio(
"Select Page",
["📤 Upload Image", "🤖 Model Info", "ℹ️ About Project"]
)

# ---------------- PAGE 1 : Upload Image ----------------

if page == "📤 Upload Image":

```
st.markdown('<p class="title">AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    if st.button("✨ Generate Caption"):

        with st.spinner("🤖 AI is analyzing the image..."):
            feature = extract_features(temp_path)
            caption = generate_caption(feature)

        st.success("Caption Generated!")

        st.markdown(
            f'<div class="caption-box">{caption}</div>',
            unsafe_allow_html=True
        )
```

# ---------------- PAGE 2 : Model Info ----------------

elif page == "🤖 Model Info":

```
st.title("🤖 Model Information")

st.write("""
```

This project uses a **Deep Learning Image Captioning Model**.

### Architecture

CNN + LSTM Model

• **Xception CNN**
Extracts image features.

• **LSTM Network**
Generates the caption word by word.

• **Tokenizer**
Converts words into numerical sequences.

### Workflow

1️⃣ Upload Image
2️⃣ Extract Features using CNN
3️⃣ Generate Caption using LSTM
""")

# ---------------- PAGE 3 : About ----------------

elif page == "ℹ️ About Project":

```
st.title("ℹ️ About This Project")

st.write("""
```

This **AI Image Caption Generator** automatically describes images using deep learning.

### Technologies Used

* TensorFlow
* Keras
* Streamlit
* CNN + LSTM Architecture

### Developer

Anil Kumar
Machine Learning Enthusiast 🚀
""")
