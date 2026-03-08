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

.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
font-family: 'Segoe UI', sans-serif;
}

.block-container{
max-width:900px;
padding-top:30px;
}

.title{
font-size:42px;
font-weight:700;
color:#22c55e;
text-align:center;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#cbd5f5;
margin-bottom:35px;
}

.upload-card{
background:#1e293b;
padding:25px;
border-radius:12px;
border:1px solid #334155;
box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

.caption-box{
background:#22c55e;
padding:18px;
border-radius:10px;
font-size:20px;
text-align:center;
font-weight:600;
margin-top:20px;
}

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

img{
border-radius:10px;
margin-top:10px;
}

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

# ---------------- Upload Page ----------------

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

# ---------------- Model Info Page ----------------

elif page == "🤖 Model Info":

```
st.title("🤖 Model Information")

st.write("""
```

This project uses a Deep Learning Image Captioning Model.

Architecture:

• CNN (Xception) – Extracts image features
• LSTM Network – Generates the caption
• Tokenizer – Converts words into sequences

Workflow:

1. Upload Image
2. Extract features using CNN
3. Generate caption using LSTM
   """)

# ---------------- About Page ----------------

elif page == "ℹ️ About Project":

```
st.title("ℹ️ About This Project")

st.write("""
```

AI Image Caption Generator created using:

• TensorFlow
• Keras
• Streamlit
• CNN + LSTM architecture

This application automatically describes images using deep learning.
""")
