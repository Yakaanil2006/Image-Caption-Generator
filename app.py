import streamlit as st
from PIL import Image
import tempfile
from utils import extract_features, generate_caption

# Page config

st.set_page_config(
page_title="AI Image Caption Generator",
page_icon="🧠",
layout="wide"
)

# CSS

st.markdown("""

<style>
.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
font-family: 'Segoe UI', sans-serif;
}

.title{
font-size:42px;
font-weight:700;
color:#22c55e;
text-align:center;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#cbd5f5;
margin-bottom:30px;
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
}

div.stButton > button:hover{
background:#16a34a;
}

footer{
visibility:hidden;
}
</style>

""", unsafe_allow_html=True)

# Sidebar navigation

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"Go to",
["Upload Image", "Model Info", "About"]
)

# Upload Page

if page == "Upload Image":

```
st.markdown('<p class="title">AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    if st.button("Generate Caption"):

        with st.spinner("AI analyzing image..."):
            feature = extract_features(temp_path)
            caption = generate_caption(feature)

        st.success("Caption Generated")

        st.markdown(
            f'<div class="caption-box">{caption}</div>',
            unsafe_allow_html=True
        )
```

# Model Info

elif page == "Model Info":

```
st.title("Model Information")

st.write("""
```

CNN + LSTM architecture used.

CNN (Xception) extracts image features.
LSTM generates captions word by word.
Tokenizer converts words into sequences.
""")

# About

elif page == "About":

```
st.title("About This Project")

st.write("""
```

AI Image Caption Generator built using:

TensorFlow
Keras
Streamlit
CNN + LSTM architecture
""")
