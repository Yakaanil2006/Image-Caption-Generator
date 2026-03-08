import streamlit as st
from PIL import Image
import tempfile
from utils import extract_features, generate_caption

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#1f1f2e,#2b2b45);
    color: white;
}

.title {
    text-align: center;
    font-size: 46px;
    font-weight: 700;
    color: #4CAF50;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #d0d0d0;
    margin-bottom: 35px;
}

.upload-box {
    background: #2f2f4a;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #3e3e5a;
}

.caption-box {
    background: #4CAF50;
    color: white;
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    font-weight: 500;
    margin-top: 15px;
}

div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 25px;
    border: none;
}

div.stButton > button:hover {
    background-color: #45a049;
}

img {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<p class="title">🖼️ AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

# ---------------- Upload Section ----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Image Processing ----------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        if st.button("✨ Generate Caption"):

            with st.spinner("AI is analyzing the image..."):
                feature = extract_features(temp_path)
                caption = generate_caption(feature)

            st.success("Caption Generated!")

            st.markdown(
                f'<div class="caption-box">{caption}</div>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error("⚠️ Error processing the image. Please upload a valid image.")
        st.write(str(e))
