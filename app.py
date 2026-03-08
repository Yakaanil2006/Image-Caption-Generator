import streamlit as st
from PIL import Image
import tempfile
from utils import extract_features, generate_caption

# Page settings
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4CAF50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.caption-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    text-align: center;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🖼️ AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it!</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    st.write("")

    if st.button("✨ Generate Caption"):

        with st.spinner("AI is analyzing the image..."):

            feature = extract_features(temp_file.name)
            caption = generate_caption(feature)

        st.success("Caption Generated!")

        st.markdown(
            f'<div class="caption-box">{caption}</div>',
            unsafe_allow_html=True
        )