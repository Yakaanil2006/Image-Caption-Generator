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

# ---------------- Improved CSS ----------------

st.markdown("""

<style>

/* ---------- Main Background ---------- */
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- Container Width ---------- */
.block-container {
    max-width: 750px;
    padding-top: 40px;
}

/* ---------- Title ---------- */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    color: #22c55e;
    margin-bottom: 8px;
}

/* ---------- Subtitle ---------- */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 35px;
}

/* ---------- Upload Card ---------- */
.upload-box {
    background: #1e293b;
    padding: 30px;
    border-radius: 14px;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* ---------- File Uploader ---------- */
section[data-testid="stFileUploader"] {
    border: 2px dashed #475569;
    border-radius: 12px;
    padding: 25px;
    background: #0f172a;
}

/* ---------- Button ---------- */
div.stButton > button {
    background: #22c55e;
    color: white;
    font-size: 17px;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 28px;
    border: none;
    transition: 0.25s;
}

div.stButton > button:hover {
    background: #16a34a;
    transform: scale(1.03);
}

/* ---------- Caption Result ---------- */
.caption-box {
    background: #22c55e;
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    font-weight: 600;
    margin-top: 20px;
}

/* ---------- Image Styling ---------- */
img {
    border-radius: 10px;
    margin-top: 10px;
}

/* ---------- Hide Streamlit Footer ---------- */
footer {
    visibility: hidden;
}

</style>

""", unsafe_allow_html=True)

# ---------------- Title ----------------

st.markdown('<p class="title">🧠 AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

# ---------------- Upload Section ----------------

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Image Processing ----------------

if uploaded_file is not None:

```
try:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    # Generate Caption
    if st.button("✨ Generate Caption"):

        with st.spinner("🤖 AI is analyzing the image..."):
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
```
