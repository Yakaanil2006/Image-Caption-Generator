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

# ---------------- Improved CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    color: #22c55e;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 35px;
}

.box {
    background: #1e293b;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

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

.caption-box {
    background: #22c55e;
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    font-weight: 600;
    margin-top: 20px;
}

img {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<p class="title">🧠 AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

# ---------------- Two Vertical Layout ----------------
col1, col2 = st.columns(2)

# ---------------- Left Side : Upload + Image ----------------
with col1:

    st.markdown('<div class="box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Right Side : Caption Generator ----------------
with col2:

    st.markdown('<div class="box">', unsafe_allow_html=True)

    if uploaded_file is not None:

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

    else:
        st.info("Upload an image to generate caption")

    st.markdown('</div>', unsafe_allow_html=True)
