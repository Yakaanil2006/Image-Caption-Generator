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

/* Background */
.stApp {
background: linear-gradient(135deg,#141E30,#243B55);
color:white;
font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title {
text-align:center;
font-size:50px;
font-weight:800;
background: linear-gradient(90deg,#4CAF50,#00E676);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-top:10px;
}

/* Subtitle */
.subtitle {
text-align:center;
font-size:18px;
color:#cfcfcf;
margin-bottom:35px;
}

/* Upload Card */
.upload-box {
background:rgba(255,255,255,0.05);
padding:30px;
border-radius:15px;
border:1px solid rgba(255,255,255,0.1);
backdrop-filter:blur(10px);
box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

/* Caption Box */
.caption-box {
background:linear-gradient(90deg,#4CAF50,#2ecc71);
padding:20px;
border-radius:12px;
font-size:22px;
text-align:center;
font-weight:600;
margin-top:20px;
box-shadow:0 6px 25px rgba(0,0,0,0.4);
}

/* Button */
div.stButton > button {
background:linear-gradient(90deg,#4CAF50,#2ecc71);
color:white;
font-size:18px;
font-weight:600;
border-radius:10px;
padding:12px 28px;
border:none;
transition:all 0.3s ease;
}

/* Button Hover */
div.stButton > button:hover {
transform:scale(1.05);
background:linear-gradient(90deg,#43a047,#27ae60);
}

/* Image Styling */
img {
border-radius:12px;
box-shadow:0 10px 30px rgba(0,0,0,0.5);
}

/* Remove Streamlit Footer */
footer {
visibility:hidden;
}

/* Center file uploader */
section[data-testid="stFileUploader"] {
display:flex;
justify-content:center;
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

```
try:
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

except Exception as e:
    st.error("⚠️ Error processing the image. Please upload a valid image.")
    st.write(str(e))
```
