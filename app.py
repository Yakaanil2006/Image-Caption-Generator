import streamlit as st
from PIL import Image
import tempfile
import os
from utils import extract_features, generate_caption

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🧠",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg,#0f172a,#1e293b);
        color:white;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size:42px;
        font-weight:700;
        color:#22c55e;
        text-align:center;
        margin-bottom:5px;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        color:#cbd5f5;
        margin-bottom:30px;
    }
    .upload-card {
        background:#1e293b;
        padding:25px;
        border-radius:12px;
        border:1px solid #334155;
        box-shadow:0 10px 30px rgba(0,0,0,0.4);
    }
    .caption-box {
        background:#22c55e;
        padding:18px;
        border-radius:10px;
        font-size:20px;
        text-align:center;
        font-weight:600;
        margin-top:20px;
        color: white;
    }
    div.stButton > button {
        background:#22c55e;
        color:white;
        font-size:16px;
        border-radius:8px;
        padding:10px 25px;
        border:none;
        transition:0.25s;
        width: 100%;
    }
    div.stButton > button:hover {
        background:#16a34a;
        transform:scale(1.02);
    }
    img {
        border-radius:10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📤 Upload Image", "🤖 Model Info", "ℹ️ About Project"]
)

# ---------------- Upload Image Page ----------------
if page == "📤 Upload Image":
    st.markdown('<p class="title">AI Image Caption Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            # Create a temporary file to pass to the feature extractor
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            if st.button("✨ Generate Caption"):
                with st.spinner("🤖 AI is analyzing the image..."):
                    try:
                        feature = extract_features(temp_path)
                        caption = generate_caption(feature)
                        
                        st.success("Analysis Complete!")
                        st.markdown(
                            f'<div class="caption-box">{caption}</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

# ---------------- Model Info Page ----------------
elif page == "🤖 Model Info":
    st.title("🤖 Model Information")
    
    
    
    st.write("""
    This project utilizes a **Merge Architecture** for Image Captioning.
    
    - **Encoder (CNN):** A pre-trained **Xception** model (trained on ImageNet) acts as the feature extractor. We remove the final classification layer to get a vector representation of the image.
    - **Decoder (RNN/LSTM):** An **LSTM (Long Short-Term Memory)** network that takes the image features and the previous word to predict the next word in the sequence.
    - **Dataset:** Trained on the **Flickr8k** dataset, which contains 8,000 images, each paired with five different descriptive captions.
    """)

# ---------------- About Page ----------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.info("This is a deep learning application built to bridge the gap between Computer Vision and Natural Language Processing.")
    
    st.markdown("""
    ### Technical Stack:
    - **Frontend:** Streamlit
    - **Deep Learning:** TensorFlow / Keras
    - **Image Processing:** Pillow, OpenCV
    - **Deployment:** Streamlit Cloud / AWS
    *Computer Science Student | AI & ML Enthusiast*
    """)
