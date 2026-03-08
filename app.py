import streamlit as st
from PIL import Image
import tempfile
from utils import extract_features, generate_caption

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="VisionAI | Caption Generator",
    page_icon="✨",
    layout="wide"
)

# ---------------- Advanced Custom CSS ----------------
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Typography */
    .hero-title {
        background: linear-gradient(90deg, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 50px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Styled Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4);
    }

    /* Result Box */
    .result-card {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }

    .caption-text {
        font-size: 1.4rem;
        font-weight: 500;
        color: #4ade80;
        font-style: italic;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header Section ----------------
st.markdown('<h1 class="hero-title">VisionAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Transforming pixels into poetic descriptions with Neural Networks</p>', unsafe_allow_html=True)

# ---------------- Main Interface ----------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🖼️ Media Upload")
    uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Using a container to ensure the button stays within the card
        if st.button("✨ Analyze & Describe"):
            # Trigger logic handled in col2
            st.session_state.process = True
        else:
            st.session_state.process = False
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        # Image Preview
        st.image(image, use_container_width=True, caption="Target Image")
        
        # Processing Logic
        if st.session_state.get('process', False):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name

                with st.status("🛠️ Working magic...", expanded=True) as status:
                    st.write("Extracting visual features...")
                    feature = extract_features(path)
                    st.write("Synthesizing caption...")
                    caption = generate_caption(feature)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                st.markdown(f"""
                <div class="result-card">
                    <p style="margin-bottom:5px; color:#94a3b8; font-size:0.9rem;">AI GENERATED CAPTION</p>
                    <p class="caption-text">"{caption}"</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Computation Error: {e}")
    else:
        # Empty State
        st.markdown("""
        <div style="text-align:center; margin-top:50px; opacity:0.5;">
            <p style="font-size:50px;">📷</p>
            <p>Upload an image to begin the analysis</p>
        </div>
        """, unsafe_allow_html=True)
