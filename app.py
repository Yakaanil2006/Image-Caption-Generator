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

# ---------------- Advanced UI CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#020617);
    color:white;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#22c55e;
}

.subtitle{
    text-align:center;
    color:#94a3b8;
    margin-bottom:40px;
}

/* Glass Cards */
.card{
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(14px);
    border-radius:16px;
    padding:28px;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 10px 30px rgba(0,0,0,0.45);
}

/* Upload box */
section[data-testid="stFileUploader"]{
    border:2px dashed #475569;
    border-radius:12px;
    padding:35px;
    background:#0f172a;
}

/* Button */
div.stButton > button{
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color:white;
    font-size:18px;
    font-weight:600;
    border-radius:10px;
    padding:12px 32px;
    border:none;
    transition:0.3s;
}

div.stButton > button:hover{
    transform:scale(1.05);
    box-shadow:0 8px 20px rgba(34,197,94,0.5);
}

/* Caption Box */
.caption{
    background: linear-gradient(90deg,#22c55e,#16a34a);
    padding:22px;
    border-radius:12px;
    font-size:22px;
    text-align:center;
    font-weight:600;
    margin-top:20px;
}

/* Image */
img{
    border-radius:14px;
    margin-top:15px;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<p class="title">🧠 AI Image Caption Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI describe it instantly</p>', unsafe_allow_html=True)

# ---------------- Layout ----------------
left, right = st.columns([1,1])

# ---------------- Upload Column ----------------
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image,use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Caption Column ----------------
with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path=temp_file.name

        if st.button("✨ Generate Caption"):

            with st.spinner("🤖 AI analyzing image..."):

                feature=extract_features(temp_path)
                caption=generate_caption(feature)

            st.success("Caption Generated!")

            st.markdown(
                f'<div class="caption">{caption}</div>',
                unsafe_allow_html=True
            )

    else:
        st.info("Upload an image to generate caption")

    st.markdown('</div>', unsafe_allow_html=True)
