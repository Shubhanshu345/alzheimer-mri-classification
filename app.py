import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Alzheimer's MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Premium navy / blue theme ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#071426; --panel:#10233d; --blue:#4ea1ff; --muted:#9bb0c9; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(circle at 82% 0%,#17395d 0%,#0a1b30 36%,#071426 78%); color:#edf5ff; }
.block-container { max-width:1400px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#08172a; border-right:1px solid #1d3858; }
[data-testid="stSidebar"] * { color:#dceaff; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:#f4f8ff; }
.hero { padding:28px 30px; border:1px solid #2a4d72; border-radius:22px;
background:linear-gradient(120deg,rgba(28,70,115,.92),rgba(13,32,55,.96)); margin-bottom:24px; }
.eyebrow { color:#83c0ff; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero h1 { font-size:clamp(27px,4vw,38px); margin:8px 0 6px; }
.hero p { color:#bed0e5; margin:0; font-size:15px; }
.metric { background:rgba(16,35,61,.9); border:1px solid #294866; border-radius:16px; padding:18px; min-height:112px; }
.metric-label { color:#9bb0c9; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#f5f9ff; font-family:'Manrope'; font-weight:800; font-size:25px; margin-top:8px; }
.panel { background:rgba(16,35,61,.84); border:1px solid #294866; border-radius:18px; padding:20px; }
.kicker { color:#83c0ff; font-size:12px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; }
.muted { color:#9bb0c9; font-size:13px; }
div.stButton > button { background:linear-gradient(90deg,#2678d8,#4ea1ff); color:white; border:0; border-radius:11px; font-weight:700; min-height:46px; }
div.stButton > button:hover { background:linear-gradient(90deg,#398ce9,#6bb2ff); color:white; border:0; }
[data-testid="stFileUploader"] { background:rgba(12,29,50,.65); border:1px dashed #47749e; border-radius:14px; padding:10px; }
.stProgress > div > div > div > div { background:#4ea1ff; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- Existing model and labels ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "alzheimer_cnn_model.keras"
CLASS_NAMES = [
    "Mild Demented",
    "Moderate Demented",
    "Non-Demented",
    "Very Mild Demented",
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- Sidebar navigation ----------------
with st.sidebar:
    st.markdown("## 🧠 Alzheimer's MRI")
    st.caption("Classification research dashboard")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "MRI Analysis", "Model Information", "About Project"],
    )
    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    st.markdown("🟢 Academic research prototype")
    st.caption("Not intended for clinical diagnosis.")

# ---------------- Main banner ----------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered medical imaging research</div>
  <h1>Alzheimer's MRI Classification</h1>
  <p>Experimental CNN-based classification of brain MRI images across four dataset categories.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="kicker">DASHBOARD OVERVIEW</div>', unsafe_allow_html=True)
    st.subheader("Welcome to your research workspace")
    st.write("Upload an MRI image, run the trained model, and review its experimental output scores.")
    cols = st.columns(3)
    metrics = [
        ("MODEL TYPE", "CNN", "Convolutional Neural Network"),
        ("INPUT SIZE", "128 × 128", "RGB image preprocessing"),
        ("OUTPUT CLASSES", "04", "Dataset classification labels"),
    ]
    for col, (label, value, detail) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="muted">{detail}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Analysis workflow")
        st.markdown(
            "1. Upload a brain MRI image.\n"
            "2. Preview the selected image.\n"
            "3. Run the trained CNN model.\n"
            "4. Review the predicted class and all four model scores."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Important limitation")
        st.warning(
            "This is an academic prototype. Results are experimental and must not "
            "be used for diagnosis, treatment, or clinical decisions."
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "MRI Analysis":
    st.markdown('<div class="kicker">IMAGE INFERENCE</div>', unsafe_allow_html=True)
    st.subheader("Upload MRI for analysis")
    st.write("Supported formats: JPG, JPEG, PNG.")
    uploaded_file = st.file_uploader(
        "Select MRI image", type=["jpg", "jpeg", "png"], key="mri_upload"
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            preview_col, details_col = st.columns([1.1, 1])
            with preview_col:
                st.markdown("#### MRI preview")
                st.image(original_image, use_container_width=True)
            with details_col:
                st.markdown("#### Image information")
                st.markdown(
                    f'<div class="metric"><div class="metric-label">ORIGINAL DIMENSIONS</div>'
                    f'<div class="metric-value">{original_image.width} × {original_image.height}</div>'
                    f'<div class="muted">RGB image</div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "The image is resized to 128 × 128 pixels and normalized to the 0–1 range."
                )

            if st.button("Run MRI classification", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading model and analysing image..."):
                        model = load_model()
                        resized_image = original_image.resize((128, 128))
                        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
                        image_array = np.expand_dims(image_array, axis=0)
                        predictions = np.asarray(
                            model.predict(image_array, verbose=0)[0], dtype=float
                        )

                    if len(predictions) != len(CLASS_NAMES):
                        st.error(
                            "The model output does not match the four expected classes. "
                            "Please verify the model and class order."
                        )
                    else:
                        predicted_index = int(np.argmax(predictions))
                        st.markdown("---")
                        st.markdown(
                            '<div class="kicker">ANALYSIS RESULT</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"### {CLASS_NAMES[predicted_index]}")
                        st.info(
                            f"Highest model output score: "
                            f"**{predictions[predicted_index] * 100:.2f}%**"
                        )
                        st.markdown("#### Scores across all classes")
                        for class_name, score in zip(CLASS_NAMES, predictions):
                            st.write(f"**{class_name}** — {score * 100:.2f}%")
                            st.progress(float(np.clip(score, 0.0, 1.0)))
                        st.caption(
                            "These are model output scores, not calibrated diagnostic "
                            "probabilities or measures of medical certainty."
                        )
                        st.warning(
                            "Research use only. Do not use this result to make medical decisions."
                        )
                except Exception as error:
                    st.error(f"Unable to run the model: {error}")
        except Exception:
            st.error("This file could not be opened. Please upload a valid JPG, JPEG, or PNG image.")
    else:
        st.markdown(
            '<div class="panel"><h3>Ready for image upload</h3>'
            '<p class="muted">Your MRI preview and analysis results will appear here.</p></div>',
            unsafe_allow_html=True,
        )

elif page == "Model Information":
    st.markdown('<div class="kicker">TECHNICAL DETAILS</div>', unsafe_allow_html=True)
    st.subheader("Model information")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Architecture")
        st.write("Convolutional Neural Network (CNN)")
        st.write("Input dimensions: 128 × 128 × 3")
        st.write("Preprocessing: RGB conversion, resizing, and pixel normalization (/255).")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Classification categories")
        for name in CLASS_NAMES:
            st.markdown(f"• {name}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info(
        "This repository does not include a verified evaluation report for this dashboard. "
        "Accuracy and other performance metrics are therefore not fabricated or displayed."
    )

else:
    st.markdown('<div class="kicker">PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.subheader("About this project")
    st.write(
        "Alzheimer's Disease Stage Classification Using MRI is an academic project "
        "exploring CNN-based classification of brain MRI images into four categories "
        "represented in the dataset."
    )
    st.markdown("### Technology stack")
    st.markdown("- Python\n- TensorFlow / Keras\n- NumPy\n- Pillow\n- Streamlit")
    st.markdown("### Intended use")
    st.write("This application is designed for academic learning and experimentation only.")
    st.warning(
        "This is not a medical device. A model prediction cannot confirm or rule out "
        "Alzheimer's disease. Consult a qualified healthcare professional for medical concerns."
    )

st.markdown("---")
st.caption(
    "Alzheimer's MRI Classification · Academic research prototype · Not for medical diagnosis"
)
