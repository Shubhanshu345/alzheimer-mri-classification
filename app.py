import streamlit as st
import numpy as np
import csv
import io
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

:root { --navy:#071426; --panel:#10233d; --blue:#4ea1ff; --muted:#9bb0c9; }
html, body, [class*="css"] { font-family:Arial,sans-serif; }
.stApp { background:radial-gradient(circle at 82% 0%,#17395d 0%,#0a1b30 36%,#071426 78%); color:#edf5ff; }
.block-container { max-width:1400px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#08172a; border-right:1px solid #1d3858; }
[data-testid="stSidebar"] * { color:#dceaff; }
h1,h2,h3 { font-family:Arial,sans-serif; color:#f4f8ff; }
.hero { padding:28px 30px; border:1px solid #2a4d72; border-radius:22px;
background:linear-gradient(120deg,rgba(28,70,115,.92),rgba(13,32,55,.96)); margin-bottom:24px; }
.eyebrow { color:#83c0ff; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero h1 { font-size:clamp(27px,4vw,38px); margin:8px 0 6px; }
.hero p { color:#bed0e5; margin:0; font-size:15px; }
.team-card { background:linear-gradient(145deg,rgba(20,48,78,.96),rgba(10,25,44,.96)); border:1px solid #315779; border-radius:18px; padding:22px 18px; min-height:150px; box-shadow:0 8px 24px rgba(0,0,0,.16); }
.team-avatar { width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#4ea1ff,#7c5cff); color:#fff; font-size:20px; font-weight:800; margin-bottom:14px; }
.team-name { font-family:'Manrope'; font-size:18px; font-weight:800; color:#f4f8ff; margin-bottom:5px; }
.team-role { color:#9fc9f5; font-size:12px; letter-spacing:.5px; text-transform:uppercase; }
.institution-banner { border:1px solid #355b7e; border-radius:18px; padding:20px 24px; background:linear-gradient(120deg,rgba(129,32,44,.26),rgba(16,35,61,.92)); display:flex; align-items:center; gap:16px; margin:18px 0; }
.iilm-mark { width:58px; height:58px; flex-shrink:0; display:flex; align-items:center; justify-content:center; border-radius:12px; background:#81202c; color:#fff; font-family:'Manrope'; font-weight:800; font-size:18px; letter-spacing:1px; }
.institution-title { font-family:'Manrope'; font-size:19px; font-weight:800; color:#f4f8ff; }
.institution-subtitle { color:#b9cde2; font-size:13px; margin-top:4px; }
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

@st.cache_resource(show_spinner="Loading CNN model (first load may take a little while)...")
def load_model():
    # Import TensorFlow only when the user requests MRI inference.
    import tensorflow as tf
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(show_spinner=False)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

# ---------------- Sidebar navigation ----------------
with st.sidebar:
    st.markdown("## 🧠 Alzheimer's MRI")
    st.caption("Classification research dashboard")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "MRI Analysis", "Model Information", "Dataset Information", "Model Performance", "Project Team", "About Project"],
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


# ---------------- Full-width visual hero banners for all dashboard sections ----------------
def render_section_hero(title, subtitle, artwork):
    st.markdown(f"""
    <div style="width:100%;min-height:205px;display:flex;align-items:center;justify-content:space-between;gap:24px;
    padding:24px 34px;margin:0 0 25px;border-radius:24px;overflow:hidden;border:1px solid #315779;
    background:linear-gradient(115deg,#102a49 0%,#123b62 48%,#07182b 100%);box-shadow:0 12px 36px rgba(0,0,0,.20);">
      <div style="flex:1;min-width:0">
        <div style="font-size:11px;letter-spacing:2.2px;text-transform:uppercase;color:#7fc5ff;font-weight:800">ALZHEIMER'S MRI · RESEARCH DASHBOARD</div>
        <div style="font-size:clamp(25px,3.2vw,38px);line-height:1.15;font-weight:800;color:#f4f8ff;margin:12px 0 10px">{title}</div>
        <div style="font-size:15px;line-height:1.55;color:#c1d8ef;max-width:680px">{subtitle}</div>
      </div>
      <div style="width:clamp(130px,22%,245px);flex-shrink:0">{artwork}</div>
    </div>
    """, unsafe_allow_html=True)

HERO_ART = {
 "Overview": '<svg viewBox="0 0 220 150" width="100%"><ellipse cx="110" cy="75" rx="76" ry="64" fill="#0b1c31" stroke="#5bb7ff" stroke-width="3"/><path d="M110 22 C88 10 68 32 76 49 C52 55 54 78 70 88 C59 108 80 126 97 114 C105 132 117 132 125 114 C146 127 166 106 150 88 C169 72 158 52 140 49 C146 29 126 18 110 22Z" fill="#b9dcff" stroke="#e4f4ff" stroke-width="2"/><path d="M110 25V117M82 44Q100 55 89 70M139 44Q122 55 133 70M72 88Q93 82 94 109M150 88Q130 83 127 109" fill="none" stroke="#4b83b8" stroke-width="4" stroke-linecap="round"/></svg>',
 "MRI Analysis": '<svg viewBox="0 0 220 150" width="100%"><rect x="28" y="12" width="164" height="126" rx="16" fill="#0b1c31" stroke="#5bb7ff" stroke-width="2"/><circle cx="110" cy="75" r="49" fill="#253d59" stroke="#9dd7ff" stroke-width="3"/><circle cx="110" cy="75" r="37" fill="#9fb8d0"/><path d="M110 39C90 36 77 54 84 68C70 81 88 104 101 94C109 110 124 99 126 92C145 100 153 79 137 68C143 52 124 37 110 39Z" fill="#e5edf5"/><path d="M110 42V98M87 62Q104 70 92 82M132 60Q119 70 132 83" stroke="#536e89" stroke-width="4" fill="none"/></svg>',
 "Model Information": '<svg viewBox="0 0 220 150"><g stroke="#62bdff" stroke-width="2" fill="#143b61"><circle cx="40" cy="40" r="14"/><circle cx="40" cy="110" r="14"/><circle cx="110" cy="28" r="14"/><circle cx="110" cy="75" r="14"/><circle cx="110" cy="122" r="14"/><circle cx="180" cy="50" r="14"/><circle cx="180" cy="100" r="14"/></g><g stroke="#8bd5ff" stroke-width="2"><path d="M54 40L96 28M54 40L96 75M54 110L96 75M54 110L96 122M124 28L166 50M124 75L166 50M124 75L166 100M124 122L166 100"/></g></svg>',
 "Dataset Information": '<svg viewBox="0 0 220 150"><rect x="22" y="16" width="176" height="118" rx="14" fill="#0b1c31" stroke="#5bb7ff" stroke-width="2"/><g fill="#3c8fdb"><rect x="42" y="86" width="24" height="30" rx="4"/><rect x="78" y="58" width="24" height="58" rx="4"/><rect x="114" y="30" width="24" height="86" rx="4"/><rect x="150" y="70" width="24" height="46" rx="4"/></g><path d="M34 118H184" stroke="#b9dcff" stroke-width="2"/><circle cx="54" cy="39" r="8" fill="#8c83ff"/><circle cx="82" cy="39" r="8" fill="#5bd7ff"/><circle cx="110" cy="39" r="8" fill="#68e0b0"/></svg>',
 "Model Performance": '<svg viewBox="0 0 220 150"><rect x="20" y="14" width="180" height="122" rx="14" fill="#0b1c31" stroke="#5bb7ff" stroke-width="2"/><path d="M44 112L80 83L108 95L150 48L178 35" stroke="#68d6ff" stroke-width="5" fill="none" stroke-linecap="round"/><g fill="#d8f3ff" stroke="#4ea1ff" stroke-width="3"><circle cx="44" cy="112" r="6"/><circle cx="80" cy="83" r="6"/><circle cx="108" cy="95" r="6"/><circle cx="150" cy="48" r="6"/><circle cx="178" cy="35" r="6"/></g></svg>',
 "Project Team": '<svg viewBox="0 0 220 150"><circle cx="75" cy="48" r="25" fill="#5ba8ff"/><circle cx="148" cy="48" r="25" fill="#8a7bff"/><circle cx="111" cy="38" r="25" fill="#b2dcff"/><path d="M30 130Q34 82 75 82Q116 82 120 130Z" fill="#2865a0"/><path d="M100 130Q104 75 148 75Q192 75 196 130Z" fill="#5147a6"/><path d="M65 130Q69 70 111 70Q153 70 157 130Z" fill="#3c8ecb"/></svg>',
 "About Project": '<svg viewBox="0 0 220 150"><rect x="42" y="16" width="136" height="118" rx="15" fill="#0b1c31" stroke="#5bb7ff" stroke-width="2"/><path d="M70 45H150M70 65H150M70 85H130" stroke="#80c8ff" stroke-width="5" stroke-linecap="round"/><circle cx="110" cy="111" r="10" fill="#7d83ff"/><path d="M105 111L109 115L116 107" stroke="white" stroke-width="2.5" fill="none"/></svg>'
}

if page == "Overview":
    render_section_hero("Alzheimer's MRI Classification", "AI-powered academic research prototype for exploring brain MRI image classification.", HERO_ART["Overview"])
    st.markdown('<div class="kicker">DASHBOARD OVERVIEW</div>', unsafe_allow_html=True)
    visual_left, visual_right = st.columns([1.45, 0.75], vertical_alignment="center")
    with visual_left:
        st.subheader("Welcome to your research workspace")
        st.write("A CNN-powered academic interface for exploring MRI image classification.")
        st.markdown("""
        <div class="panel">
          <div style="font-size:13px;color:#8fc8ff;font-weight:700;letter-spacing:1.2px">PROJECT SNAPSHOT</div>
          <div style="font-size:24px;font-weight:800;margin:8px 0">Brain MRI · Deep Learning · Streamlit</div>
          <div style="color:#b9cde2">Upload an image, explore the model output, and review the research workflow.</div>
        </div>
        """, unsafe_allow_html=True)
    with visual_right:
        st.markdown("""
        <div style="border:1px solid #315779;border-radius:22px;padding:24px;text-align:center;
        background:radial-gradient(circle at 50% 35%,#284f7b,#0d2037 75%);">
          <svg viewBox="0 0 220 180" width="100%" role="img" aria-label="Stylized brain MRI illustration">
            <defs><radialGradient id="brainGlow"><stop offset="0%" stop-color="#b8d9ff"/><stop offset="100%" stop-color="#3d78b8"/></radialGradient></defs>
            <ellipse cx="110" cy="88" rx="79" ry="72" fill="#09192b" stroke="#68b6ff" stroke-width="3"/>
            <ellipse cx="110" cy="88" rx="62" ry="57" fill="url(#brainGlow)" opacity=".9"/>
            <path d="M110 35 C91 28 76 46 82 61 C61 62 57 83 72 95 C60 113 79 132 96 122 C103 139 117 139 124 122 C145 133 162 113 148 96 C166 80 154 62 138 60 C143 43 125 31 110 35Z" fill="#d8eaff" stroke="#eaf5ff" stroke-width="2"/>
            <path d="M109 38 L109 124 M83 57 C99 60 97 76 85 83 M136 58 C120 62 123 78 137 84 M73 94 C91 91 97 104 92 118 M149 94 C132 91 124 104 129 119 M95 45 C88 54 101 63 109 59 M126 44 C133 55 120 64 109 59" fill="none" stroke="#4d83b7" stroke-width="4" stroke-linecap="round"/>
            <circle cx="168" cy="34" r="7" fill="#65d5ff"/><circle cx="48" cy="134" r="5" fill="#8e83ff"/>
          </svg>
          <div style="font-weight:800;font-size:17px">AI-assisted MRI Research</div>
          <div style="color:#a9c6e4;font-size:12px">Academic prototype · Non-diagnostic</div>
        </div>
        """, unsafe_allow_html=True)
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

    st.markdown("---")
    st.markdown('<div class="kicker">PROJECT PURPOSE</div>', unsafe_allow_html=True)
    objective_col, abstract_col = st.columns([0.9, 1.3])
    with objective_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Project Objective")
        st.write(
            "To develop an academic CNN-based system that explores classification of "
            "brain MRI images into four dataset categories: Mild Demented, Moderate "
            "Demented, Non-Demented, and Very Mild Demented."
        )
        st.markdown("#### Key objectives")
        st.markdown(
            "- Prepare MRI images using resizing and pixel normalization.\\n"
            "- Integrate a trained CNN model into a web application.\\n"
            "- Provide an easy-to-use interface for image upload and model output review."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with abstract_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Abstract")
        st.write(
            "Alzheimer’s disease is a progressive condition that affects memory and "
            "cognitive function. This academic project explores the use of deep learning "
            "for classifying brain MRI images into four dataset-defined categories. "
            "A Convolutional Neural Network (CNN) is integrated with a Streamlit "
            "application. Uploaded images are converted to RGB, resized to 128 × 128 "
            "pixels, and normalized before inference. The application displays the "
            "predicted dataset class and the model’s output scores to support learning "
            "and experimentation. The system is a research prototype only; its outputs "
            "are not medical diagnoses and must not guide clinical decisions."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="kicker">SYSTEM METHODOLOGY</div>', unsafe_allow_html=True)
    st.subheader("How the application works")
    methodology = [
        ("01", "MRI Image Upload", "User selects a JPG, JPEG, or PNG image."),
        ("02", "Image Preprocessing", "Convert to RGB, resize to 128 × 128, and scale pixels to 0–1."),
        ("03", "CNN Inference", "Pass the processed image to the saved CNN model."),
        ("04", "Output Display", "Show the predicted dataset class and model output scores."),
    ]
    method_cols = st.columns(4)
    for col, (number, title, description) in zip(method_cols, methodology):
        with col:
            st.markdown(f"""
            <div class="team-card" style="min-height:185px">
              <div class="team-avatar">{number}</div>
              <div class="team-name">{title}</div>
              <div class="muted">{description}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Workflow reflects the current app implementation. Model output is experimental and not a clinical assessment.")

elif page == "MRI Analysis":
    render_section_hero("MRI Analysis", "Upload and preview an MRI image, run the CNN model, and inspect experimental class outputs.", HERO_ART["MRI Analysis"])
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
                preview_image = original_image.copy()
                preview_image.thumbnail((480, 480), Image.Resampling.BILINEAR)
                st.image(preview_image, use_container_width=True)
                st.caption("Preview optimized for faster display; model input is resized to 128 × 128.")
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
                        image_array = preprocess_image(uploaded_file.getvalue())
                        predictions = np.asarray(
                            model.predict(image_array, verbose=0)[0], dtype=float
                        )

                    if len(predictions) != len(CLASS_NAMES):
                        st.error(
                            "The model output does not match the four expected classes. "
                            "Please verify the model and class order."
                        )
                    else:
                        st.session_state["mri_prediction"] = predictions.tolist()
                        st.session_state["mri_prediction_class"] = CLASS_NAMES[int(np.argmax(predictions))]
                        st.session_state["mri_prediction_filename"] = uploaded_file.name

                except Exception as error:
                    st.error(f"Unable to run the model: {error}")

            # Persist results in session state so the chart and report remain available
            # after Streamlit reruns (for example, when downloading the report).
            if "mri_prediction" in st.session_state:
                saved_predictions = np.asarray(st.session_state["mri_prediction"], dtype=float)
                saved_class = st.session_state["mri_prediction_class"]
                st.markdown("---")
                st.markdown('<div class="kicker">ANALYSIS RESULT</div>', unsafe_allow_html=True)
                st.markdown(f"### Predicted dataset category: {saved_class}")
                st.info(
                    f"Highest model output score: "
                    f"**{float(np.max(saved_predictions)) * 100:.2f}%**"
                )
                st.markdown("#### Model output scores")
                score_table = {
                    "Dataset category": CLASS_NAMES,
                    "Output score (%)": [round(float(score) * 100, 2) for score in saved_predictions],
                }
                st.bar_chart(
                    {
                        "Dataset category": score_table["Dataset category"],
                        "Output score (%)": score_table["Output score (%)"],
                    },
                    x="Dataset category",
                    y="Output score (%)",
                    horizontal=True,
                )
                for class_name, score in zip(CLASS_NAMES, saved_predictions):
                    st.write(f"**{class_name}** — {float(score) * 100:.2f}%")
                    st.progress(float(np.clip(score, 0.0, 1.0)))

                report_buffer = io.StringIO()
                writer = csv.writer(report_buffer)
                writer.writerow(["MRI Classification Research Report"])
                writer.writerow(["Uploaded file", st.session_state.get("mri_prediction_filename", uploaded_file.name)])
                writer.writerow(["Predicted dataset category", saved_class])
                writer.writerow(["Class", "Model output score (%)"])
                for class_name, score in zip(CLASS_NAMES, saved_predictions):
                    writer.writerow([class_name, f"{float(score) * 100:.2f}"])
                writer.writerow([])
                writer.writerow(["Disclaimer", "Academic research only; not for medical diagnosis or treatment."])
                st.download_button(
                    "Download prediction report (CSV)",
                    data=report_buffer.getvalue(),
                    file_name="mri_classification_research_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(
                    "Scores are model outputs, not calibrated diagnostic probabilities or medical certainty."
                )
                st.warning(
                    "Research use only. Do not use this result to make medical decisions."
                )
        except Exception:
            st.error("This file could not be opened. Please upload a valid JPG, JPEG, or PNG image.")
    else:
        st.markdown(
            '<div class="panel"><h3>Ready for image upload</h3>'
            '<p class="muted">Your MRI preview and analysis results will appear here.</p></div>',
            unsafe_allow_html=True,
        )

elif page == "Model Information":
    render_section_hero("Model Information", "Explore the CNN model, preprocessing pipeline, and inference integration.", HERO_ART["Model Information"])
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

elif page == "Dataset Information":
    render_section_hero("Dataset Information", "Review dataset labels, image counts, and class distribution.", HERO_ART["Dataset Information"])
    st.markdown('<div class="kicker">DATASET OVERVIEW</div>', unsafe_allow_html=True)
    st.subheader("MRI Dataset Information")
    st.write("The model was developed using a four-class Alzheimer's MRI image dataset.")
    dataset_counts = [
        ("Mild Demented", 896),
        ("Moderate Demented", 64),
        ("Non-Demented", 3200),
        ("Very Mild Demented", 2240),
    ]
    st.markdown("### Class distribution")
    cols = st.columns(4)
    for col, (label, count) in zip(cols, dataset_counts):
        with col:
            st.metric(label, f"{count:,} images")
    st.info("Total dataset images: 6,400. The class distribution is imbalanced, especially the Moderate Demented class.")
    st.markdown("### Dataset distribution visual")
    distribution = {
        "MRI dataset class": [name for name, count in dataset_counts],
        "Number of images": [count for name, count in dataset_counts],
    }
    st.bar_chart(distribution, x="MRI dataset class", y="Number of images", horizontal=True)
    st.caption("Dataset information is provided for academic project documentation.")

elif page == "Model Performance":
    render_section_hero("Model Performance", "Review evaluation information and the limitations of experimental model outputs.", HERO_ART["Model Performance"])
    st.markdown('<div class="kicker">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.subheader("Model Performance")
    st.write("This dashboard currently does not include a verified final test-set evaluation report.")
    st.warning("Accuracy, precision, recall, F1-score, and confusion matrix are not displayed here because verified final evaluation results have not been added.")
    st.markdown("### Model configuration")
    st.markdown("- Model: Convolutional Neural Network (CNN)\n- Input: 128 × 128 × 3 RGB image\n- Classes: 4\n- Preprocessing: resize and normalize pixel values to 0–1")
    st.caption("Individual prediction scores shown in MRI Analysis are experimental model outputs, not diagnostic probabilities.")

elif page == "Project Team":
    render_section_hero("Project Team & Institution", "Meet the project members and view the academic institution details.", HERO_ART["Project Team"])
    st.markdown('<div class="kicker">PROJECT TEAM & INSTITUTION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="institution-banner">
      <div class="iilm-mark">IILM</div>
      <div>
        <div class="institution-title">IILM UNIVERSITY</div>
        <div class="institution-subtitle">Greater Noida · School of Computer Science & Engineering</div>
        <div class="institution-subtitle">Learn with Purpose, Lead with Purpose</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## Alzheimer's Disease Stage Classification Using MRI")
    st.caption("Academic major project · CNN-based MRI image classification")
    st.markdown("### Project Lead & Developer")
    st.markdown("""
    <div class="team-card">
      <div class="team-avatar">SM</div>
      <div class="team-name">Shubhanshu Mishra</div>
      <div class="team-role">Project Lead · Website Developer · Model Integration</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Project Team")
    team_data = [
        ("BS", "Brhamansh Singh"),
        ("SP", "Sumit Pathak"),
        ("K", "Khushi"),
    ]
    team_cols = st.columns(3)
    for col, (initials, member) in zip(team_cols, team_data):
        with col:
            st.markdown(f"""
            <div class="team-card">
              <div class="team-avatar">{initials}</div>
              <div class="team-name">{member}</div>
              <div class="team-role">Project Team Member</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("### Faculty Guide")
    st.markdown("""
    <div class="team-card">
      <div class="team-avatar">FK</div>
      <div class="team-name">Kanika Kansal</div>
      <div class="team-role">Faculty Guide</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Institution")
    st.markdown("**IILM University, Greater Noida**")
    st.caption("This is an academic research prototype and is not intended for clinical diagnosis.")

elif page == "About Project":
    render_section_hero("About the Project", "Project background, objectives, scope, and research disclaimer.", HERO_ART["About Project"])
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
