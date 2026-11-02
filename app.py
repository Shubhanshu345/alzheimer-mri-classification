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


# ---------------- Full-width photo banners for dashboard sections ----------------
SECTION_BANNERS = {
    "Overview": "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?auto=format&fit=crop&w=1800&q=85",
    "MRI Analysis": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1800&q=85",
    "Model Information": "https://images.unsplash.com/photo-1634017839464-5c339ebe3cb4?auto=format&fit=crop&w=1800&q=85",
    "Dataset Information": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1800&q=85",
    "Model Performance": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=1800&q=85",
    "Project Team": "https://images.unsplash.com/photo-1521737711867-e3b97375f902?auto=format&fit=crop&w=1800&q=85",
    "About Project": "https://images.unsplash.com/photo-1532094349884-543bc11b234d?auto=format&fit=crop&w=1800&q=85",
}

def render_section_hero(title, subtitle, image_url):
    st.markdown(f"""
    <div style="width:100%;min-height:300px;display:flex;align-items:center;padding:36px 42px;
    margin:0 0 28px;border-radius:24px;overflow:hidden;border:1px solid #315779;
    background-image:linear-gradient(90deg,rgba(5,16,31,.94) 0%,rgba(7,24,43,.82) 43%,rgba(7,24,43,.18) 100%),url('{image_url}');
    background-position:center;background-size:cover;background-repeat:no-repeat;
    box-shadow:0 14px 40px rgba(0,0,0,.28);">
      <div style="max-width:720px">
        <div style="font-size:12px;letter-spacing:2.4px;text-transform:uppercase;color:#8dccff;font-weight:800">ALZHEIMER'S MRI · RESEARCH DASHBOARD</div>
        <div style="font-size:clamp(30px,4vw,48px);line-height:1.12;font-weight:900;color:#f7fbff;margin:14px 0 12px;text-shadow:0 2px 12px rgba(0,0,0,.35)">{title}</div>
        <div style="font-size:16px;line-height:1.6;color:#e1efff;max-width:650px;text-shadow:0 1px 8px rgba(0,0,0,.45)">{subtitle}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

if page == "Overview":
    render_section_hero("Alzheimer's MRI Classification", "AI-powered academic research prototype for exploring brain MRI image classification.", SECTION_BANNERS["Overview"])
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
    render_section_hero("MRI Analysis", "Upload and preview an MRI image, run the CNN model, and inspect experimental class outputs.", SECTION_BANNERS["MRI Analysis"])
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
    render_section_hero("Model Information", "Explore the CNN model, preprocessing pipeline, and inference integration.", SECTION_BANNERS["Model Information"])
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
    render_section_hero("Dataset Information", "Review dataset labels, image counts, and class distribution.", SECTION_BANNERS["Dataset Information"])
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
    render_section_hero("Model Performance", "Review evaluation information and the limitations of experimental model outputs.", SECTION_BANNERS["Model Performance"])
    st.markdown('<div class="kicker">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.subheader("Model Performance")
    st.write("This dashboard currently does not include a verified final test-set evaluation report.")
    st.warning("Accuracy, precision, recall, F1-score, and confusion matrix are not displayed here because verified final evaluation results have not been added.")
    st.markdown("### Model configuration")
    st.markdown("- Model: Convolutional Neural Network (CNN)\n- Input: 128 × 128 × 3 RGB image\n- Classes: 4\n- Preprocessing: resize and normalize pixel values to 0–1")
    st.caption("Individual prediction scores shown in MRI Analysis are experimental model outputs, not diagnostic probabilities.")

elif page == "Project Team":
    render_section_hero("Project Team & Institution", "Meet the project members and view the academic institution details.", SECTION_BANNERS["Project Team"])
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
    render_section_hero("About the Project", "Project background, objectives, scope, and research disclaimer.", SECTION_BANNERS["About Project"])
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
