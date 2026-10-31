# Alzheimer's MRI Classification

An academic Streamlit application using the existing CNN model to classify brain MRI images into four dataset categories.

## Dashboard features
- Premium dark navy and electric-blue interface
- Sidebar navigation: Overview, MRI Analysis, Model Information, About Project
- MRI image upload and preview
- CNN inference using the existing `alzheimer_cnn_model.keras` model
- Predicted class and model output scores for all four categories
- Research-only and non-diagnostic notices

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit deployment
Keep `app.py`, `requirements.txt`, and `alzheimer_cnn_model.keras` in the repository root. In Streamlit Community Cloud, deploy the same GitHub repository and select `app.py` as the main file.

## Important
This is an academic research prototype, not a medical device. Do not use it for diagnosis, treatment, or clinical decisions. Model output scores are experimental and are not calibrated diagnostic probabilities.
