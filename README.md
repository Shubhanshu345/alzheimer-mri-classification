# Alzheimer's Disease Stage Classification Using MRI

An academic research prototype built with **Python, TensorFlow/Keras, CNN, and Streamlit**. The application integrates a trained CNN model to explore classification of brain MRI images into four dataset-defined categories.

> **Research disclaimer:** This application is for academic and educational purposes only. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

## Live Demo

[Open the Streamlit application](https://alzheimer-mri-classification-7ny5bqge99mq4es6divwbf.streamlit.app/)

## Project Overview

The goal of this project is to build an accessible web interface around a CNN-based MRI image classification model. Users can upload an MRI image, preview it, run model inference, and review the predicted dataset category and output scores.

## Features

- Dark navy and blue dashboard interface
- Overview page with project objective, abstract, and methodology
- MRI image upload and preview (JPG, JPEG, PNG)
- CNN inference using the saved model
- Predicted dataset category and model output scores
- Model Information and Dataset Information pages
- Model Performance page for evaluation documentation
- Project Team page with student, faculty guide, and institution details
- Research-only / non-diagnostic warning

## Dataset Categories

The model's four output labels correspond to the following dataset categories:

1. Mild Demented
2. Moderate Demented
3. Non-Demented
4. Very Mild Demented

Dataset class counts used in project documentation:

| Dataset category | Images |
|---|---:|
| Mild Demented | 896 |
| Moderate Demented | 64 |
| Non-Demented | 3,200 |
| Very Mild Demented | 2,240 |
| **Total** | **6,400** |

The dataset is imbalanced; the Moderate Demented category has fewer examples than the other categories. Dataset labels are not a substitute for clinical diagnosis.

## Methodology

1. **Upload:** Select a brain MRI image in JPG, JPEG, or PNG format.
2. **Preprocessing:** Convert the image to RGB, resize it to 128 × 128 pixels, and normalize pixel values to the 0–1 range.
3. **Inference:** Pass the processed image to the saved CNN model.
4. **Display:** Review the predicted dataset category and the model's output scores.

## Technology Stack

- Python
- Streamlit
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- NumPy
- Pillow (PIL)

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shubhanshu345/alzheimer-mri-classification.git
cd alzheimer-mri-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

Make sure the required CNN model file, `alzheimer_cnn_model.keras`, is present in the repository root alongside `app.py`.

## Streamlit Community Cloud Deployment

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Deploy this GitHub repository.
3. Select `app.py` as the main file.
4. Ensure `requirements.txt` and `alzheimer_cnn_model.keras` are available in the repository root.

## Project Team

| Name | Project role |
|---|---|
| Shubhanshu Mishra | Project lead, website development, model integration |
| Brhamansh Singh | Project team member |
| Sumit Pathak | Project team member |
| Khushi | Project team member |

**Faculty Guide:** Kanika Kansal  
**Institution:** IILM University, Greater Noida

## Limitations

- This is an academic prototype, not a validated clinical system.
- Model predictions may be incorrect and should not be interpreted as a diagnosis.
- Displayed output scores are experimental and are not calibrated diagnostic probabilities.
- Performance metrics should only be reported after evaluation on a properly held-out test set.

## License

No license has been specified yet. Contact the repository owner before reusing or redistributing this project.
