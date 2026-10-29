
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Alzheimer MRI Classification",
    page_icon="🧠"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alzheimer_cnn_model.keras")

model = load_model()

class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non-Demented",
    "Very Mild Demented"
]

st.title("Alzheimer's Disease Stage Classification")
st.write("Upload an MRI image to see the model's experimental prediction.")

st.warning(
    "This is an academic research prototype, not a medical diagnostic tool."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    if st.button("Classify MRI"):
        image = image.resize((128, 128))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        st.subheader("Experimental Prediction")
        st.write(f"Predicted class: **{class_names[predicted_index]}**")
        st.write(
            f"Model score: {predictions[predicted_index] * 100:.2f}%"
        )
        st.caption(
            "This score is not diagnostic certainty. Do not use it "
            "to make medical decisions."
        )
