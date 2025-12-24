import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Mammogram Breast Cancer Classifier",
    layout="centered"
)

# ---------------------------------
# Class Mapping
# ---------------------------------
class_mapping = {
    0: "Benign",
    1: "Malignant"
}

# ---------------------------------
# Load Model (CORRECT for compatible model)
# ---------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "cbis_ddsm_final_model.keras"
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

model = load_model()

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict(image, model):
    img = np.array(image)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds[0])

    return class_mapping[idx], preds[0]

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("Mammogram Breast Cancer Classification")

st.markdown(
    """
    Upload a **mammogram image** and the AI model will classify it as:
    - **Benign**
    - **Malignant**

    Model trained on **CBIS-DDSM using DenseNet121**.
    """
)

uploaded_file = st.file_uploader(
    "Upload a mammogram image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Mammogram", use_column_width=True)

    with st.spinner("Analyzing mammogram..."):
        label, probs = predict(image, model)

    st.success(f"🧠 Prediction: **{label}**")

    st.subheader("Confidence Scores")
    for i, name in class_mapping.items():
        st.write(f"{name}: **{probs[i] * 100:.2f}%**")
