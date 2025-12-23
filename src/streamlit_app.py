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
# Load Model (Keras 3 SAFE)
# ---------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "cbis_ddsm_best_model.keras")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,      # 🔥 critical fix
        safe_mode=False     # 🔥 avoids Keras 3 layer bug
    )
    return model

model = load_model()

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict(image, model):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_mapping[predicted_index]

    return predicted_label, predictions[0]

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("Mammogram Breast Cancer Classification")

st.markdown(
    """
    Upload a **mammogram image** and the AI model will classify it as:
    - **Benign**
    - **Malignant**

    Model trained on **CBIS-DDSM (DenseNet-based CNN)**.
    """
)

uploaded_file = st.file_uploader(
    "Upload a mammogram image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Mammogram", use_column_width=True)

    with st.spinner("Analyzing mammogram..."):
        predicted_class, probs = predict(image, model)

    st.success(f"🧠 Prediction: **{predicted_class}**")

    st.subheader("Confidence Scores")
    for i, label in class_mapping.items():
        st.write(f"{label}: **{probs[i] * 100:.2f}%**")
