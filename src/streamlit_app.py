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
# Class Mapping (CBIS-DDSM)
# ---------------------------------
class_mapping = {
    0: "Benign",
    1: "Malignant"
}

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cbis_ddsm_best_model.keras")
    return model

model = load_model()

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict(image, model):
    img_array = np.array(image)

    # Resize to model input size
    img_array = tf.image.resize(img_array, (256, 256))

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = tf.expand_dims(img_array, axis=0)

    # Predict
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

    This model is trained using the **CBIS-DDSM dataset** and DenseNet121.
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
        predicted_class, probs = predict(image, model)

    st.success(f"🧠 Prediction: **{predicted_class}**")

    st.subheader("Confidence Scores")
    for i, label in class_mapping.items():
        st.write(f"{label}: **{probs[i] * 100:.2f}%**")
