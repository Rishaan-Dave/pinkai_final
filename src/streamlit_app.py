import os
import io
import base64
import random
import string

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="PINKAI Mammogram Analyzer",
    layout="centered"
)

# =================================================
# APP BOOTSTRAP (UI MUST RENDER FIRST)
# =================================================
st.write("🟢 App online — please be patient")

st.title("PINKAI Mammogram Analyzer")

st.markdown("""
Upload or paste a mammogram image to receive:
- **AI-based malignancy classification**
- **Confidence scores**

⚠️ Research & decision support only — not a clinical diagnosis.
""")

# =================================================
# INPUT METHODS
# =================================================
uploaded_file = st.file_uploader(
    "Upload a mammogram image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

st.markdown("**OR (if upload fails): paste image as Base64**")
base64_input = st.text_area(
    "Paste base64 image string here (advanced / fallback)",
    height=120
)

# =================================================
# MODEL LOADER (HF + STREAMLIT SAFE)
# =================================================
@st.cache_resource
def load_model():
    cwd = os.getcwd()

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        file_dir = None

    search_paths = [
        cwd,
        os.path.join(cwd, "src"),
        os.path.join(cwd, "app"),
    ]

    if file_dir:
        search_paths.insert(1, file_dir)

    model_name = "cbis_ddsm_streamlit_safe.keras"

    for base in search_paths:
        candidate = os.path.join(base, model_name)
        if os.path.exists(candidate):
            return tf.keras.models.load_model(candidate, compile=False)

    st.error("❌ Model file not found.")
    st.stop()

# =================================================
# IMAGE HANDLING
# =================================================
def random_filename(ext="png", length=12):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length)) + f".{ext}"

def load_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image, random_filename("png")

def load_image_from_base64(b64_string):
    try:
        decoded = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
        return image, random_filename("png")
    except Exception:
        return None, None

# =================================================
# PREDICTION LOGIC
# =================================================
CLASS_MAPPING = {0: "Benign", 1: "Malignant"}

def predict(image, model):
    img = np.array(image)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    idx = int(np.argmax(preds[0]))

    return CLASS_MAPPING[idx], preds[0]

# =================================================
# INFERENCE PIPELINE
# =================================================
image = None
safe_name = None

if uploaded_file is not None:
    image, safe_name = load_image_from_upload(uploaded_file)

elif base64_input.strip():
    image, safe_name = load_image_from_base64(base64_input.strip())
    if image is None:
        st.error("❌ Invalid base64 image.")
        st.stop()

if image is not None:
    st.image(
        image,
        caption=f"Processed Image ({safe_name})",
        use_column_width=True
    )

    with st.spinner("Loading AI model..."):
        model = load_model()

    with st.spinner("Analyzing mammogram..."):
        label, probs = predict(image, model)

    st.success(f"🧠 Prediction: **{label}**")

    st.subheader("Confidence Scores")
    for i, name in CLASS_MAPPING.items():
        st.write(f"{name}: **{probs[i] * 100:.2f}%**")
