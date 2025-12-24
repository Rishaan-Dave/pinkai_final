import os
import io
import random
import string

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =================================================
# APP BOOTSTRAP (UI MUST RENDER FIRST)
# =================================================
st.write("app is on mumma have patience")

st.title("PINKAI Mammogram Analyzer")

st.markdown("""
Upload a mammogram image to receive:
- AI-based malignancy classification
- Confidence scores

""")

uploaded_file = st.file_uploader(
    "Upload a mammogram image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

# =================================================
# MODEL LOADER (HF + STREAMLIT SAFE)
# =================================================
@st.cache_resource
def load_model():
    cwd = os.getcwd()

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
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

    raise FileNotFoundError(
        f"""
❌ Could not find {model_name}

Searched paths:
{chr(10).join(search_paths)}

Files in working directory:
{os.listdir(cwd)}
"""
    )

# =================================================
# IMAGE NORMALIZATION (AUTO-CONVERT + SAFE NAME)
# =================================================
def random_filename(ext="png", length=12):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length)) + f".{ext}"


def normalize_uploaded_image(uploaded_file):
    raw_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    safe_name = random_filename("png")
    return image, safe_name

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
# INFERENCE (LAZY + SAFE)
# =================================================
if uploaded_file is not None:

    try:
        image, safe_name = normalize_uploaded_image(uploaded_file)
    except Exception:
        st.error("❌ The uploaded file could not be processed as an image.")
        st.stop()

    st.image(
        image,
        caption=f"Uploaded Mammogram ({safe_name})",
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
