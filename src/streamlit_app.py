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
# UI BOOTSTRAP (RENDER FIRST)
# =================================================
st.title("PINKAI Mammogram Analyzer")

st.markdown("""
Upload a mammogram image to receive:
- **AI-based malignancy classification**
- **Confidence scores**

⚠️ Research & decision support only — not a clinical diagnosis.
""")

uploaded_file = st.file_uploader(
    "Upload mammogram image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

# =================================================
# MODEL LOADER (ROBUST)
# =================================================
@st.cache_resource
def load_model():
    model_name = "cbis_ddsm_streamlit_safe.keras"

    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), "src"),
        os.path.join(os.getcwd(), "app"),
    ]

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.insert(1, file_dir)
    except Exception:
        pass

    for base in search_paths:
        candidate = os.path.join(base, model_name)
        if os.path.exists(candidate):
            return tf.keras.models.load_model(candidate, compile=False)

    st.error("❌ Model file not found.")
    st.stop()

# =================================================
# IMAGE → BASE64 → IMAGE (FIREWALL SAFE)
# =================================================
def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(b64_string: str) -> Image.Image:
    decoded = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(decoded)).convert("RGB")

def random_filename(ext="png", length=10):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length)) + f".{ext}"

# =================================================
# PREDICTION LOGIC
# =================================================
CLASS_MAPPING = {0: "Benign", 1: "Malignant"}

def predict(image: Image.Image, model):
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
if uploaded_file is not None:

    try:
        # Load original image
        original_image = Image.open(uploaded_file).convert("RGB")

        # 🔐 Firewall-safe step
        encoded = image_to_base64(original_image)
        processed_image = base64_to_image(encoded)

        safe_name = random_filename()

    except Exception:
        st.error("❌ Invalid image file.")
        st.stop()

    st.image(
        processed_image,
        caption=f"Processed Mammogram ({safe_name})",
        use_column_width=True
    )

    with st.spinner("Loading AI model..."):
        model = load_model()

    with st.spinner("Analyzing mammogram..."):
        label, probs = predict(processed_image, model)

    st.success(f"🧠 Prediction: **{label}**")

    st.subheader("Confidence Scores")
    for i, name in CLASS_MAPPING.items():
        st.write(f"{name}: **{probs[i] * 100:.2f}%**")
