import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# App Bootstrap (UI MUST RENDER FIRST)
# ---------------------------------
st.write("✅ App started")

st.title("PINKAI Mammogram Analyzer")

st.markdown("""
Upload a mammogram image to receive:
- AI-based malignancy classification
- Confidence scores

⚠️ This tool is for research and decision support only.
""")

uploaded_file = st.file_uploader(
    "Upload a mammogram image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

# ---------------------------------
# Load Model (Keras .keras — SAFE)
# ---------------------------------
@st.cache_resource
def load_model():
    """
    Robust model loader that works locally, on Hugging Face,
    and in Streamlit Cloud without blocking UI.
    """

    cwd = os.getcwd()

    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        file_dir = None  # __file__ not available (HF Spaces)

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

# ---------------------------------
# Prediction Logic
# ---------------------------------
CLASS_MAPPING = {0: "Benign", 1: "Malignant"}

def predict(image, model):
    img = np.array(image)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = int(np.argmax(preds[0]))

    return CLASS_MAPPING[idx], preds[0]

# ---------------------------------
# Inference Trigger (LAZY LOAD)
# ---------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Mammogram", use_column_width=True)

    with st.spinner("Loading AI model..."):
        model = load_model()

    with st.spinner("Analyzing mammogram..."):
        label, probs = predict(image, model)

    st.success(f"🧠 Prediction: **{label}**")

    st.subheader("Confidence Scores")
    for i, name in CLASS_MAPPING.items():
        st.write(f"{name}: **{probs[i] * 100:.2f}%**")
