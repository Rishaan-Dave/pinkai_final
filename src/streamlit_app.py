import os
import streamlit as st
import tensorflow as tf

# ---------------------------------
# Load Model (Keras .keras — SAFE)
# ---------------------------------
st.write("✅ App started")

@st.cache_resource
def load_model():
    """
    Robust model loader that works locally, on Hugging Face,
    and in Streamlit Cloud without blocking UI.
    """

    # Resolve base directories safely
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
            return tf.keras.models.load_model(
                candidate,
                compile=False
            )

    # Hard fail with debug info
    raise FileNotFoundError(
        f"""
❌ Could not find {model_name}

Searched paths:
{chr(10).join(search_paths)}

Files in working directory:
{os.listdir(cwd)}
"""
    )
