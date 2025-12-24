# ---------------------------------
# Load Model (Keras .keras — SAFE)
# ---------------------------------
@st.cache_resource
def load_model():
    """
    Tries multiple reasonable locations so deployment
    doesn't break if folder structure changes.
    """

    # Possible locations to search
    search_paths = [
        os.getcwd(),                              # project root
        os.path.dirname(os.path.abspath(__file__)),  # src/ or app/
        os.path.join(os.getcwd(), "src"),
        os.path.join(os.getcwd(), "app"),
    ]

    model_name = "cbis_ddsm_streamlit_safe.keras"

    for base in search_paths:
        candidate = os.path.join(base, model_name)
        if os.path.exists(candidate):
            st.info(f"✅ Loaded model from: {candidate}")
            return tf.keras.models.load_model(
                candidate,
                compile=False
            )

    # If we reach here → model not found anywhere
    raise FileNotFoundError(
        f"""
❌ Could not find {model_name}

Searched locations:
{chr(10).join(search_paths)}

Files in cwd:
{os.listdir(os.getcwd())}
"""
    )
