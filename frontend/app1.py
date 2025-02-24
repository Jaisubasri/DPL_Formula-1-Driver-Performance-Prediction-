import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import tempfile

# Title
st.title("📈 LSTM Model Loader (.keras)")

# File uploader for .keras files
model_file = st.file_uploader("📂 Upload LSTM Model (.keras)", type=["keras"])

if model_file:
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
            tmp_file.write(model_file.read())
            tmp_model_path = tmp_file.name

        # Load the .keras model with custom_objects
        model = load_model(tmp_model_path, custom_objects={'Orthogonal': Orthogonal})

        st.success("✅ Model loaded successfully!")
        st.write("Model Summary:")
        model.summary(print_fn=lambda x: st.text(x))

    except Exception as e:
        st.error(f"❌ Failed to load the model. Error: {e}")
