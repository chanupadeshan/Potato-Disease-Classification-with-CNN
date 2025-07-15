import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
MODEL = tf.keras.models.load_model("../models/potato_disease_classification_v1.0.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# --- Streamlit UI setup ---
st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")

# Center the entire content
st.markdown("""
    <style>
    .block-container {
        max-width: 1100px;
        margin: auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div style='text-align:center;'>
        <h1>🥔 Potato Disease Classifier</h1>
        <p>Upload a potato leaf image and let AI detect the condition (Healthy, Early Blight, or Late Blight).</p>
    </div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📷 Input Image")
        st.image(uploaded_file, width=350, caption="Potato Leaf")

    with col2:
        st.markdown("### 🔎 Prediction Result")
        image_data = read_file_as_image(uploaded_file)
        prediction = MODEL.predict(image_data)

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0])) * 100

        st.markdown(f"""
            <div style='background-color:#262730; padding:20px; border-radius:10px;'>
                <h3 style='color:#f63366;'>🧠 Condition: <span style='color:white;'>{predicted_class}</span></h3>
                <h4 style='color:#00FFAA;'>✅ Confidence: <span style='color:white;'>{confidence:.2f}%</span></h4>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("🔬 *This model detects: Early Blight, Late Blight, and Healthy potato leaves.*")
