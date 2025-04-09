import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# Load model
model = keras.models.load_model("model.keras")

# Class names
class_names = ["Congested", "Uncongested"]  # Replace with your actual class labels

# Set page config
st.set_page_config(page_title="Image Classifier", layout="centered")

# Custom title style
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Traffic Image Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image below to see its predicted class using our Traffic Classifier Deep Learning model.")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction logic
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0]
        predicted_index = np.argmin(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

    st.success(f"✅ Prediction: **{predicted_class}**")
    st.progress(min(int(confidence), 100))
    st.write(f"🧪 Confidence: **{confidence:.2f}%**")
