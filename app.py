import streamlit as st
import numpy as np
import keras
from PIL import Image
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

CONFIDENCE_THRESHOLD = 0.7  # or 0.6 depending on your needs

# Load model
model = keras.models.load_model("model.keras")

IMG_SIZE = (224, 224)

# Class names
class_names = ["Congested", "Uncongested"] 

# Set page config
st.set_page_config(page_title="Image Classifier", layout="centered")

# Custom title style
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Traffic Image Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image below to see its predicted class using our Traffic Classifier Deep Learning model.")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

# Preprocessing image manually before passing to model
def preprocess_image(image: Image.Image):
    img = image
    img = img.resize(IMG_SIZE)
    img_array = img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  
    return img_array


# Prediction logic
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)  
        prediction = prediction[0][0] * 100


    # Define threshold 
    CONFIDENCE_THRESHOLD = 0.5

    # Interpret result
    if prediction >= CONFIDENCE_THRESHOLD:
        st.markdown(f"<h3 style='color:red;'>🚗 Congested</h3>", unsafe_allow_html=True)

    elif prediction <= (1 - CONFIDENCE_THRESHOLD):
        st.markdown(f"<h3 style='color:green;'>✅ Clear </h3>", unsafe_allow_html=True)

    else:
        st.markdown(f"<h3 style='color:orange;'>⚠️ Uncertain ({prediction * 100:.2f}% congested confidence)</h3>", unsafe_allow_html=True)
