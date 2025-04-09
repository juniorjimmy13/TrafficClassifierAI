import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

# Load your model
model = keras.models.load_model("model.keras")

# Preprocessing function (adjust to your model's expected input)
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # or your model's input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit UI
st.title("Keras Model Inference (.keras)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    processed = preprocess_image(image)
    prediction = model.predict(processed)

    # Display result
    class_names = ["Congested", "Uncongested"]  # Change this to match your labels
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"### Prediction: {predicted_class}")
