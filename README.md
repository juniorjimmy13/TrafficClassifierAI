# **Traffic Image Classifier**

# Overview
The Traffic Image Classifier is a deep learning model built to predict traffic congestion levels in images. It uses a MobileNetV2 model trained to classify images into two categories: Congested and Uncongested.
This repository contains the code for uploading images, preprocessing them, and using a trained model to predict traffic congestion. The user can upload an image via a web interface and receive real-time predictions with confidence scores.

# Features
Real-time image classification: Classify traffic images as "Congested" or "Uncongested."

Confidence thresholds: Display predictions with associated confidence scores.

Simple web interface: Upload an image and view results directly in your browser.

MobileNetV2 architecture: Leveraging a lightweight and efficient pre-trained model for fast predictions.

# Demo
You can view the live demo here : https://trafficclassifierai-gxxu6roksapppmgfpwvpngh.streamlit.app/

# Technologies Used
* Python: Core programming language.
* Streamlit: Web framework for interactive data apps.
* TensorFlow/Keras: For building and deploying the deep learning model.
* MobileNetV2: Pre-trained model used for image classification.
* NumPy: For handling image data and array manipulations.

# Model Details
This model uses the MobileNetV2 architecture, which is a lightweight neural network optimized for mobile and embedded devices. It has been trained to predict traffic congestion based on images. The dataset used for training was sourced from: https://data.mendeley.com/datasets/wtp4ssmwsd/1.

# Screenshots from the project
![image](https://github.com/user-attachments/assets/516a1ac9-365a-4ec4-9428-deda71793555)

![image](https://github.com/user-attachments/assets/fa418033-a5d8-4cbe-9e89-67789c53b045)

## Getting Started 

To get a local copy of this project running follow the steps below

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Installation

Clone the repository:
```bash
git clone https://github.com/juniorjimmy13/traffic-image-classifier.git
cd traffic-image-classifier
```
Create a Virtual Enviroment this is reccomended
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
streamlit run app.py
```


