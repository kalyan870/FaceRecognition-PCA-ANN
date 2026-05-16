import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# File Paths
MODEL_PATH = "model.pkl"
PCA_PATH = "pca.pkl"
LABELS_PATH = "labels.pkl"

# Page Settings
st.set_page_config(
    page_title="PCA + ANN Face Recognition",
    layout="centered"
)

# Title
st.title("PCA + ANN Face Recognition")
st.write("Upload a face image to recognize the person.")

# Load Models
try:
    classifier = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)

    # Corrected labels loading
    loaded_labels = joblib.load(LABELS_PATH)

    # Handle tuple/list safely
    if isinstance(loaded_labels, tuple) or isinstance(loaded_labels, list):
        label_names = loaded_labels[0]
    else:
        label_names = loaded_labels

    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload Image
uploaded_file = st.file_uploader(
    "Choose a face image...",
    type=["jpg", "jpeg", "png"]
)

# Prediction
if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("L")

    # Convert to NumPy
    img_array = np.array(image)

    # Resize
    img_resized = cv2.resize(img_array, (100, 100))

    # Flatten
    img_flatten = img_resized.flatten().reshape(1, -1)

    # PCA Transform
    img_pca = pca.transform(img_flatten)

    # Predict
    prediction = classifier.predict(img_pca)[0]

    # Probabilities
    probabilities = classifier.predict_proba(img_pca)[0]

    # Confidence
    confidence = np.max(probabilities) * 100

    # Predicted Name
    pred_name = label_names[prediction]

    # Display Image
    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    # Results
    st.success(f"Predicted Person: {pred_name}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Probability Scores
    st.subheader("Class Probabilities")

    for i, prob in enumerate(probabilities):
        st.write(f"{label_names[i]} : {prob * 100:.2f}%")