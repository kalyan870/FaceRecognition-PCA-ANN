import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from src.runtime import normalize_label_names, square_image_size

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

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(PCA_PATH),
        normalize_label_names(joblib.load(LABELS_PATH)),
    )


try:
    classifier, pca, label_names = load_artifacts()
    image_side = square_image_size(int(pca.n_features_in_))
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
    img_resized = cv2.resize(img_array, (image_side, image_side))

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
    class_index = int(prediction)
    if class_index < 0 or class_index >= len(label_names):
        st.error("The model prediction does not match the labels artifact.")
        st.stop()
    pred_name = label_names[class_index]

    # Display Image
    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    # Results
    st.success(f"Predicted Person: {pred_name}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.caption("Prediction confidence is the classifier's probability estimate, not a validated real-world recognition accuracy.")

    # Probability Scores
    st.subheader("Class Probabilities")

    for i, prob in enumerate(probabilities):
        class_id = int(classifier.classes_[i])
        if 0 <= class_id < len(label_names):
            st.write(f"{label_names[class_id]} : {prob * 100:.2f}%")
