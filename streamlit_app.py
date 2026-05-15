import streamlit as st
import numpy as np
import joblib
from PIL import Image

MODEL_PATH = "model.pkl"
PCA_PATH = "pca.pkl"
LABELS_PATH = "labels.pkl"

st.set_page_config(page_title="PCA + ANN Face Recognition", layout="centered")

st.title("PCA + ANN Face Recognition")
st.write("Upload a face image to recognize the person.")

try:
    classifier = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    label_names, _, _, _, _ = joblib.load(LABELS_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    img_resized = np.array(image.resize((100, 100))).flatten().reshape(1, -1)
    img_pca = pca.transform(img_resized)
    prediction = classifier.predict(img_pca)[0]
    probabilities = classifier.predict_proba(img_pca)[0]
    confidence = np.max(probabilities) * 100

    pred_name = label_names[prediction]

    st.image(image, caption="Uploaded Image", width=250)
    st.success(f"**Predicted Person:** {pred_name}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")
    prob_dict = {label_names[i]: float(probabilities[i]) for i in range(len(label_names))}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    for name, prob in sorted_probs:
        st.write(f"{name}: {prob*100:.2f}%")
