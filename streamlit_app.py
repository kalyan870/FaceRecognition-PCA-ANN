import streamlit as st
import numpy as np
import joblib
from PIL import Image

model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")
label_names = joblib.load("labels.pkl")

st.set_page_config(page_title="PCA + ANN Face Recognition", layout="wide")

st.sidebar.title("Face Recognition System")
st.sidebar.write("PCA + ANN Based AI Project")
st.sidebar.markdown("---")
st.sidebar.info("Model Accuracy: 100%")
st.sidebar.markdown("---")
st.sidebar.write("**Dataset:** 6 persons, 10 images each")
st.sidebar.write("**PCA Components:** 39")
st.sidebar.write("**Classifier:** MLP Neural Network")

st.title("PCA + ANN Face Recognition")
st.markdown("Upload a face image to recognize the person using PCA dimensionality reduction and ANN classification.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image.resize((100, 100)))

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    image_flatten = image.flatten().reshape(1, -1)
    image_pca = pca.transform(image_flatten)
    prediction = model.predict(image_pca)
    predicted_person = label_names[prediction[0]]

    with col2:
        st.success(f"**Predicted Person:** {predicted_person}")
        st.balloons()

st.markdown("---")
st.caption("Developed by Saragadam Kalyan")
