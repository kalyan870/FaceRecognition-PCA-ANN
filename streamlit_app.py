import streamlit as st
import numpy as np
import joblib
from PIL import Image

model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")
label_names = joblib.load("labels.pkl")

st.title("PCA + ANN Face Recognition")

uploaded_file = st.file_uploader("Upload Face Image")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image.resize((100, 100)))

    st.image(image, caption="Uploaded Image", width=250)

    image_flatten = image.flatten().reshape(1, -1)
    image_pca = pca.transform(image_flatten)
    prediction = model.predict(image_pca)
    predicted_person = label_names[prediction[0]]

    st.success(f"Predicted Person: {predicted_person}")
