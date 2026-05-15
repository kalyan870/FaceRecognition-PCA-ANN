import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")
label_names = list(joblib.load("labels.pkl"))

st.set_page_config(page_title="PCA + ANN Face Recognition", layout="wide")

st.sidebar.title("Face Recognition System")
st.sidebar.write("PCA + ANN based ML project")
st.sidebar.markdown("---")
st.sidebar.metric("Model Accuracy", "100%")
st.sidebar.metric("Total Persons", str(len(label_names)))
st.sidebar.metric("Images per Person", "10")
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Classes")
for name in label_names:
    st.sidebar.write(f"- {name}")
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This project uses **PCA** (Principal Component Analysis)
for dimensionality reduction and **ANN** (Artificial Neural Network)
for face classification.
""")

st.title("PCA + ANN Face Recognition")
st.markdown("Upload a face image to recognize the person using PCA dimensionality reduction and ANN classification.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image.resize((100, 100)))

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

st.subheader("Eigenfaces Visualization")
eigenface_path = "outputs/eigenfaces/eigenface_0.png"
if os.path.exists(eigenface_path):
    st.image(eigenface_path, caption="Top Eigenface (PCA Components)", width=400)
else:
    st.info("Run train_model.py to generate eigenfaces.")

col3, col4 = st.columns(2)
with col3:
    graph_path = "outputs/graphs/accuracy_graph.png"
    if os.path.exists(graph_path):
        st.image(graph_path, caption="Model Accuracy", width=400)

st.markdown("---")
st.markdown("""
### About
This project uses **PCA** (Principal Component Analysis)
for dimensionality reduction and **ANN** (Artificial Neural Network)
for face classification. PCA extracts the most important features
(Eigenfaces) from face images, and the ANN classifies them into
recognized persons.
""")

st.markdown("---")
st.caption("Developed by Saragadam Kalyan")
