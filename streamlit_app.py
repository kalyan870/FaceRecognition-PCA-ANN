import os
import sys
import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_dataset
from src.pca import PCA
from src.ann_model import ANNModel
from src.train import train_pipeline
from src.test import preprocess_single_image
from src.eigenfaces import save_eigenfaces, save_explained_variance
from src.utils import plot_accuracy_vs_k

st.set_page_config(page_title="Face Recognition - PCA + ANN", layout="wide")

st.title(" Face Recognition using PCA (Eigenfaces) + ANN")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Train & Evaluate", "Test Single Image", "Test Multiple Images"])
k_values_input = st.sidebar.text_input("k values (comma separated)", "5,10,20,30,40")
k_values = [int(k.strip()) for k in k_values_input.split(",") if k.strip()]

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMPOSTER_DIR = "dataset/imposters"

# Session state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "pca" not in st.session_state:
    st.session_state.pca = None
if "ann" not in st.session_state:
    st.session_state.ann = None
if "label_map" not in st.session_state:
    st.session_state.label_map = None
if "accuracy_data" not in st.session_state:
    st.session_state.accuracy_data = None

def train():
    with st.spinner("Training in progress..."):
        pca, ann, label_map, k_vals, accs = train_pipeline(TRAIN_DIR, TEST_DIR, k_values, save_models=True)
        st.session_state.pca = pca
        st.session_state.ann = ann
        st.session_state.label_map = label_map
        st.session_state.trained = True
        st.session_state.accuracy_data = (k_vals, accs)
    st.success("Training complete!")

def load_trained_model():
    if st.session_state.trained:
        return st.session_state.pca, st.session_state.ann, st.session_state.label_map
    if os.path.exists("outputs/models/final_pca.joblib") and os.path.exists("outputs/models/final_ann.joblib"):
        pca = joblib.load("outputs/models/final_pca.joblib")
        ann = joblib.load("outputs/models/final_ann.joblib")
        _, _, _, _, label_map, _, _ = load_dataset(TRAIN_DIR, TEST_DIR)
        st.session_state.pca = pca
        st.session_state.ann = ann
        st.session_state.label_map = label_map
        st.session_state.trained = True
        return pca, ann, label_map
    return None, None, None

def predict_image(image_file, pca, ann, label_map):
    bytes_data = image_file.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img_resized = cv2.resize(img, (64, 64))
    img_vec = img_resized.flatten().astype(np.float64)
    img_pca = pca.transform(img_vec.reshape(1, -1))
    results = ann.predict_with_confidence(img_pca, threshold=0.6)
    return results[0], img_resized

# Main content
if mode == "Train & Evaluate":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Training")
        if st.button(" Start Training", type="primary"):
            train()

        if st.session_state.trained:
            st.success("Model is trained and ready!")

    with col2:
        if os.path.exists("outputs/graphs/accuracy_vs_k.png"):
            st.subheader("Accuracy vs k")
            st.image("outputs/graphs/accuracy_vs_k.png")

        if os.path.exists("outputs/graphs/explained_variance.png"):
            st.subheader("Explained Variance")
            st.image("outputs/graphs/explained_variance.png")

    # Eigenfaces display
    if os.path.exists("outputs/eigenfaces/eigenfaces_grid.png"):
        st.subheader("Eigenfaces")
        st.image("outputs/eigenfaces/eigenfaces_grid.png")

    if os.path.exists("outputs/eigenfaces/mean_face.png"):
        st.subheader("Mean Face")
        st.image("outputs/eigenfaces/mean_face.png")

    # Accuracy table
    if st.session_state.accuracy_data:
        k_vals, accs = st.session_state.accuracy_data
        st.subheader("Accuracy Results")
        acc_table = {f"k={k}": f"{acc:.2f}%" for k, acc in zip(k_vals, accs)}
        st.table(acc_table)

elif mode == "Test Single Image":
    st.subheader("Test a Single Face Image")

    pca, ann, label_map = load_trained_model()

    if not st.session_state.trained:
        st.warning("No trained model found. Train first or check model files.")
        if st.button("Train Now"):
            train()
            pca, ann, label_map = st.session_state.pca, st.session_state.ann, st.session_state.label_map

    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png", "pgm", "bmp"])

    if uploaded_file and st.session_state.trained:
        result, img_disp = predict_image(uploaded_file, pca, ann, label_map)
        if result:
            label, confidence = result
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_disp, caption="Input Image", width=250, channels="gray")
            with col2:
                if "Unknown" in label:
                    st.error(f"**Prediction:** {label}")
                else:
                    st.success(f"**Prediction:** {label}")
                st.info(f"**Confidence:** {confidence:.2%}")
        else:
            st.error("Could not process the image. Please try another.")

elif mode == "Test Multiple Images":
    st.subheader("Test Multiple Images")

    pca, ann, label_map = load_trained_model()

    if not st.session_state.trained:
        st.warning("No trained model found. Train first or check model files.")
        if st.button("Train Now"):
            train()
            pca, ann, label_map = st.session_state.pca, st.session_state.ann, st.session_state.label_map

    uploaded_files = st.file_uploader("Choose face images...", type=["jpg", "jpeg", "png", "pgm", "bmp"], accept_multiple_files=True)

    if uploaded_files and st.session_state.trained:
        results_data = []
        cols = st.columns(3)
        for i, f in enumerate(uploaded_files):
            result, img_disp = predict_image(f, pca, ann, label_map)
            if result:
                label, confidence = result
                results_data.append({"Image": f.name, "Prediction": label, "Confidence": f"{confidence:.2%}"})
                with cols[i % 3]:
                    st.image(img_disp, caption=f.name, width=150, channels="gray")
                    if "Unknown" in label:
                        st.error(f"{label}")
                    else:
                        st.success(f"**{label}**")
                    st.caption(f"Confidence: {confidence:.2%}")

        if results_data:
            st.subheader("Results Summary")
            st.table(results_data)

# Footer
st.markdown("---")
st.markdown("###  Face Recognition using PCA (Eigenfaces) + ANN")
st.markdown("Built with Python, OpenCV, NumPy, Scikit-learn, Matplotlib, Streamlit")
