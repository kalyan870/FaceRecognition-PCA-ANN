# PCA + ANN Face Recognition System

Built and deployed a PCA + ANN Face Recognition System using Python, OpenCV, Scikit-learn, and Streamlit.

The project performs facial recognition using Principal Component Analysis (PCA) for dimensionality reduction and an Artificial Neural Network (ANN) for classification.

## Features

- Real-time image prediction
- PCA-based feature extraction
- ANN classification model
- Streamlit deployment
- Accuracy graph visualization
- Eigenfaces generation

## Tech Stack

Python, OpenCV, Scikit-learn, Streamlit, NumPy, Matplotlib, Joblib, Pillow

## Project Structure

```
FaceRecognition_PCA_ANN/
├── dataset/faces/          # Face images organized by person
│   ├── Kalyan/
│   ├── Rahul/
│   ├── Suresh/
│   ├── Priya/
│   ├── Ananya/
│   └── Vikram/
├── outputs/
│   ├── graphs/             # Accuracy graph
│   ├── eigenfaces/         # Eigenface visualizations
│   └── predictions/        # Sample predictions
├── streamlit_app.py        # Streamlit web application
├── train_model.py          # Training script
├── model.pkl               # Trained ANN classifier
├── pca.pkl                 # Fitted PCA model
├── labels.pkl              # Label encoder
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for Streamlit Cloud
└── README.md
```

## Results

### Home UI
![Home UI](outputs/predictions/app_working.png)

### Accuracy Graph
![Accuracy Graph](outputs/graphs/accuracy_graph.png)

### Eigenfaces
![Eigenface 0](outputs/eigenfaces/eigenface_0.png)
![Eigenface 1](outputs/eigenfaces/eigenface_1.png)

### Training Output
```
Loading Dataset...
Total Images Loaded: 60
Applying PCA...
Training ANN Model...
Model Accuracy: 100.00%
Model Saved Successfully!
Accuracy Graph Saved!
Eigenfaces Saved!
Training Completed Successfully!
```

## How It Works

1. **Dataset**: 60 face images of 6 persons (10 each) from the Olivetti faces dataset
2. **PCA**: Reduces 10,000-dimensional image space to 39 principal components
3. **ANN**: MLPClassifier with 100 hidden neurons trains on PCA-reduced features
4. **Prediction**: Uploaded image goes through same PCA transform, classified by ANN

## Deployment

Live Demo: [https://appapppy-82gfkkrhjfskrpay5mtyqz.streamlit.app/](https://appapppy-82gfkkrhjfskrpay5mtyqz.streamlit.app/)

## License

MIT
