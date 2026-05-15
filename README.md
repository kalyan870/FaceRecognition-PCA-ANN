# PCA + ANN Face Recognition

Face recognition system using Principal Component Analysis (PCA) for dimensionality reduction and Artificial Neural Network (ANN) for classification.

## Project Structure

```
FaceRecognition_PCA_ANN/
├── dataset/faces/          # Face images organized in subfolders per person
├── outputs/
│   ├── eigenfaces/         # Generated eigenface visualizations
│   ├── graphs/             # Accuracy vs K components plot
│   └── predictions/        # Sample predictions and classification report
├── app.py                  # Streamlit web application
├── train_model.py          # Training script
├── model.pkl               # Trained ANN classifier
├── pca.pkl                 # Fitted PCA model
├── labels.pkl              # Label encoder and test data
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place face images in `dataset/faces/` — one subfolder per person:
   ```
   dataset/faces/
   ├── person1/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── person2/
       ├── img1.jpg
       └── img2.jpg
   ```

3. Train the model:
   ```
   python train_model.py
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## How It Works

- **PCA** reduces face image dimensions from 10,000 (100x100) to k components
- **ANN (MLPClassifier)** classifies the reduced features into person labels
- Tests k values [10, 20, 30, 40, 50] and selects the best accuracy
