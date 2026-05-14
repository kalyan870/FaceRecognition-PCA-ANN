# Face Recognition System using PCA (Eigenfaces) + ANN

## Overview

A complete face recognition system that combines **Principal Component Analysis (PCA)** for dimensionality reduction and feature extraction (Eigenfaces) with an **Artificial Neural Network (ANN)** for classification.

## Methodology

### PCA (Principal Component Analysis)

PCA transforms high-dimensional face images into a lower-dimensional subspace by finding the directions (principal components) that maximize variance. The principal components, when visualized, resemble face-like patterns called **Eigenfaces**.

**Steps:**
1. Flatten each face image (m×n) into a vector
2. Build a data matrix (p images × mn pixels)
3. Compute the **mean face**
4. Normalize by subtracting the mean face
5. Compute the surrogate covariance matrix: **C = ΔᵀΔ**
6. Perform eigenvalue decomposition using `np.linalg.eig()`
7. Select top-k eigenvectors (Eigenfaces)
8. Project face data onto the eigenface subspace

### ANN (Artificial Neural Network)

A multi-layer perceptron (MLP) classifier from scikit-learn that learns to map PCA-reduced face features to person identities.

**Architecture:**
- **Input Layer:** k PCA features
- **Hidden Layer 1:** 128 neurons (ReLU)
- **Hidden Layer 2:** 64 neurons (ReLU)
- **Output Layer:** number of persons (Softmax)

## Project Structure

```
FaceRecognition_PCA_ANN/
├── dataset/
│   ├── train/          # Training images (60%)
│   └── test/           # Testing images (40%)
├── src/
│   ├── preprocess.py   # Image loading & preprocessing
│   ├── pca.py          # PCA implementation (Eigenfaces)
│   ├── eigenfaces.py   # Eigenface visualization
│   ├── ann_model.py    # ANN (MLPClassifier) model
│   ├── train.py        # Training pipeline
│   ├── test.py         # Testing & imposter detection
│   └── utils.py        # Utility functions
├── outputs/
│   ├── graphs/         # Accuracy & variance graphs
│   ├── eigenfaces/     # Eigenface visualizations
│   ├── predictions/    # Prediction results
│   └── models/         # Saved PCA + ANN models
├── app.py              # Main entry point
├── requirements.txt
└── README.md
```

## Results

### Accuracy vs Number of Eigenfaces (k)

The system is evaluated across different k values (5, 10, 20, 30, 40):

| k   | Accuracy |
|-----|----------|
| 5   | —%       |
| 10  | —%       |
| 20  | —%       |
| 30  | —%       |
| 40  | —%       |

### Imposter Detection

Faces not present in the training set are classified as **"Unknown (Imposter)"** when the maximum Softmax probability falls below a confidence threshold (0.6).

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Organize Dataset

Place training images in `dataset/train/<person_name>/` and testing images in `dataset/test/<person_name>/`.

### Run Full Pipeline

```bash
python app.py --mode full
```

### Train Only

```bash
python app.py --mode train
```

### Test Only

```bash
python app.py --mode test
```

### Test Single Image

```bash
python app.py --mode test --image path/to/image.jpg
```

### Custom k Values

```bash
python app.py --mode full --k_values 5 10 20 30 40
```

## Tech Stack

- **Python** — Core programming language
- **OpenCV** — Image reading and preprocessing
- **NumPy** — Linear algebra and matrix operations
- **Scikit-learn** — ANN (MLPClassifier)
- **Matplotlib** — Visualization and plotting
- **Jupyter Notebook** — Interactive exploration
