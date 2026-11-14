# PCA + ANN Face Recognition

<p align="center"><strong>A computer-vision learning project that represents face images with PCA eigenfaces and classifies the resulting features with an artificial neural network.</strong></p>

<p align="center">Python · OpenCV · NumPy · scikit-learn · Streamlit</p>

## Live demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appapppy-82gfkkrhjfskrpay5mtyqz.streamlit.app/)

**[Launch the PCA + ANN face-recognition app →](https://appapppy-82gfkkrhjfskrpay5mtyqz.streamlit.app/)**

## Project overview

This repository contains a dataset preparation pipeline, PCA/ANN training code, saved model artifacts, visual outputs, and a Streamlit interface. The implementation is intended for experimentation with a fixed set of known identities.

## Workflow

```mermaid
flowchart LR
  D[Training face images] --> P[Dataset preparation]
  P --> F[Face preprocessing]
  F --> PCA[PCA eigenface projection]
  PCA --> ANN[ANN classifier]
  ANN --> M[Saved model artifacts]
  M --> S[Streamlit prediction interface]
  S --> R[Predicted class and evaluation visuals]
```

## Features

- Dataset preparation and train/test folders
- PCA feature projection and saved eigenface model
- `MLPClassifier`-based identity classification in the root-level demo artifacts
- Training and evaluation scripts
- Streamlit interface and generated visualizations
- ZIP path validation before extracting the downloaded dataset

## Run locally

Use Python 3.10 or newer. Create an environment and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Launch the checked-in model demo with:

```bash
streamlit run streamlit_app.py
```

The root-level demo artifacts (`model.pkl`, `pca.pkl`, and `labels.pkl`) are loaded as trusted local joblib files. Do not replace them with artifacts from an untrusted source. `train_model.py` expects images under `dataset/faces/<identity>/`; `prepare_dataset.py` creates `dataset/train/` and `dataset/test/` for the separate custom pipeline under `src/`. The root training script and custom pipeline use different model implementations and input dimensions; keep each model paired with its own preprocessing and labels. Scikit-learn is pinned to the version recorded in the checked-in model artifacts to reduce serialization incompatibility.

## Architecture and results

See [ARCHITECTURE.md](ARCHITECTURE.md) for the repository's architecture notes and the [`screenshots/`](screenshots/) directory for saved UI, prediction, and evaluation images. Open the [live Streamlit demo](https://appapppy-82gfkkrhjfskrpay5mtyqz.streamlit.app/). The app may take a moment to wake if the hosting service has been idle.

The checked-in classification report is one historical evaluation run, not a guarantee of accuracy on new people or camera conditions. The interface no longer presents a hard-coded accuracy score. Face recognition is biometric processing: use consented data, protect the dataset and model files, and do not treat this classroom prototype as an access-control system.

## Project structure

- `prepare_dataset.py` — dataset preparation
- `train_model.py` — PCA/ANN training pipeline
- `app.py`, `streamlit_app.py` — interface entry points
- `src/` — project implementation modules
- `dataset/` — included sample face dataset
- `outputs/`, `screenshots/` — generated examples and visual results
- `ARCHITECTURE.md` — existing architecture notes
