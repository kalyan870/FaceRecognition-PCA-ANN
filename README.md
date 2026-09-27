# PCA + ANN Face Recognition

<p align="center"><strong>A computer-vision learning project that represents face images with PCA eigenfaces and classifies the resulting features with an artificial neural network.</strong></p>

<p align="center">Python · OpenCV · NumPy · scikit-learn · Streamlit</p>

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
- ANN-based identity classification
- Training and evaluation scripts
- Streamlit interface and generated visualizations

## Run locally

Use Python 3.10 or newer. Create an environment and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Review `train_model.py` and `prepare_dataset.py` for the expected dataset layout and training options before regenerating model artifacts. Launch the interface with:

```bash
streamlit run streamlit_app.py
```

## Architecture and results

See [ARCHITECTURE.md](ARCHITECTURE.md) for the repository's architecture notes and the [`screenshots/`](screenshots/) directory for saved UI, prediction, and evaluation images. The [Streamlit demo](https://appapppy-82gfkkrhjfskrpay5mtyqz.streamlit.app/) is linked from the existing project page; availability can change.

The README's historical training output is a recorded run, not a guarantee of accuracy on new people or camera conditions. Face recognition is biometric processing: use consented data, protect the dataset and model files, and do not treat this classroom prototype as an access-control system.

## Project structure

- `prepare_dataset.py` — dataset preparation
- `train_model.py` — PCA/ANN training pipeline
- `app.py`, `streamlit_app.py` — interface entry points
- `src/` — project implementation modules
- `dataset/` — included sample face dataset
- `outputs/`, `screenshots/` — generated examples and visual results
- `ARCHITECTURE.md` — existing architecture notes
