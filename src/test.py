import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
from src.utils import IMG_SIZE, ensure_dir
from src.preprocess import load_imposter_images

def preprocess_single_image(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, img_size)
    return img.flatten().astype(np.float64)

def test_single_image(image_path, pca, ann, label_map, save_dir="outputs/predictions"):
    ensure_dir(save_dir)
    img_vec = preprocess_single_image(image_path)
    img_pca = pca.transform(img_vec.reshape(1, -1))
    results = ann.predict_with_confidence(img_pca, threshold=0.6)
    label, confidence = results[0]

    fname = os.path.basename(image_path)
    print(f"[TEST] {fname} -> Predicted: {label} (confidence: {confidence:.4f})")

    # Visualize
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ax[0].imshow(original_rgb)
    ax[0].set_title(f'Input: {fname}')
    ax[0].axis('off')

    ax[1].text(0.5, 0.5, f'Prediction: {label}\nConfidence: {confidence:.2%}',
               ha='center', va='center', fontsize=14, transform=ax[1].transAxes)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].axis('off')
    ax[1].set_title('Result')

    save_path = os.path.join(save_dir, f'pred_{fname}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Prediction visualization saved to {save_path}")
    return label, confidence

def test_folder(folder_path, pca, ann, label_map, save_dir="outputs/predictions"):
    ensure_dir(save_dir)
    results = {}
    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.ppm']:
            continue
        try:
            label, conf = test_single_image(path, pca, ann, label_map, save_dir)
            results[fname] = (label, conf)
        except Exception as e:
            print(f"[ERROR] Processing {fname}: {e}")
    return results

def test_imposters(imposter_dir, pca, ann, save_dir="outputs/predictions"):
    ensure_dir(save_dir)
    X_imp, imp_files = load_imposter_images(imposter_dir)
    if not isinstance(X_imp, np.ndarray) or X_imp.size == 0 or X_imp.shape[0] == 0:
        print("[INFO] No imposter images found. Skipping imposter detection.")
        return {}

    X_imp_pca = pca.transform(X_imp)
    results = ann.predict_with_confidence(X_imp_pca, threshold=0.6)

    imposter_results = {}
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, (fname, (label, conf)) in enumerate(zip(imp_files, results)):
        imposter_results[fname] = (label, conf)
        print(f"[IMPOSTER] {fname} -> {label} (confidence: {conf:.4f})")
        if i < 10:
            path = os.path.join(imposter_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f'{label}\n({conf:.2f})', fontsize=8)
            axes[i].axis('off')

    if len(imp_files) > 0:
        plt.suptitle('Imposter Detection Results', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'imposter_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Imposter results saved to {save_path}")

    return imposter_results
