import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

IMG_SIZE = (64, 64)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img.flatten())
            filenames.append(fname)
            if label is not None:
                labels.append(label)
    return images, labels, filenames

def plot_accuracy_vs_k(k_values, accuracies, save_path="outputs/graphs/accuracy_vs_k.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Number of Principal Components (k)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Accuracy vs Number of Eigenfaces (k)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    for i, (k, acc) in enumerate(zip(k_values, accuracies)):
        plt.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Accuracy graph saved to {save_path}")
