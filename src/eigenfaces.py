import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from src.utils import ensure_dir

def save_eigenfaces(pca, height=64, width=64, save_dir="outputs/eigenfaces"):
    ensure_dir(save_dir)
    eigenfaces = pca.get_eigenfaces(height, width)
    n = len(eigenfaces)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(eigenfaces[i], cmap='gray')
        axes[i].set_title(f'Eigenface {i+1}', fontsize=9)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.suptitle('Eigenfaces (Principal Components)', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'eigenfaces_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Eigenfaces grid saved to {path}")

    # Save individual eigenfaces
    for i, ef in enumerate(eigenfaces[:10]):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(ef, cmap='gray')
        ax.set_title(f'Eigenface {i+1}')
        ax.axis('off')
        ind_path = os.path.join(save_dir, f'eigenface_{i+1:02d}.png')
        plt.savefig(ind_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Save mean face
    mean_face_img = pca.mean_face.reshape(height, width)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(mean_face_img, cmap='gray')
    ax.set_title('Mean Face')
    ax.axis('off')
    mean_path = os.path.join(save_dir, 'mean_face.png')
    plt.savefig(mean_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Mean face saved to {mean_path}")

def save_explained_variance(pca, save_dir="outputs/graphs"):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    cumsum = np.cumsum(pca.explained_variance_ratio)
    ax.bar(range(1, len(pca.explained_variance_ratio) + 1),
           pca.explained_variance_ratio * 100, alpha=0.6, label='Individual')
    ax.plot(range(1, len(cumsum) + 1), cumsum * 100, 'r-o', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Explained Variance by Principal Components')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    path = os.path.join(save_dir, 'explained_variance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Explained variance graph saved to {path}")
