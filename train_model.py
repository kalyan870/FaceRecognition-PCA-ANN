import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "dataset/faces"
OUTPUT_GRAPH_PATH = "outputs/graphs/accuracy_vs_k.png"
OUTPUT_EIGENFACES_DIR = "outputs/eigenfaces"
OUTPUT_PREDICTIONS_DIR = "outputs/predictions"
MODEL_PATH = "model.pkl"
PCA_PATH = "pca.pkl"
LABELS_PATH = "labels.pkl"

os.makedirs(OUTPUT_GRAPH_PATH.rsplit('/', 1)[0], exist_ok=True)
os.makedirs(OUTPUT_EIGENFACES_DIR, exist_ok=True)
os.makedirs(OUTPUT_PREDICTIONS_DIR, exist_ok=True)

def load_images(dataset_path):
    images = []
    labels = []
    label_names = []
    print("Loading dataset...")
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_names.append(person_name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (100, 100)).flatten()
                images.append(img_resized)
                labels.append(len(label_names) - 1)
    print(f"Loaded {len(images)} images from {len(label_names)} subjects.")
    return np.array(images), np.array(labels), label_names

def plot_eigenfaces(pca, n=10):
    os.makedirs(OUTPUT_EIGENFACES_DIR, exist_ok=True)
    n = min(n, pca.components_.shape[0])
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(n):
        eigenface = pca.components_[i].reshape(100, 100)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f"Eigenface {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    eigenface_path = os.path.join(OUTPUT_EIGENFACES_DIR, "eigenfaces_grid.png")
    plt.savefig(eigenface_path)
    plt.close()
    print(f"Eigenfaces saved to {eigenface_path}")

def main():
    X, y, label_names = load_images(DATASET_PATH)
    if X.size == 0:
        print("No images found! Check your dataset path.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    k_values = [10, 20, 30, 40, 50]
    accuracies = []
    best_accuracy = 0
    best_k = None
    best_pca = None
    best_X_train_pca = None
    best_X_test_pca = None

    print("Testing different k values...")
    for k in k_values:
        if k > X_train.shape[0]:
            print(f"Skipping k={k} (greater than number of training samples)")
            continue
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        classifier.fit(X_train_pca, y_train)
        y_pred = classifier.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"k={k}: Accuracy = {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_pca = pca
            best_X_train_pca = X_train_pca
            best_X_test_pca = X_test_pca
            joblib.dump(classifier, MODEL_PATH)
            joblib.dump(pca, PCA_PATH)
            joblib.dump((label_names, best_X_train_pca, best_X_test_pca, y_train, y_test), LABELS_PATH)

    if len(accuracies) == 0:
        print("No k value could be tested. Exiting.")
        return

    print(f"\nBest k = {best_k} with Accuracy = {best_accuracy:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values[:len(accuracies)], accuracies, marker='o', linestyle='-', color='b')
    plt.title('PCA Components (k) vs Accuracy')
    plt.xlabel('Number of PCA Components (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(OUTPUT_GRAPH_PATH)
    plt.close()
    print(f"Accuracy graph saved to {OUTPUT_GRAPH_PATH}")

    classifier = joblib.load(MODEL_PATH)
    y_pred = classifier.predict(best_X_test_pca) if best_X_test_pca is not None else []
    if len(y_pred) > 0:
        report = classification_report(y_test, y_pred, target_names=label_names)
        report_path = os.path.join(OUTPUT_PREDICTIONS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")

    if best_pca is not None:
        plot_eigenfaces(best_pca)

    if best_X_test_pca is not None and len(best_X_test_pca) > 0:
        sample_idx = 0
        sample_pca = best_X_test_pca[sample_idx:sample_idx+1]
        pred_label = classifier.predict(sample_pca)[0]
        true_label = y_test[sample_idx]
        img_sample = X_test[sample_idx].reshape(100, 100)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_sample, cmap='gray')
        plt.title(f"True: {label_names[true_label]}\nPred: {label_names[pred_label]}")
        plt.axis('off')
        sample_path = os.path.join(OUTPUT_PREDICTIONS_DIR, "sample_prediction.png")
        plt.savefig(sample_path)
        plt.close()
        print(f"Sample prediction saved to {sample_path}")

    print("\nTraining complete! Model saved.")

if __name__ == "__main__":
    main()
