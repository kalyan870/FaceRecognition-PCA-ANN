import os
import numpy as np
import joblib
from src.preprocess import load_dataset
from src.pca import PCA
from src.eigenfaces import save_eigenfaces, save_explained_variance
from src.ann_model import ANNModel
from src.utils import ensure_dir, plot_accuracy_vs_k

def train_pipeline(train_dir, test_dir, k_values=[10, 20, 30, 40, 50], save_models=True):
    print("=" * 60)
    print("PHASE 1: Loading Dataset")
    print("=" * 60)
    X_train, X_test, y_train, y_test, label_map, _, _ = load_dataset(train_dir, test_dir)

    accuracies = []
    models = {}

    for k in k_values:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2-3: PCA with k={k}")
        print(f"{'=' * 60}")
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        print(f"[PCA] X_train_pca shape: {X_train_pca.shape}")
        print(f"[PCA] X_test_pca shape: {X_test_pca.shape}")

        if k == k_values[0]:
            save_eigenfaces(pca)
            save_explained_variance(pca)

        print(f"\n{'=' * 60}")
        print(f"PHASE 4: Training ANN with k={k}")
        print(f"{'=' * 60}")
        ann = ANNModel()
        ann.train(X_train_pca, y_train, label_map)

        test_acc = ann.score(X_test_pca, y_test) * 100
        accuracies.append(test_acc)
        print(f"[RESULT] k={k}: Test Accuracy = {test_acc:.2f}%")

        if save_models:
            model_path = f"outputs/models/pca_k{k}.joblib"
            pca_dir = os.path.dirname(model_path)
            ensure_dir(pca_dir)

            pca_path = model_path.replace('.joblib', '_pca.joblib')
            ann_path = model_path.replace('.joblib', '_ann.joblib')
            joblib.dump(pca, pca_path)
            ann.save(ann_path)
            models[k] = {'pca': pca_path, 'ann': ann_path}

    print(f"\n{'=' * 60}")
    print("RESULTS: Accuracy vs k")
    print(f"{'=' * 60}")
    for k, acc in zip(k_values, accuracies):
        print(f"  k={k:2d}: {acc:.2f}%")

    plot_accuracy_vs_k(k_values, accuracies)
    best_k = k_values[np.argmax(accuracies)]
    print(f"\n[INFO] Best k = {best_k} with accuracy = {max(accuracies):.2f}%")

    # Train final model with best k
    print(f"\n{'=' * 60}")
    print(f"PHASE: Training Final Model with k={best_k}")
    print(f"{'=' * 60}")
    best_pca = PCA(n_components=best_k)
    X_train_pca = best_pca.fit_transform(X_train)
    best_ann = ANNModel()
    best_ann.train(X_train_pca, y_train, label_map)

    if save_models:
        joblib.dump(best_pca, "outputs/models/final_pca.joblib")
        best_ann.save("outputs/models/final_ann.joblib")

    print(f"\n[INFO] Training complete! Best accuracy: {max(accuracies):.2f}% with k={best_k}")
    return best_pca, best_ann, label_map, k_values, accuracies
