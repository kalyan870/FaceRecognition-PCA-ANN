import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from src.utils import ensure_dir

class ANNModel:
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=500, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            activation='relu',
            solver='adam',
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.label_map = None
        self.reverse_label_map = None

    def train(self, X_train, y_train, label_map):
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        print(f"[ANN] Training MLPClassifier with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        train_acc = self.model.score(X_train, y_train)
        print(f"[ANN] Training accuracy: {train_acc:.4f}")
        print(f"[ANN] Number of iterations: {self.model.n_iter_}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_with_confidence(self, X, threshold=0.6):
        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)
        max_probas = np.max(probas, axis=1)
        results = []
        for i, (pred, conf) in enumerate(zip(preds, max_probas)):
            if conf >= threshold:
                label = self.reverse_label_map.get(pred, "Unknown")
            else:
                label = "Unknown (Imposter)"
            results.append((label, conf))
        return results

    def score(self, X, y):
        return self.model.score(X, y)

    def save(self, path="outputs/models/ann_model.joblib"):
        ensure_dir(os.path.dirname(path))
        joblib.dump(self, path)
        print(f"[INFO] ANN model saved to {path}")

    @staticmethod
    def load(path="outputs/models/ann_model.joblib"):
        model = joblib.load(path)
        print(f"[INFO] ANN model loaded from {path}")
        return model
