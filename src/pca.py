import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_face = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, X):
        self.mean_face = np.mean(X, axis=0)
        delta = X - self.mean_face

        # Surrogate covariance: C = delta.T @ delta (p x p matrix)
        C = delta.T @ delta
        eigenvalues, eigenvectors = np.linalg.eig(C)

        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvectors to unit length
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        total = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

        if self.n_components is not None:
            self.eigenvectors = self.eigenvectors[:, :self.n_components]
            self.eigenvalues = self.eigenvalues[:self.n_components]
            self.explained_variance_ratio = self.explained_variance_ratio[:self.n_components]

        print(f"[PCA] Fitted with {self.eigenvectors.shape[1]} components")
        print(f"[PCA] Explained variance: {np.sum(self.explained_variance_ratio):.4f}")
        return self

    def transform(self, X):
        delta = X - self.mean_face
        return delta @ self.eigenvectors

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_proj):
        return X_proj @ self.eigenvectors.T + self.mean_face

    def get_eigenfaces(self, height=64, width=64):
        ef = []
        for i in range(min(20, self.eigenvectors.shape[1])):
            ev = self.eigenvectors[:, i]
            ev_min, ev_max = ev.min(), ev.max()
            if ev_max - ev_min > 0:
                ev_norm = (ev - ev_min) / (ev_max - ev_min)
            else:
                ev_norm = np.zeros_like(ev)
            ef.append(ev_norm.reshape(height, width))
        return ef
