import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_face = None
        self.eigenvectors = None
        self.eigenvalue_ratios = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean_face = np.mean(X, axis=0)
        delta = X - self.mean_face

        # Use small surrogate covariance: C = delta @ delta.T (n_samples x n_samples)
        # Eigenvectors of original covariance can be recovered via: delta.T @ v_small
        C_small = delta @ delta.T
        eigenvalues_small, eigenvectors_small = np.linalg.eig(C_small)

        eigenvalues = np.real(eigenvalues_small)
        eigenvectors_small = np.real(eigenvectors_small)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors_small = eigenvectors_small[:, idx]

        # Recover original eigenvectors: U = delta.T @ V_small
        eigenvectors = delta.T @ eigenvectors_small

        # Normalize to unit length
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        norms[norms == 0] = 1
        eigenvectors = eigenvectors / norms

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        total = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

        if self.n_components is not None and self.n_components < self.eigenvectors.shape[1]:
            self.eigenvectors = self.eigenvectors[:, :self.n_components]
            self.eigenvalues = self.eigenvalues[:self.n_components]
            self.explained_variance_ratio = self.explained_variance_ratio[:self.n_components]

        print(f"[PCA] Fitted with {self.eigenvectors.shape[1]} components | "
              f"Explained variance: {np.sum(self.explained_variance_ratio):.4f}")
        return self

    def transform(self, X):
        delta = X - self.mean_face
        return delta @ self.eigenvectors

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_proj):
        return X_proj @ self.eigenvectors.T + self.mean_face

    def get_eigenfaces(self, height=64, width=64, max_faces=20):
        ef = []
        n = min(max_faces, self.eigenvectors.shape[1])
        for i in range(n):
            ev = self.eigenvectors[:, i]
            ev_min, ev_max = ev.min(), ev.max()
            if ev_max - ev_min > 0:
                ev_norm = (ev - ev_min) / (ev_max - ev_min)
            else:
                ev_norm = np.zeros_like(ev)
            ef.append(ev_norm.reshape(height, width))
        return ef
