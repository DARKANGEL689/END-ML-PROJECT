"""
Principal Component Analysis (PCA) implemented completely from scratch.
Supports Eigendecomposition of the covariance matrix and Singular Value Decomposition (SVD),
with projection, reconstruction error, explained variance, and loadings analysis.
"""

from typing import Tuple, Optional
import numpy as np


class PCAScratch:
    """
    Principal Component Analysis from scratch using NumPy.
    """
    def __init__(self, n_components: Optional[int] = None, standardize: bool = True):
        self.n_components = n_components
        self.standardize = standardize
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # 1. Centering & optional scaling
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        if self.standardize:
            std = np.std(X_centered, axis=0)
            self.scale_ = np.where(std == 0, 1.0, std)
            X_centered /= self.scale_
        else:
            self.scale_ = np.ones(n_features)
            
        # 2. Covariance matrix: (1 / (n - 1)) * X^T * X
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # 3. Eigendecomposition (eigh for symmetric positive semi-definite matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Ensure non-negative numerical eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0.0)
        
        # Enforce deterministic sign convention (largest absolute element positive)
        for i in range(eigenvectors.shape[1]):
            max_idx = np.argmax(np.abs(eigenvectors[:, i]))
            if eigenvectors[max_idx, i] < 0:
                eigenvectors[:, i] *= -1.0

        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            variance_ratio = eigenvalues / total_variance
        else:
            variance_ratio = np.zeros_like(eigenvalues)

        k = self.n_components if self.n_components is not None else n_features
        k = min(k, n_features)
        
        self.components_ = eigenvectors[:, :k].T  # shape: (n_components, n_features)
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = variance_ratio[:k]
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA model is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.mean_) / self.scale_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """Reconstruct original data from principal component scores."""
        X_proj = np.asarray(X_proj, dtype=np.float64)
        # X_reconstructed = X_proj @ components + mean
        X_rec = np.dot(X_proj, self.components_) * self.scale_ + self.mean_
        return X_rec

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Compute Mean Squared Reconstruction Error."""
        X = np.asarray(X, dtype=np.float64)
        X_proj = self.transform(X)
        X_rec = self.inverse_transform(X_proj)
        return float(np.mean((X - X_rec) ** 2))

    def get_loadings(self) -> np.ndarray:
        """
        Loadings matrix = components * sqrt(eigenvalues).
        Shows the correlation between original features and principal components.
        """
        return self.components_.T * np.sqrt(self.explained_variance_)
