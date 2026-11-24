from typing import Tuple, List, Optional
import numpy as np

class LDAScratch:

    def __init__(self, n_components: Optional[int]=None, reg_param: float=0.0001):
        self.n_components = n_components
        self.reg_param = reg_param
        self.scalings_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.overall_mean_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDAScratch':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.overall_mean_ = np.mean(X, axis=0)
        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            N_c = X_c.shape[0]
            self.priors_[idx] = N_c / n_samples
            m_c = np.mean(X_c, axis=0)
            self.means_[idx] = m_c
            X_c_centered = X_c - m_c
            Sw += np.dot(X_c_centered.T, X_c_centered)
            mean_diff = (m_c - self.overall_mean_).reshape(-1, 1)
            Sb += N_c * np.dot(mean_diff, mean_diff.T)
        Sw += self.reg_param * np.eye(n_features)
        Sw_inv = np.linalg.pinv(Sw)
        A = np.dot(Sw_inv, Sb)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        max_possible_components = min(n_classes - 1, n_features)
        k = self.n_components if self.n_components is not None else max_possible_components
        k = min(k, max_possible_components)
        k = max(1, k)
        self.scalings_ = eigenvectors[:, :k]
        pos_eigenvalues = np.maximum(eigenvalues[:max_possible_components], 0.0)
        tot_var = np.sum(pos_eigenvalues)
        if tot_var > 0:
            self.explained_variance_ratio_ = (pos_eigenvalues / tot_var)[:k]
        else:
            self.explained_variance_ratio_ = np.zeros(k)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scalings_ is None:
            raise ValueError('LDA model is not fitted.')
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.scalings_)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_proj = self.transform(X)
        means_proj = self.transform(self.means_)
        preds = []
        for x in X_proj:
            scores = []
            for idx in range(len(self.classes_)):
                dist = np.sum((x - means_proj[idx]) ** 2)
                score = dist - 2.0 * np.log(self.priors_[idx] + 1e-09)
                scores.append(score)
            preds.append(self.classes_[np.argmin(scores)])
        return np.array(preds)
