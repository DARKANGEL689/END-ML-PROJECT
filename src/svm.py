from typing import Tuple, List, Optional
import numpy as np
from sklearn.svm import SVC

class LinearSVMScratch:

    def __init__(self, C: float=1.0, learning_rate: float=0.001, epochs: int=500):
        self.C = float(C)
        self.lr = float(learning_rate)
        self.epochs = epochs
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVMScratch':
        X = np.asarray(X, dtype=np.float64)
        y_binary = np.where(np.asarray(y) <= 0, -1.0, 1.0)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(42)
        self.w_ = rng.normal(0, 0.01, size=n_features)
        self.b_ = 0.0
        self.loss_history_ = []
        for epoch in range(1, self.epochs + 1):
            current_lr = self.lr / np.sqrt(epoch)
            margins = y_binary * (np.dot(X, self.w_) + self.b_)
            hinge_losses = np.maximum(0.0, 1.0 - margins)
            loss = 0.5 * np.sum(self.w_ ** 2) + self.C * np.sum(hinge_losses)
            self.loss_history_.append(loss)
            violating = margins < 1.0
            grad_w = self.w_ - self.C * np.dot(violating * y_binary, X)
            grad_b = -self.C * np.sum(violating * y_binary)
            self.w_ -= current_lr * grad_w
            self.b_ -= current_lr * grad_b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return np.where(df >= 0.0, 1, 0)

class KernelSVMWrapper:

    def __init__(self, kernel: str='rbf', C: float=1.0, gamma: str='scale', degree: int=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, probability=True, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelSVMWrapper':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
