"""
Support Vector Machine (SVM) algorithms.
Includes:
1. LinearSVMScratch: Linear SVM from scratch optimizing hinge loss via sub-gradient descent.
2. KernelSVMWrapper: Unified interface wrapping scikit-learn's SVC for linear, polynomial, RBF, and sigmoid kernels.
"""

from typing import Tuple, List, Optional
import numpy as np
from sklearn.svm import SVC


class LinearSVMScratch:
    """
    Linear Support Vector Machine from scratch.
    Objective: min 0.5 * ||w||^2 + C * sum max(0, 1 - y_i * (w^T * x_i + b))
    Optimized via Sub-Gradient Descent with learning rate decay.
    """
    def __init__(self, C: float = 1.0, learning_rate: float = 0.001, epochs: int = 500):
        self.C = float(C)
        self.lr = float(learning_rate)
        self.epochs = epochs
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        X = np.asarray(X, dtype=np.float64)
        # Convert {0, 1} to {-1, +1}
        y_binary = np.where(np.asarray(y) <= 0, -1.0, 1.0)
        n_samples, n_features = X.shape
        
        # Initialize weights
        rng = np.random.default_rng(42)
        self.w_ = rng.normal(0, 0.01, size=n_features)
        self.b_ = 0.0
        self.loss_history_ = []

        for epoch in range(1, self.epochs + 1):
            # Step decay
            current_lr = self.lr / np.sqrt(epoch)
            
            # Margin condition: y_i * (w^T * x_i + b)
            margins = y_binary * (np.dot(X, self.w_) + self.b_)
            
            # Hinge loss
            hinge_losses = np.maximum(0.0, 1.0 - margins)
            loss = 0.5 * np.sum(self.w_ ** 2) + self.C * np.sum(hinge_losses)
            self.loss_history_.append(loss)
            
            # Sub-gradients
            # Indicator of margin violation: margin < 1
            violating = margins < 1.0
            
            grad_w = self.w_ - self.C * np.dot(violating * y_binary, X)
            grad_b = - self.C * np.sum(violating * y_binary)
            
            # Update
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
    """
    Kernel SVM wrapper supporting Linear, Polynomial, RBF, and Sigmoid kernels.
    Provides uniform API and probability calibration support.
    """
    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", degree: int = 3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=True,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSVMWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
