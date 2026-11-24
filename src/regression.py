from typing import Tuple, List, Dict, Any, Optional
import numpy as np

class SimpleLinearRegressionScratch:

    def __init__(self):
        self.slope_: Optional[float] = None
        self.intercept_: Optional[float] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SimpleLinearRegressionScratch':
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        n = len(x)
        if n != len(y):
            raise ValueError('x and y must have the same length.')
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            self.slope_ = 0.0
            self.intercept_ = y_mean
        else:
            self.slope_ = float(numerator / denominator)
            self.intercept_ = float(y_mean - self.slope_ * x_mean)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.slope_ is None or self.intercept_ is None:
            raise ValueError('Model is not fitted yet.')
        x = np.asarray(x, dtype=np.float64).flatten()
        return self.intercept_ + self.slope_ * x

    def r2_score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

class MultipleLinearRegressionScratch:

    def __init__(self, fit_intercept: bool=True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.cost_history_: List[float] = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        intercept_col = np.ones((n_samples, 1), dtype=np.float64)
        return np.hstack((intercept_col, X))

    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> 'MultipleLinearRegressionScratch':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        if self.fit_intercept:
            X_design = self._add_intercept(X)
        else:
            X_design = X
        XTX = np.dot(X_design.T, X_design)
        XTy = np.dot(X_design.T, y)
        try:
            theta = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            theta = np.dot(np.linalg.pinv(XTX), XTy)
        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        return self

    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, learning_rate: float=0.01, epochs: int=1000) -> 'MultipleLinearRegressionScratch':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X_design = self._add_intercept(X)
            weights = np.zeros(n_features + 1, dtype=np.float64)
        else:
            X_design = X
            weights = np.zeros(n_features, dtype=np.float64)
        self.cost_history_ = []
        for epoch in range(epochs):
            predictions = np.dot(X_design, weights)
            errors = predictions - y
            cost = 1.0 / (2.0 * n_samples) * np.sum(errors ** 2)
            self.cost_history_.append(cost)
            gradients = 1.0 / n_samples * np.dot(X_design.T, errors)
            weights -= learning_rate * gradients
        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError('Model is not fitted.')
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.coef_) + self.intercept_

class RidgeRegressionScratch:

    def __init__(self, alpha: float=1.0, fit_intercept: bool=True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionScratch':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape
        if self.fit_intercept:
            x_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - x_mean
            y_centered = y - y_mean
            I = np.eye(n_features)
            XTX_ridge = np.dot(X_centered.T, X_centered) + self.alpha * I
            XTy = np.dot(X_centered.T, y_centered)
            self.coef_ = np.linalg.solve(XTX_ridge, XTy)
            self.intercept_ = float(y_mean - np.dot(x_mean, self.coef_))
        else:
            I = np.eye(n_features)
            XTX_ridge = np.dot(X.T, X) + self.alpha * I
            XTy = np.dot(X.T, y)
            self.coef_ = np.linalg.solve(XTX_ridge, XTy)
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.coef_) + self.intercept_

class LassoRegressionScratch:

    def __init__(self, alpha: float=0.1, max_iter: int=1000, tol: float=0.0001):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressionScratch':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape
        x_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_c = X - x_mean
        y_c = y - y_mean
        z_j = np.sum(X_c ** 2, axis=0)
        z_j = np.where(z_j == 0, 1e-09, z_j)
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        for iteration in range(self.max_iter):
            max_change = 0.0
            for j in range(n_features):
                old_coef_j = self.coef_[j]
                residual = y_c - (np.dot(X_c, self.coef_) - X_c[:, j] * old_coef_j)
                rho_j = np.dot(X_c[:, j], residual)
                new_coef_j = self._soft_threshold(rho_j, self.alpha * n_samples) / z_j[j]
                self.coef_[j] = new_coef_j
                change = abs(new_coef_j - old_coef_j)
                if change > max_change:
                    max_change = change
            if max_change < self.tol:
                break
        self.intercept_ = float(y_mean - np.dot(x_mean, self.coef_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.coef_) + self.intercept_

class LogisticRegressionScratch:

    def __init__(self, learning_rate: float=0.05, epochs: int=1000, fit_intercept: bool=True):
        self.lr = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.loss_history_: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X_design = np.hstack((np.ones((n_samples, 1)), X))
            weights = np.zeros(n_features + 1, dtype=np.float64)
        else:
            X_design = X
            weights = np.zeros(n_features, dtype=np.float64)
        self.loss_history_ = []
        eps = 1e-15
        for epoch in range(self.epochs):
            z = np.dot(X_design, weights)
            p = self._sigmoid(z)
            loss = -(1.0 / n_samples) * np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
            self.loss_history_.append(loss)
            gradients = 1.0 / n_samples * np.dot(X_design.T, p - y)
            weights -= self.lr * gradients
        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.coef_) + self.intercept_
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X: np.ndarray, threshold: float=0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

def mean_absolute_error_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
