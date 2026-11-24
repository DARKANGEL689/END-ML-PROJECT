"""
Naive Bayes classifiers implemented completely from scratch using NumPy.
Includes Gaussian Naive Bayes (for continuous features) and
Multinomial Naive Bayes (for text token frequencies and TF-IDF) with Laplace smoothing.
Calculations are executed strictly in log-space to prevent numerical underflow.
"""

from typing import Tuple, Optional
import numpy as np


class GaussianNBScratch:
    """
    Gaussian Naive Bayes classifier from scratch.
    Models continuous features as normal distributions: P(x_i | y) ~ N(mu_y, sigma_y^2).
    """
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_priors_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # Class means: shape (n_classes, n_features)
        self.var_: Optional[np.ndarray] = None    # Class variances: shape (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBScratch":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.class_priors_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = X_c.shape[0] / n_samples
            self.theta_[idx, :] = np.mean(X_c, axis=0)
            # Add variance smoothing for numerical stability
            self.var_[idx, :] = np.var(X_c, axis=0) + self.var_smoothing
            
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            prior = np.log(self.class_priors_[idx])
            mean = self.theta_[idx]
            var = self.var_[idx]
            
            # Log Gaussian likelihood: -0.5 * sum(log(2*pi*var) + ((x - mean)^2 / var))
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            
            joint_log_likelihood[:, idx] = prior + log_prob
            
        return joint_log_likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        
        # Log-sum-exp trick for numerical stability
        max_log = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - max_log)
        probs = exp_jll / np.sum(exp_jll, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]


class MultinomialNBScratch:
    """
    Multinomial Naive Bayes classifier from scratch.
    Models discrete word counts / TF-IDF frequencies with Laplace smoothing (alpha).
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None  # shape: (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNBScratch":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            self.class_log_prior_[idx] = np.log(n_c / n_samples)
            
            # Feature counts for class c
            count_c = np.sum(X_c, axis=0) + self.alpha
            total_count = np.sum(count_c)
            
            # Log probability: log( (N_ci + alpha) / (N_c + alpha * d) )
            self.feature_log_prob_[idx, :] = np.log(count_c / total_count)
            
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        # jll = log_prior + X @ log_prob.T
        return self.class_log_prior_ + np.dot(X, self.feature_log_prob_.T)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        max_log = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - max_log)
        return exp_jll / np.sum(exp_jll, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
