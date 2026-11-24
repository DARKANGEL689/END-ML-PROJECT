import pytest
import numpy as np
from src.svm import LinearSVMScratch, KernelSVMWrapper

def test_linear_svm():
    rng = np.random.default_rng(42)
    X_pos = rng.normal(loc=2.0, scale=0.5, size=(50, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.5, size=(50, 2))
    X = np.vstack((X_pos, X_neg))
    y = np.array([1] * 50 + [0] * 50)
    svm = LinearSVMScratch(C=1.0, learning_rate=0.01, epochs=300).fit(X, y)
    preds = svm.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95

def test_kernel_svm_wrapper():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.0).astype(int)
    svm = KernelSVMWrapper(kernel='rbf', C=1.0).fit(X, y)
    preds = svm.predict(X)
    assert len(preds) == len(y)
