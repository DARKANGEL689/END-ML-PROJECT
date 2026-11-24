import pytest
import numpy as np
from sklearn.datasets import load_iris
from src.lda import LDAScratch

def test_lda_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lda = LDAScratch(n_components=2).fit(X, y)
    X_proj = lda.transform(X)
    assert X_proj.shape == (150, 2)
    preds = lda.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.9
