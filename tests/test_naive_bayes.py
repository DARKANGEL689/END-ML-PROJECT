"""Unit tests for Naive Bayes classifiers."""
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from src.naive_bayes import GaussianNBScratch, MultinomialNBScratch

def test_gaussian_nb_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    gnb_scratch = GaussianNBScratch().fit(X, y)
    gnb_sk = GaussianNB().fit(X, y)
    
    preds_scratch = gnb_scratch.predict(X)
    preds_sk = gnb_sk.predict(X)
    
    # Should agree with sklearn on almost all points
    assert np.mean(preds_scratch == preds_sk) >= 0.95

def test_multinomial_nb():
    rng = np.random.default_rng(42)
    # Discrete word counts
    X = rng.integers(0, 5, size=(100, 10))
    y = (X[:, 0] + X[:, 1] > 4).astype(int)
    
    mnb_scratch = MultinomialNBScratch(alpha=1.0).fit(X, y)
    mnb_sk = MultinomialNB(alpha=1.0).fit(X, y)
    
    preds_scratch = mnb_scratch.predict(X)
    preds_sk = mnb_sk.predict(X)
    
    assert np.mean(preds_scratch == preds_sk) >= 0.95
