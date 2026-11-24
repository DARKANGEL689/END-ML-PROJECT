"""Unit tests for PCA from scratch."""
import pytest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from src.pca import PCAScratch

def test_pca_vs_sklearn_iris():
    iris = load_iris()
    X = iris.data
    
    pca_scratch = PCAScratch(n_components=2, standardize=True).fit(X)
    
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)
    pca_sk = PCA(n_components=2).fit(X_std)
    
    # Check explained variance ratios match closely
    assert np.allclose(pca_scratch.explained_variance_ratio_, pca_sk.explained_variance_ratio_, atol=1e-3)
    
    # Check transformed coordinates match up to sign flip
    proj_scratch = pca_scratch.transform(X)
    proj_sk = pca_sk.transform(X_std)
    
    for c in range(2):
        correlation = np.abs(np.corrcoef(proj_scratch[:, c], proj_sk[:, c])[0, 1])
        assert correlation > 0.999
        
    # Check reconstruction error is low
    err = pca_scratch.reconstruction_error(X)
    assert err < 0.5
