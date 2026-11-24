"""Unit tests for K-Means clustering and silhouette scoring."""
import pytest
import numpy as np
from sklearn.cluster import KMeans
from src.kmeans import KMeansScratch, silhouette_score_scratch

def test_kmeans_clustering():
    rng = np.random.default_rng(42)
    # 3 distinct clusters
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    c3 = rng.normal(loc=[0, 5], scale=0.3, size=(30, 2))
    X = np.vstack((c1, c2, c3))
    
    km = KMeansScratch(n_clusters=3, n_init=3, random_state=42).fit(X)
    assert km.cluster_centers_.shape == (3, 2)
    assert km.inertia_ < 50.0

def test_silhouette_score():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0, 0], scale=0.2, size=(30, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.2, size=(30, 2))
    X = np.vstack((c1, c2))
    labels = np.array([0] * 30 + [1] * 30)
    
    # Tight distinct clusters should have high silhouette score (> 0.7)
    score = silhouette_score_scratch(X, labels)
    assert score > 0.70
