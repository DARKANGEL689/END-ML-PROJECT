"""
Distance and similarity measures implemented from scratch using vectorized NumPy.
Includes Euclidean, Manhattan, Minkowski (general p), Chebyshev, Cosine, Jaccard, and Hamming.
"""

from typing import Union, Set, Any
import numpy as np


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Euclidean distance: sqrt(sum((u - v)^2))"""
    diff = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.sum(diff ** 2)))


def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Manhattan (L1) distance: sum(|u - v|)"""
    diff = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    return float(np.sum(np.abs(diff)))


def minkowski_distance(u: np.ndarray, v: np.ndarray, p: float = 3.0) -> float:
    """Minkowski (Lp) distance: (sum(|u - v|^p))^(1/p)"""
    if p <= 0:
        raise ValueError("Minkowski p must be strictly positive.")
    diff = np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))
    return float(np.sum(diff ** p) ** (1.0 / p))


def chebyshev_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Chebyshev (L_infinity) distance: max(|u - v|)"""
    diff = np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))
    return float(np.max(diff))


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    """Cosine similarity: (u . v) / (||u|| * ||v||)"""
    u = np.asarray(u, dtype=np.float64).flatten()
    v = np.asarray(v, dtype=np.float64).flatten()
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < eps or norm_v < eps:
        return 0.0
    return float(dot / (norm_u * norm_v))


def cosine_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    """Cosine distance: 1 - cosine_similarity"""
    return float(1.0 - cosine_similarity(u, v, eps=eps))


def jaccard_similarity_sets(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Jaccard similarity for sets: |A intersect B| / |A union B|"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / union) if union > 0 else 0.0


def jaccard_distance_sets(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Jaccard distance: 1 - Jaccard similarity"""
    return float(1.0 - jaccard_similarity_sets(set_a, set_b))


def hamming_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Hamming distance: proportion of elements that differ."""
    u = np.asarray(u)
    v = np.asarray(v)
    if len(u) != len(v):
        raise ValueError("Arrays must have the same length.")
    return float(np.mean(u != v))


# =====================================================================
# Vectorized Pairwise Distance Computations (No Double Python Loops)
# =====================================================================

def pairwise_euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix between rows of X and Y.
    Using identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x . y
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)
    y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T
    dists_sq = x_sq + y_sq - 2.0 * np.dot(X, Y.T)
    return np.sqrt(np.maximum(dists_sq, 0.0))


def pairwise_manhattan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Manhattan distance using broadcasting."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # Shape: (n_X, 1, n_feat) - (1, n_Y, n_feat)
    return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)


def pairwise_cosine_similarity(X: np.ndarray, Y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Vectorized pairwise cosine similarity: S = (X @ Y.T) / (||X|| @ ||Y||.T)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    
    norm_X = np.where(norm_X < eps, eps, norm_X)
    norm_Y = np.where(norm_Y < eps, eps, norm_Y)
    
    X_norm = X / norm_X
    Y_norm = Y / norm_Y
    
    return np.dot(X_norm, Y_norm.T)


def pairwise_cosine_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Vectorized pairwise cosine distance: 1 - cosine_similarity"""
    return np.clip(1.0 - pairwise_cosine_similarity(X, Y, eps=eps), 0.0, 2.0)
