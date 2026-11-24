from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

class KMeansScratch:

    def __init__(self, n_clusters: int=5, init: str='k-means++', n_init: int=5, max_iter: int=300, tol: float=0.0001, random_state: int=42):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float('inf')

    def _init_kmeans_plus_plus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=np.float64)
        first_idx = rng.integers(0, n_samples)
        centers[0] = X[first_idx]
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for c_idx in range(1, self.n_clusters):
            total_dist_sq = np.sum(closest_dist_sq)
            if total_dist_sq > 0:
                probs = closest_dist_sq / total_dist_sq
            else:
                probs = np.ones(n_samples) / n_samples
            next_idx = rng.choice(n_samples, p=probs)
            centers[c_idx] = X[next_idx]
            new_dist_sq = np.sum((X - centers[c_idx]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
        return centers

    def _single_fit(self, X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
        n_samples, n_features = X.shape
        rng = np.random.default_rng(seed)
        if self.init == 'k-means++':
            centers = self._init_kmeans_plus_plus(X, rng)
        else:
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            centers = X[indices].copy()
        labels = np.zeros(n_samples, dtype=int)
        for iteration in range(self.max_iter):
            x_sq = np.sum(X ** 2, axis=1, keepdims=True)
            c_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
            dist_sq = x_sq + c_sq - 2.0 * np.dot(X, centers.T)
            new_labels = np.argmin(dist_sq, axis=1)
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                cluster_mask = new_labels == k
                if np.any(cluster_mask):
                    new_centers[k] = np.mean(X[cluster_mask], axis=0)
                else:
                    farthest_idx = np.argmax(np.min(dist_sq, axis=1))
                    new_centers[k] = X[farthest_idx]
            center_shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            labels = new_labels
            if center_shift < self.tol:
                break
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[k]) ** 2)
        return (centers, labels, float(inertia))

    def fit(self, X: np.ndarray) -> 'KMeansScratch':
        X = np.asarray(X, dtype=np.float64)
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        for i in range(self.n_init):
            seed = self.random_state + i
            centers, labels, inertia = self._single_fit(X, seed)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError('KMeans model is not fitted.')
        X = np.asarray(X, dtype=np.float64)
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        c_sq = np.sum(self.cluster_centers_ ** 2, axis=1, keepdims=True).T
        dist_sq = x_sq + c_sq - 2.0 * np.dot(X, self.cluster_centers_.T)
        return np.argmin(dist_sq, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

def silhouette_score_scratch(X: np.ndarray, labels: np.ndarray, sample_size: int=1000) -> float:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(X):
        return 0.0
    n_samples = len(X)
    if n_samples > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_samples, size=sample_size, replace=False)
        X = X[indices]
        labels = labels[indices]
        n_samples = sample_size
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(x_sq + x_sq.T - 2.0 * np.dot(X, X.T), 0.0))
    silhouettes = []
    for i in range(n_samples):
        current_cluster = labels[i]
        same_cluster_mask = labels == current_cluster
        same_count = np.sum(same_cluster_mask)
        if same_count > 1:
            a_i = np.sum(dists[i, same_cluster_mask]) / (same_count - 1)
        else:
            a_i = 0.0
        b_i = float('inf')
        for other_cluster in unique_labels:
            if other_cluster == current_cluster:
                continue
            other_cluster_mask = labels == other_cluster
            if np.any(other_cluster_mask):
                mean_dist = np.mean(dists[i, other_cluster_mask])
                if mean_dist < b_i:
                    b_i = mean_dist
        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        silhouettes.append(s_i)
    return float(np.mean(silhouettes))
