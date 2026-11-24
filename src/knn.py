from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.distances import pairwise_euclidean, pairwise_manhattan, pairwise_cosine_distance, pairwise_cosine_similarity

class KNNClassifierScratch:

    def __init__(self, n_neighbors: int=5, metric: str='euclidean', weights: str='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric.lower()
        self.weights = weights.lower()
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifierScratch':
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _compute_distances(self, X_test: np.ndarray) -> np.ndarray:
        if self.metric == 'euclidean':
            return pairwise_euclidean(X_test, self.X_train_)
        elif self.metric == 'manhattan':
            return pairwise_manhattan(X_test, self.X_train_)
        elif self.metric == 'cosine':
            return pairwise_cosine_distance(X_test, self.X_train_)
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        distances = self._compute_distances(X)
        n_samples = X.shape[0]
        y_pred = []
        knn_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        for i in range(n_samples):
            neighbor_indices = knn_indices[i]
            neighbor_labels = self.y_train_[neighbor_indices]
            neighbor_dists = distances[i, neighbor_indices]
            if self.weights == 'uniform':
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                y_pred.append(unique_labels[np.argmax(counts)])
            elif self.weights == 'distance':
                weights = 1.0 / (neighbor_dists + 1e-06)
                class_weights = {}
                for lbl, w in zip(neighbor_labels, weights):
                    class_weights[lbl] = class_weights.get(lbl, 0.0) + w
                best_class = max(class_weights, key=class_weights.get)
                y_pred.append(best_class)
            else:
                raise ValueError(f'Unsupported weights: {self.weights}')
        return np.array(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        distances = self._compute_distances(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        probas = np.zeros((n_samples, n_classes))
        knn_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        for i in range(n_samples):
            neighbor_indices = knn_indices[i]
            neighbor_labels = self.y_train_[neighbor_indices]
            neighbor_dists = distances[i, neighbor_indices]
            if self.weights == 'uniform':
                for lbl in neighbor_labels:
                    probas[i, class_to_idx[lbl]] += 1.0
                probas[i] /= self.n_neighbors
            else:
                weights = 1.0 / (neighbor_dists + 1e-06)
                total_w = np.sum(weights)
                for lbl, w in zip(neighbor_labels, weights):
                    probas[i, class_to_idx[lbl]] += w
                probas[i] /= total_w
        return probas

class ContentBasedKNNRecommender:

    def __init__(self, top_k: int=10):
        self.top_k = top_k
        self.item_ids_: List[str] = []
        self.item_vectors_: Optional[np.ndarray] = None
        self.item_to_idx_: Dict[str, int] = {}

    def fit(self, item_ids: List[str], item_tfidf_vectors: np.ndarray) -> 'ContentBasedKNNRecommender':
        self.item_ids_ = list(item_ids)
        self.item_vectors_ = np.asarray(item_tfidf_vectors, dtype=np.float64)
        self.item_to_idx_ = {item_id: idx for idx, item_id in enumerate(self.item_ids_)}
        return self

    def recommend_for_history(self, history_item_ids: List[str], candidate_item_ids: Optional[List[str]]=None, k: Optional[int]=None) -> List[Tuple[str, float]]:
        if k is None:
            k = self.top_k
        valid_history = [self.item_to_idx_[item] for item in history_item_ids if item in self.item_to_idx_]
        if not valid_history:
            return []
        user_vector = np.mean(self.item_vectors_[valid_history], axis=0, keepdims=True)
        if candidate_item_ids is None:
            candidate_indices = [idx for idx in range(len(self.item_ids_)) if idx not in valid_history]
        else:
            candidate_indices = [self.item_to_idx_[item] for item in candidate_item_ids if item in self.item_to_idx_ and item not in history_item_ids]
        if not candidate_indices:
            return []
        candidate_vectors = self.item_vectors_[candidate_indices]
        sims = pairwise_cosine_similarity(user_vector, candidate_vectors).flatten()
        top_k_ranks = np.argsort(sims)[::-1][:k]
        recommendations = [(self.item_ids_[candidate_indices[idx]], float(sims[idx])) for idx in top_k_ranks]
        return recommendations

class ItemBasedCFKNN:

    def __init__(self, n_neighbors: int=15):
        self.n_neighbors = n_neighbors
        self.sim_matrix_: Optional[np.ndarray] = None
        self.item_list_: List[str] = []
        self.user_list_: List[str] = []
        self.item_to_idx_: Dict[str, int] = {}
        self.user_to_idx_: Dict[str, int] = {}
        self.R_: Optional[np.ndarray] = None

    def fit(self, user_item_matrix: np.ndarray, user_list: List[str], item_list: List[str]) -> 'ItemBasedCFKNN':
        self.R_ = np.asarray(user_item_matrix, dtype=np.float64)
        self.user_list_ = list(user_list)
        self.item_list_ = list(item_list)
        self.user_to_idx_ = {u: i for i, u in enumerate(self.user_list_)}
        self.item_to_idx_ = {item: j for j, item in enumerate(self.item_list_)}
        item_vectors = self.R_.T
        self.sim_matrix_ = pairwise_cosine_similarity(item_vectors, item_vectors)
        np.fill_diagonal(self.sim_matrix_, 0.0)
        return self

    def score(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_to_idx_ or item_id not in self.item_to_idx_:
            return 0.0
        u_idx = self.user_to_idx_[user_id]
        i_idx = self.item_to_idx_[item_id]
        user_ratings = self.R_[u_idx]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return 0.0
        sims = self.sim_matrix_[i_idx, rated_items]
        ratings = user_ratings[rated_items]
        if np.sum(np.abs(sims)) == 0:
            return 0.0
        if len(sims) > self.n_neighbors:
            top_indices = np.argsort(sims)[::-1][:self.n_neighbors]
            sims = sims[top_indices]
            ratings = ratings[top_indices]
        pred = np.sum(sims * ratings) / (np.sum(np.abs(sims)) + 1e-09)
        return float(pred)
