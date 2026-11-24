import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.knn import KNNClassifierScratch, ContentBasedKNNRecommender

def test_knn_classifier_vs_sklearn():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)
    knn_scratch = KNNClassifierScratch(n_neighbors=5, metric='euclidean').fit(X, y)
    knn_sk = KNeighborsClassifier(n_neighbors=5, metric='euclidean').fit(X, y)
    preds_scratch = knn_scratch.predict(X[:20])
    preds_sk = knn_sk.predict(X[:20])
    assert np.mean(preds_scratch == preds_sk) >= 0.95

def test_content_knn_recommender():
    item_ids = ['post_01', 'post_02', 'post_03']
    vectors = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    recommender = ContentBasedKNNRecommender().fit(item_ids, vectors)
    recs = recommender.recommend_for_history(['post_01'], k=2)
    assert len(recs) == 2
    assert recs[0][0] == 'post_02'
    assert recs[0][1] > 0.85
