import pytest
from src.evaluation import precision_at_k, recall_at_k, f1_at_k, hit_rate_at_k, map_at_k, ndcg_at_k, mean_reciprocal_rank

def test_evaluation_metrics_hand_computed():
    actual = ['item_A', 'item_B', 'item_C']
    predicted = ['item_A', 'item_X', 'item_B', 'item_Y', 'item_Z']
    assert precision_at_k(actual, predicted, k=3) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(actual, predicted, k=3) == pytest.approx(2.0 / 3.0)
    assert hit_rate_at_k(actual, predicted, k=3) == 1.0
    assert hit_rate_at_k(['item_W'], predicted, k=3) == 0.0
    assert mean_reciprocal_rank(actual, predicted) == 1.0
    pred2 = ['item_X', 'item_A']
    assert mean_reciprocal_rank(actual, pred2) == 0.5
    ndcg3 = ndcg_at_k(actual, predicted, k=3)
    assert ndcg3 == pytest.approx(1.5 / (1.0 + 1.0 / 1.5849625 + 0.5), rel=0.001)
