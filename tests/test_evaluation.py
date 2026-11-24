"""Unit tests for ranking and recommendation metrics on hand-computed examples."""
import pytest
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    mean_reciprocal_rank
)

def test_evaluation_metrics_hand_computed():
    actual = ["item_A", "item_B", "item_C"]
    predicted = ["item_A", "item_X", "item_B", "item_Y", "item_Z"]
    
    # Top 3 predicted: [A, X, B]. Hits: A, B (2 hits).
    # Precision@3 = 2 / 3
    assert precision_at_k(actual, predicted, k=3) == pytest.approx(2.0 / 3.0)
    
    # Recall@3 = 2 / 3
    assert recall_at_k(actual, predicted, k=3) == pytest.approx(2.0 / 3.0)
    
    # Hit rate@3 = 1.0
    assert hit_rate_at_k(actual, predicted, k=3) == 1.0
    
    # Hit rate@1 for disjoint set
    assert hit_rate_at_k(["item_W"], predicted, k=3) == 0.0
    
    # MRR: First hit is item_A at rank 1 -> 1 / 1 = 1.0
    assert mean_reciprocal_rank(actual, predicted) == 1.0
    
    # If first hit at rank 2:
    pred2 = ["item_X", "item_A"]
    assert mean_reciprocal_rank(actual, pred2) == 0.5
    
    # NDCG@3:
    # rank 1: item_A (hit, rel=1, disc=1/log2(2)=1.0)
    # rank 2: item_X (miss, rel=0)
    # rank 3: item_B (hit, rel=1, disc=1/log2(4)=0.5)
    # DCG = 1.0 + 0.5 = 1.5
    # IDCG (3 items): 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.63092975 + 0.5 = 2.13092975
    # NDCG = 1.5 / 2.13092975 = 0.7039
    ndcg3 = ndcg_at_k(actual, predicted, k=3)
    assert ndcg3 == pytest.approx(1.5 / (1.0 + 1.0/1.5849625 + 0.5), rel=1e-3)
