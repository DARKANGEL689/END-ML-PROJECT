"""
Comprehensive recommendation, ranking, and classification evaluation metrics
implemented completely from scratch using NumPy.
Includes Precision@K, Recall@K, F1@K, Hit Rate@K, MAP@K, NDCG@K, MRR,
Intra-List Diversity, Catalog Coverage, and Novelty.
"""

from typing import List, Dict, Set, Any, Optional
import numpy as np
import pandas as pd


def precision_at_k(actual: List[Any], predicted: List[Any], k: int = 10) -> float:
    """
    Precision@K = |actual intersect predicted[:k]| / k
    """
    if k <= 0:
        return 0.0
    pred_k = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / k)


def recall_at_k(actual: List[Any], predicted: List[Any], k: int = 10) -> float:
    """
    Recall@K = |actual intersect predicted[:k]| / |actual|
    """
    if not actual:
        return 0.0
    pred_k = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / len(actual_set))


def f1_at_k(actual: List[Any], predicted: List[Any], k: int = 10) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(actual, predicted, k=k)
    r = recall_at_k(actual, predicted, k=k)
    if (p + r) == 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def hit_rate_at_k(actual: List[Any], predicted: List[Any], k: int = 10) -> float:
    """Hit Rate@K = 1 if at least one actual item is in predicted[:k], else 0."""
    pred_k = predicted[:k]
    actual_set = set(actual)
    return 1.0 if any(item in actual_set for item in pred_k) else 0.0


def map_at_k(actual: List[Any], predicted: List[Any], k: int = 10) -> float:
    """
    Average Precision at K:
    AP@K = sum_{i=1}^K (Precision@i * rel(i)) / min(|actual|, K)
    """
    if not actual or k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(pred_k):
        if item in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    denom = min(len(actual_set), k)
    return float(score / denom) if denom > 0 else 0.0


def ndcg_at_k(actual: List[Any], predicted: List[Any], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance:
    DCG@K = sum_{i=1}^K rel(i) / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(|actual|, K)} 1 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    """
    if not actual or k <= 0:
        return 0.0
    actual_set = set(actual)
    pred_k = predicted[:k]
    
    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2.0)  # index i=0 -> rank 1 -> log2(2) = 1.0
            
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(min(len(actual_set), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_reciprocal_rank(actual: List[Any], predicted: List[Any]) -> float:
    """
    Reciprocal Rank (MRR) = 1 / rank of first relevant item.
    """
    actual_set = set(actual)
    for i, item in enumerate(predicted):
        if item in actual_set:
            return float(1.0 / (i + 1.0))
    return 0.0


def intra_list_diversity(
    recommended_items: List[str],
    item_vectors_dict: Dict[str, np.ndarray]
) -> float:
    """
    Intra-List Diversity (ILD): Average pairwise cosine distance between recommended items.
    Higher value indicates broader diversity of suggested content.
    """
    valid_vectors = [item_vectors_dict[item] for item in recommended_items if item in item_vectors_dict]
    if len(valid_vectors) < 2:
        return 0.0
        
    mat = np.array(valid_vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = mat / norms
    
    sim_mat = np.dot(normed, normed.T)
    # Cosine distance = 1 - similarity
    dist_mat = 1.0 - sim_mat
    
    n = len(valid_vectors)
    # Upper triangle without diagonal
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(dist_mat[triu_indices]))


def catalog_coverage(all_recommended_items: Set[str], total_catalog_items: Set[str]) -> float:
    """Percentage of catalog items recommended at least once."""
    if not total_catalog_items:
        return 0.0
    return float(len(all_recommended_items.intersection(total_catalog_items)) / len(total_catalog_items)) * 100.0


def novelty_at_k(
    recommended_items: List[str],
    item_popularity_counts: Dict[str, int],
    total_interactions: int,
    k: int = 10
) -> float:
    """
    Novelty@K = (1 / K) * sum_{i in pred[:k]} -log2(P(i))
    P(i) = count(i) / total_interactions. Measures surprise / non-obviousness.
    """
    if k <= 0 or total_interactions <= 0:
        return 0.0
    pred_k = recommended_items[:k]
    novelty_sum = 0.0
    for item in pred_k:
        cnt = item_popularity_counts.get(item, 1)
        p_i = max(cnt / total_interactions, 1e-9)
        novelty_sum += -np.log2(p_i)
    return float(novelty_sum / k)


# =====================================================================
# Batch Recommender Evaluation Harness
# =====================================================================

def evaluate_recommender(
    recommender: Any,
    test_interactions: pd.DataFrame,
    k_list: List[int] = [5, 10, 20],
    max_eval_users: int = 300
) -> Dict[str, float]:
    """
    Evaluate recommender on held-out test interactions across multiple K values.
    Computes Precision@K, Recall@K, MAP@K, NDCG@K, HitRate@K, and MRR.
    """
    # Group test items by user (positive click items)
    positive_test = test_interactions[test_interactions["clicked"] == 1]
    user_ground_truth = positive_test.groupby("user_id")["item_id"].apply(list).to_dict()
    
    eval_users = list(user_ground_truth.keys())[:max_eval_users]
    if not eval_users:
        return {}

    metrics = {f"{m}@{k}": [] for k in k_list for m in ["Precision", "Recall", "MAP", "NDCG", "HitRate"]}
    metrics["MRR"] = []
    
    for u in eval_users:
        actual_items = user_ground_truth[u]
        try:
            recs = recommender.recommend(u, k=max(k_list))
            rec_ids = [item_id for item_id, _ in recs]
        except Exception:
            rec_ids = []
            
        for k in k_list:
            metrics[f"Precision@{k}"].append(precision_at_k(actual_items, rec_ids, k=k))
            metrics[f"Recall@{k}"].append(recall_at_k(actual_items, rec_ids, k=k))
            metrics[f"MAP@{k}"].append(map_at_k(actual_items, rec_ids, k=k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(actual_items, rec_ids, k=k))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(actual_items, rec_ids, k=k))
            
        metrics["MRR"].append(mean_reciprocal_rank(actual_items, rec_ids))

    # Average metrics
    avg_metrics = {m: float(np.mean(vals)) for m, vals in metrics.items()}
    return avg_metrics
