"""
Data preprocessing, text normalization, interaction matrix construction,
and temporal/cold-start splitting for Substack recommendation system.
"""

from typing import Tuple, List, Dict, Any, Optional
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE

# Basic English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def clean_text(text: str) -> str:
    """
    Clean raw text string: lowercase, strip punctuation and digits, remove stop words.
    
    Args:
        text: Raw text string.
        
    Returns:
        str: Cleaned, whitespace-normalized string.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Strip URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    # Remove non-alphabet characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # Tokenize and filter stop words and single characters
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def clean_dataframes(
    df_posts: pd.DataFrame,
    df_interactions: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform validation and cleanup across posts and interaction tables.
    Removes duplicates, missing identifiers, and orphaned interactions.
    """
    # 1. Clean posts
    posts_clean = df_posts.copy()
    posts_clean = posts_clean.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"])
    posts_clean["title"] = posts_clean["title"].fillna("")
    posts_clean["abstract"] = posts_clean["abstract"].fillna("")
    posts_clean["clean_title"] = posts_clean["title"].apply(clean_text)
    posts_clean["clean_abstract"] = posts_clean["abstract"].apply(clean_text)
    posts_clean["combined_text"] = posts_clean["clean_title"] + " " + posts_clean["clean_abstract"]
    
    # 2. Clean interactions
    inter_clean = df_interactions.copy()
    inter_clean = inter_clean.dropna(subset=["user_id", "item_id", "clicked"])
    inter_clean["clicked"] = inter_clean["clicked"].astype(int)
    
    # Keep only interactions corresponding to valid posts
    valid_items = set(posts_clean["item_id"])
    inter_clean = inter_clean[inter_clean["item_id"].isin(valid_items)].copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(inter_clean["timestamp"]):
        inter_clean["timestamp"] = pd.to_datetime(inter_clean["timestamp"])
        
    # Sort chronologically
    inter_clean = inter_clean.sort_values("timestamp").reset_index(drop=True)
    posts_clean = posts_clean.reset_index(drop=True)
    
    return posts_clean, inter_clean


def build_user_item_matrix(
    df_interactions: pd.DataFrame,
    value_col: str = "clicked",
    user_col: str = "user_id",
    item_col: str = "item_id"
) -> Tuple[np.ndarray, csr_matrix, List[str], List[str], float]:
    """
    Build user-item interaction matrix as both dense NumPy array and SciPy CSR sparse matrix.
    
    Returns:
        Tuple:
            - dense_matrix: np.ndarray (sampled to <= 2000 users for memory safety)
            - sparse_matrix: csr_matrix (full matrix)
            - user_list: List of unique user IDs
            - item_list: List of unique item IDs
            - sparsity: percentage of zeros in the matrix
    """
    unique_users = sorted(df_interactions[user_col].unique())
    unique_items = sorted(df_interactions[item_col].unique())
    
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {m: j for j, m in enumerate(unique_items)}
    
    row_indices = df_interactions[user_col].map(user_to_idx).values
    col_indices = df_interactions[item_col].map(item_to_idx).values
    values = df_interactions[value_col].values.astype(np.float32)
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    # Build CSR matrix
    sparse_mat = csr_matrix((values, (row_indices, col_indices)), shape=(n_users, n_items))
    
    # Calculate sparsity
    non_zeros = sparse_mat.nnz
    total_elements = n_users * n_items
    sparsity = (1.0 - (non_zeros / total_elements)) * 100.0 if total_elements > 0 else 0.0
    
    # Dense matrix sample (limit to first 1000 users to prevent out-of-memory errors)
    sample_users_count = min(1000, n_users)
    dense_sample = sparse_mat[:sample_users_count, :].toarray()
    
    return dense_sample, sparse_mat, unique_users, unique_items, sparsity


# =====================================================================
# NumPy Demonstrations (Lab Assignment 1 Requirements)
# =====================================================================

def vectorized_row_normalize(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Vectorized L2 row normalization using NumPy broadcasting: X_norm = X / ||X||_2.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, eps, norms)
    return matrix / norms


def boolean_masking_demo(matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Demonstrate NumPy boolean masking: extract rows with mean activity above threshold.
    """
    row_means = matrix.mean(axis=1)
    mask = row_means > threshold
    return matrix[mask]


def top_k_argsort(scores: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Select indices of top-K elements in descending order using NumPy argsort.
    """
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices[:k]


def matrix_cosine_similarity(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Vectorized pairwise cosine similarity between row vectors using NumPy matrix multiplication:
    S = (A @ A.T) / (||A|| @ ||A||.T)
    """
    normed = vectorized_row_normalize(matrix, eps=eps)
    return np.dot(normed, normed.T)


# =====================================================================
# Split Protocols (Temporal and Cold-Start)
# =====================================================================

def temporal_train_val_test_split(
    df_interactions: pd.DataFrame,
    val_frac: float = VAL_SIZE,
    test_frac: float = TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interaction events strictly chronologically:
    Train = earlier events, Validation = intermediate events, Test = latest events.
    
    Prevents temporal data leakage and simulates real-world production deployment.
    """
    df_sorted = df_interactions.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * (1.0 - val_frac - test_frac))
    val_end = int(n * (1.0 - test_frac))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def cold_start_user_split(
    df_interactions: pd.DataFrame,
    holdout_user_frac: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split readers into warm-start (train) and cold-start (unseen holdout readers).
    
    Used specifically to evaluate system behavior on brand-new readers with zero historical clicks.
    """
    rng = np.random.default_rng(seed)
    all_users = np.array(df_interactions["user_id"].unique())
    rng.shuffle(all_users)
    
    n_cold = int(len(all_users) * holdout_user_frac)
    cold_users = set(all_users[:n_cold])
    
    warm_interactions = df_interactions[~df_interactions["user_id"].isin(cold_users)].copy()
    cold_interactions = df_interactions[df_interactions["user_id"].isin(cold_users)].copy()
    
    return warm_interactions, cold_interactions, list(cold_users)
