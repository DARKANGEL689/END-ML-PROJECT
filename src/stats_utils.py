from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

def covariance_matrix_scratch(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    if n_samples <= 1:
        raise ValueError('Sample covariance requires at least 2 samples.')
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    return cov

def pearson_correlation_matrix_scratch(X: np.ndarray, eps: float=1e-09) -> np.ndarray:
    cov = covariance_matrix_scratch(X)
    variances = np.diag(cov)
    std_devs = np.sqrt(np.maximum(variances, 0.0))
    std_devs = np.where(std_devs < eps, eps, std_devs)
    corr = cov / np.outer(std_devs, std_devs)
    return np.clip(corr, -1.0, 1.0)

def spearman_rank_correlation_scratch(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    ranked_X = np.zeros_like(X)
    for j in range(n_features):
        col = X[:, j]
        ranked_X[:, j] = pd.Series(col).rank().values
    return pearson_correlation_matrix_scratch(ranked_X)

def variance_inflation_factor_scratch(X: np.ndarray, feature_idx: int) -> float:
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    y = X[:, feature_idx]
    other_indices = [i for i in range(n_features) if i != feature_idx]
    X_others = X[:, other_indices]
    X_design = np.hstack((np.ones((n_samples, 1)), X_others))
    try:
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        y_pred = np.dot(X_design, beta)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = max(0.0, min(r2, 0.9999))
        return float(1.0 / (1.0 - r2))
    except Exception:
        return 1.0

def compute_vif_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df.values
    vif_records = []
    for i, col in enumerate(numeric_df.columns):
        vif = variance_inflation_factor_scratch(X, i)
        vif_records.append({'Feature': col, 'VIF': vif})
    return pd.DataFrame(vif_records).sort_values('VIF', ascending=False).reset_index(drop=True)

def identify_collinear_pairs(corr_df: pd.DataFrame, threshold: float=0.9) -> List[Tuple[str, str, float]]:
    pairs = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], float(r)))
    return pairs
