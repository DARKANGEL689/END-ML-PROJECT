import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.stats_utils import covariance_matrix_scratch, pearson_correlation_matrix_scratch, spearman_rank_correlation_scratch, variance_inflation_factor_scratch

def test_covariance_and_correlation():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4))
    cov_scratch = covariance_matrix_scratch(X)
    cov_np = np.cov(X, rowvar=False)
    assert np.allclose(cov_scratch, cov_np)
    corr_scratch = pearson_correlation_matrix_scratch(X)
    corr_np = np.corrcoef(X, rowvar=False)
    assert np.allclose(corr_scratch, corr_np)

def test_spearman_and_vif():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    X[:, 2] = 2.0 * X[:, 0] + rng.normal(0, 0.01, size=50)
    vif_col2 = variance_inflation_factor_scratch(X, 2)
    assert vif_col2 > 10.0
