"""Unit tests for from-scratch distance and similarity measures."""
import pytest
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine, hamming
from src.distances import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    chebyshev_distance,
    cosine_similarity,
    cosine_distance,
    jaccard_similarity_sets,
    hamming_distance,
    pairwise_euclidean,
    pairwise_manhattan,
    pairwise_cosine_distance
)

def test_scalar_distances():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 0.0, -1.0])
    
    assert np.isclose(euclidean_distance(u, v), euclidean(u, v))
    assert np.isclose(manhattan_distance(u, v), cityblock(u, v))
    assert np.isclose(chebyshev_distance(u, v), chebyshev(u, v))
    assert np.isclose(cosine_distance(u, v), cosine(u, v))
    assert np.isclose(minkowski_distance(u, v, p=1.0), cityblock(u, v))
    assert np.isclose(minkowski_distance(u, v, p=2.0), euclidean(u, v))

def test_jaccard_and_hamming():
    s1 = {"apple", "banana", "cherry"}
    s2 = {"banana", "cherry", "date", "fig"}
    # intersection: 2, union: 5 -> 0.4
    assert np.isclose(jaccard_similarity_sets(s1, s2), 0.4)
    
    a1 = np.array([1, 0, 1, 1, 0])
    a2 = np.array([1, 1, 1, 0, 0])
    assert np.isclose(hamming_distance(a1, a2), 0.4)

def test_pairwise_distances():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y = np.array([[1.0, 2.0], [0.0, 0.0]])
    
    p_euc = pairwise_euclidean(X, Y)
    assert p_euc.shape == (3, 2)
    assert np.isclose(p_euc[0, 0], 0.0)
    
    p_man = pairwise_manhattan(X, Y)
    assert p_man.shape == (3, 2)
    assert np.isclose(p_man[0, 0], 0.0)
    
    p_cos = pairwise_cosine_distance(X, Y)
    assert p_cos.shape == (3, 2)
    assert np.isclose(p_cos[0, 0], 0.0)
