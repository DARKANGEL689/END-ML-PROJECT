import pytest
import pandas as pd
import numpy as np
from src.id3 import ID3DecisionTreeScratch, shannon_entropy, information_gain

def test_entropy_and_gain():
    assert shannon_entropy(np.array([1, 1, 1])) == 0.0
    assert np.isclose(shannon_entropy(np.array([1, 0, 1, 0])), 1.0)
    df = pd.DataFrame({'feature': ['A', 'A', 'B', 'B'], 'target': [1, 1, 0, 0]})
    ig = information_gain(df, 'feature', 'target')
    assert np.isclose(ig, 1.0)

def test_id3_decision_tree():
    df = pd.DataFrame({'length': ['Short', 'Short', 'Long', 'Long'], 'topic': ['Tech', 'Art', 'Tech', 'Art'], 'target': [1, 0, 1, 0]})
    tree = ID3DecisionTreeScratch(max_depth=3).fit(df[['length', 'topic']], df['target'])
    preds = tree.predict(df[['length', 'topic']])
    assert np.all(preds == df['target'].values)
    rules = tree.extract_rules()
    assert len(rules) >= 2
