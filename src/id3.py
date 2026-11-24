from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

def shannon_entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def information_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    total_entropy = shannon_entropy(df[target].values)
    n = len(df)
    if n == 0:
        return 0.0
    values, counts = np.unique(df[feature].values, return_counts=True)
    weighted_entropy = 0.0
    for val, count in zip(values, counts):
        subset = df[df[feature] == val]
        weighted_entropy += count / n * shannon_entropy(subset[target].values)
    return float(total_entropy - weighted_entropy)

class ID3DecisionNode:

    def __init__(self, feature: Optional[str]=None, is_leaf: bool=False, prediction: Optional[Any]=None, probabilities: Optional[Dict[Any, float]]=None):
        self.feature = feature
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.probabilities = probabilities or {}
        self.children: Dict[Any, ID3DecisionNode] = {}

class ID3DecisionTreeScratch:

    def __init__(self, max_depth: int=5, min_samples_split: int=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: Optional[ID3DecisionNode] = None
        self.target_name_: str = 'target'
        self.feature_names_: List[str] = []
        self.default_class_: Any = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'ID3DecisionTreeScratch':
        df = X.copy()
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=df.index, name='target')
        self.target_name_ = y.name if y.name else 'target'
        df[self.target_name_] = y.values
        self.feature_names_ = list(X.columns)
        self.classes_ = np.unique(y.values)
        vals, counts = np.unique(y.values, return_counts=True)
        self.default_class_ = vals[np.argmax(counts)]
        self.root_ = self._build_tree(df, self.feature_names_, depth=0)
        return self

    def _build_tree(self, df: pd.DataFrame, features: List[str], depth: int) -> ID3DecisionNode:
        y_vals = df[self.target_name_].values
        vals, counts = np.unique(y_vals, return_counts=True)
        if len(vals) == 1:
            return ID3DecisionNode(is_leaf=True, prediction=vals[0], probabilities={vals[0]: 1.0})
        total_samples = len(y_vals)
        majority_class = vals[np.argmax(counts)]
        probs = {c: count / total_samples for c, count in zip(vals, counts)}
        if not features or depth >= self.max_depth or total_samples < self.min_samples_split:
            return ID3DecisionNode(is_leaf=True, prediction=majority_class, probabilities=probs)
        best_gain = -1.0
        best_feature = None
        for feat in features:
            gain = information_gain(df, feat, self.target_name_)
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
        if best_feature is None or best_gain <= 1e-06:
            return ID3DecisionNode(is_leaf=True, prediction=majority_class, probabilities=probs)
        node = ID3DecisionNode(feature=best_feature, is_leaf=False, prediction=majority_class, probabilities=probs)
        remaining_features = [f for f in features if f != best_feature]
        for val in df[best_feature].unique():
            subset = df[df[best_feature] == val].drop(columns=[best_feature])
            if len(subset) == 0:
                node.children[val] = ID3DecisionNode(is_leaf=True, prediction=majority_class, probabilities=probs)
            else:
                node.children[val] = self._build_tree(subset, remaining_features, depth + 1)
        return node

    def _predict_row(self, node: ID3DecisionNode, row: pd.Series) -> Any:
        if node.is_leaf or not node.children:
            return node.prediction
        val = row.get(node.feature, None)
        if val in node.children:
            return self._predict_row(node.children[val], row)
        return node.prediction

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_row(self.root_, row) for _, row in X.iterrows()])

    def print_tree(self, node: Optional[ID3DecisionNode]=None, indent: str='') -> str:
        if node is None:
            node = self.root_
        output = ''
        if node.is_leaf:
            output += f'{indent}-> Predict: {node.prediction} (P: {node.probabilities})\n'
        else:
            output += f'{indent}[Split on: {node.feature}]\n'
            for val, child in node.children.items():
                output += f"{indent}  |-- {node.feature} == '{val}':\n"
                output += self.print_tree(child, indent + '      ')
        return output

    def extract_rules(self, node: Optional[ID3DecisionNode]=None, current_rule: Optional[List[str]]=None) -> List[str]:
        if node is None:
            node = self.root_
        if current_rule is None:
            current_rule = []
        rules = []
        if node.is_leaf:
            rule_str = 'IF ' + ' AND '.join(current_rule) + f' THEN Predict {node.prediction}'
            rules.append(rule_str)
        else:
            for val, child in node.children.items():
                new_rule = current_rule + [f"({node.feature} == '{val}')"]
                rules.extend(self.extract_rules(child, new_rule))
        return rules
