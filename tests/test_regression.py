import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from src.regression import SimpleLinearRegressionScratch, MultipleLinearRegressionScratch, RidgeRegressionScratch, LassoRegressionScratch, LogisticRegressionScratch

def test_simple_linear_regression():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 1.2 + rng.normal(0, 0.1, size=50)
    model = SimpleLinearRegressionScratch().fit(x, y)
    sk_model = LinearRegression().fit(x.reshape(-1, 1), y)
    assert np.isclose(model.slope_, sk_model.coef_[0], atol=0.001)
    assert np.isclose(model.intercept_, sk_model.intercept_, atol=0.001)
    assert model.r2_score(x, y) > 0.95

def test_multiple_linear_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + 3.0
    model_norm = MultipleLinearRegressionScratch().fit_normal_equation(X, y)
    sk_model = LinearRegression().fit(X, y)
    assert np.allclose(model_norm.coef_, sk_model.coef_, atol=0.001)
    assert np.isclose(model_norm.intercept_, sk_model.intercept_, atol=0.001)
    model_gd = MultipleLinearRegressionScratch().fit_gradient_descent(X, y, learning_rate=0.05, epochs=800)
    assert np.allclose(model_gd.coef_, sk_model.coef_, atol=0.1)

def test_ridge_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = np.dot(X, [2.0, -1.0, 0.5, 1.2]) + 1.0
    model = RidgeRegressionScratch(alpha=1.0).fit(X, y)
    sk_model = Ridge(alpha=1.0).fit(X, y)
    assert np.allclose(model.coef_, sk_model.coef_, atol=0.01)
    assert np.isclose(model.intercept_, sk_model.intercept_, atol=0.01)

def test_lasso_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.1, size=100)
    model = LassoRegressionScratch(alpha=0.1, max_iter=500).fit(X, y)
    assert model.coef_[0] > 1.5
    assert abs(model.coef_[1]) < 0.2
    assert abs(model.coef_[2]) < 0.2

def test_logistic_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegressionScratch(learning_rate=0.1, epochs=600).fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.85
