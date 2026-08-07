"""Tests for interpretability helpers."""
import numpy as np
import pytest

from src.interpretability import permutation_importance


class _FakeModel:
    def __init__(self, n_features):
        self.n_features = n_features

    def predict(self, X, **kwargs):
        # Predict sum of first feature across time steps
        return np.sum(X[:, :, 0], axis=1, keepdims=True)


def test_permutation_importance_shape_and_sorting():
    rng = np.random.default_rng(42)
    n_samples = 30
    seq_len = 10
    n_features = 5
    X = rng.normal(size=(n_samples, seq_len, n_features))
    y = np.sum(X[:, :, 0], axis=1, keepdims=True)
    model = _FakeModel(n_features)
    y_pred = model.predict(X)
    baseline_mae = float(np.mean(np.abs(y - y_pred)))

    result = permutation_importance(
        model, X, y, baseline_mae,
        feature_names=[f"f{i}" for i in range(n_features)],
        n_repeats=3,
        seed=42,
    )
    assert len(result["importances"]) == n_features
    assert result["importances"][0]["feature"] == "f0"
    # First feature should have the largest MAE drop
    assert result["importances"][0]["mae_drop_mean"] >= result["importances"][1]["mae_drop_mean"]


def test_permutation_importance_bad_feature_count_raises():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 5, 3))
    y = np.zeros((10, 1))
    model = _FakeModel(3)
    with pytest.raises(ValueError, match="feature_names length"):
        permutation_importance(model, X, y, 0.0, feature_names=["a", "b"], n_repeats=1)
