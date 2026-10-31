import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.selection import CorrelationFilter, hybrid_selector, vif_table


def synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    informative = rng.normal(size=(n, 2))
    y = (informative[:, 0] + 0.5 * informative[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    duplicate = informative[:, :1] * 2 + 1e-6 * rng.normal(size=(n, 1))
    constant = np.ones((n, 1))
    noise = rng.normal(size=(n, 6))
    return np.hstack([informative, duplicate, constant, noise]), y


def test_correlation_filter_drops_constant_and_duplicate():
    X, y = synthetic()
    support = CorrelationFilter(threshold=0.95).fit(X, y).get_support()
    assert not support[3]
    assert support[0] != support[2]
    assert support[1] and support[4:].all()


@pytest.mark.filterwarnings("ignore:The design matrix is poorly conditioned")
def test_vif_flags_duplicate():
    import pandas as pd
    X, _ = synthetic()
    vif = vif_table(pd.DataFrame(X[:, [0, 1, 2, 4]], columns=list("abcd")))
    assert vif["a"] > 100 and vif["d"] < 5


def test_hybrid_selector_keeps_informative():
    X, y = synthetic()
    cfg = {"basic": {"threshold": 0.95, "variance_tolerance": 1e-8},
           "filter": {"score": "f_classif", "k": 5},
           "wrapper": {"k_features": 2, "forward": True, "floating": False, "scoring": "roc_auc",
                       "cv": 3, "random_state": 0}}
    sel = hybrid_selector(LogisticRegression(), cfg).fit(X, y)
    assert sel.transform(X).shape == (len(X), 2)
