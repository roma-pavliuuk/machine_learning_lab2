from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class _StatelessColumnsMixin:
    def fit(self, X, y=None):
        X = _as_frame(X)
        self._check_params(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _check_input(self, X) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(f"{type(self).__name__}: колонки на вході змінилися після fit")
        return X


class CyclicTimeTransformer(_StatelessColumnsMixin, TransformerMixin, BaseEstimator):
    def __init__(self, periods: Dict[str, float] = None, drop_original: bool = True):
        self.periods = periods
        self.drop_original = drop_original

    def _check_params(self, X):
        if not self.periods:
            raise ValueError("CyclicTimeTransformer: не задано periods")
        absent = [c for c in self.periods if c not in X.columns]
        if absent:
            raise ValueError(f"CyclicTimeTransformer: колонок {absent} немає на вході")
        bad = [c for c, p in self.periods.items() if not p or p <= 0]
        if bad:
            raise ValueError(f"CyclicTimeTransformer: період має бути додатним для {bad}")

    def transform(self, X):
        X = self._check_input(X)
        out = X.copy()
        for col, period in self.periods.items():
            angle = 2 * np.pi * X[col].to_numpy(dtype=float) / period
            out[f"{col}_sin"] = np.sin(angle)
            out[f"{col}_cos"] = np.cos(angle)
        if self.drop_original:
            out = out.drop(columns=list(self.periods))
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        names = [c for c in self.feature_names_in_ if not (self.drop_original and c in self.periods)]
        for col in self.periods:
            names += [f"{col}_sin", f"{col}_cos"]
        return np.asarray(names, dtype=object)


class InteractionTransformer(_StatelessColumnsMixin, TransformerMixin, BaseEstimator):
    def __init__(self, ratios: Sequence[Sequence[str]] = (), products: Sequence[Sequence[str]] = (),
                 epsilon: float = 1e-5):
        self.ratios = ratios
        self.products = products
        self.epsilon = epsilon

    def _pairs(self):
        return [("div", a, b) for a, b in self.ratios or []] + [("x", a, b) for a, b in self.products or []]

    def _check_params(self, X):
        pairs = self._pairs()
        if not pairs:
            raise ValueError("InteractionTransformer: не задано ні ratios, ні products")
        absent = sorted({c for _, a, b in pairs for c in (a, b)} - set(X.columns))
        if absent:
            raise ValueError(f"InteractionTransformer: колонок {absent} немає на вході")

    def transform(self, X):
        X = self._check_input(X)
        out = X.copy()
        self.n_nonfinite_ = 0
        for op, a, b in self._pairs():
            left = X[a].to_numpy(dtype=float)
            right = X[b].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                value = left / (right + self.epsilon) if op == "div" else left * right
            finite = np.isfinite(value)
            self.n_nonfinite_ += int((~finite).sum())
            out[f"{a}_{op}_{b}"] = np.where(finite, value, 0.0)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(list(self.feature_names_in_) + [f"{a}_{op}_{b}" for op, a, b in self._pairs()],
                          dtype=object)


def woe_table(values: np.ndarray, y: np.ndarray, smoothing: float) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(values, dtype=object).to_numpy(), y).reindex(columns=[0, 1], fill_value=0)
    k = len(table)
    p_good = (table[0] + smoothing) / (table[0].sum() + smoothing * k)
    p_bad = (table[1] + smoothing) / (table[1].sum() + smoothing * k)
    woe = np.log(p_good / p_bad)
    return pd.DataFrame({"n": table[0] + table[1], "woe": woe, "iv": (p_good - p_bad) * woe})


class OOFWoEEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, n_splits: int = 5, smoothing: float = 0.5, random_state: int = 0):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    @staticmethod
    def _check_target(y):
        if y is None:
            raise ValueError("OOFWoEEncoder потребує цільову змінну y у fit")
        y = np.asarray(y)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("OOFWoEEncoder очікує бінарну ціль зі значеннями 0/1")
        return y

    def fit(self, X, y=None):
        X = _as_frame(X)
        y = self._check_target(y)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.mapping_, self.iv_ = {}, {}
        for col in X.columns:
            table = woe_table(X[col].to_numpy(), y, self.smoothing)
            self.mapping_[col] = table["woe"].to_dict()
            self.iv_[col] = float(table["iv"].sum())
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X = _as_frame(X)
        y = self._check_target(y)
        self.fit(X, y)
        out = np.zeros(X.shape, dtype=float)
        folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in folds.split(X, y):
            for j, col in enumerate(X.columns):
                table = woe_table(X[col].to_numpy()[fit_idx], y[fit_idx], self.smoothing)
                mapping = table["woe"].to_dict()
                out[enc_idx, j] = [mapping.get(v, 0.0) for v in X[col].to_numpy()[enc_idx]]
        return out

    def transform(self, X):
        check_is_fitted(self, "mapping_")
        X = _as_frame(X)
        out = np.empty(X.shape, dtype=float)
        for j, col in enumerate(self.feature_names_in_):
            mapping = self.mapping_[col]
            out[:, j] = [mapping.get(v, 0.0) for v in X.iloc[:, j].to_numpy()]
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)
