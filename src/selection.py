from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest, SelectorMixin, f_classif,
                                       mutual_info_classif)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

SCORE_FUNCTIONS = {"f_classif": f_classif, "mutual_info_classif": mutual_info_classif}


def vif_table(X: pd.DataFrame) -> pd.Series:
    import warnings

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    values = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vif = [variance_inflation_factor(values, j + 1) for j in range(X.shape[1])]
    return pd.Series(vif, index=X.columns, name="VIF").sort_values(ascending=False)


def basic_statistics(X: pd.DataFrame, y, corr_threshold: float = 0.95) -> Dict:
    corr = X.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("r").reset_index()
    pairs = pairs[pairs["r"].abs() >= corr_threshold].sort_values("r", key=np.abs, ascending=False)
    target_corr = X.apply(lambda c: np.corrcoef(c, y)[0, 1] if c.std() > 0 else 0.0)
    return {
        "describe": X.describe().T,
        "near_constant": X.columns[X.std() <= 1e-12].tolist(),
        "corr_with_target": target_corr.sort_values(key=np.abs, ascending=False),
        "high_corr_pairs": pairs,
        "corr": corr,
    }


class CorrelationFilter(SelectorMixin, BaseEstimator):
    def __init__(self, threshold: float = 0.95, variance_tolerance: float = 1e-8):
        self.threshold = threshold
        self.variance_tolerance = variance_tolerance

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        variable = X.var(axis=0) > self.variance_tolerance
        idx = np.flatnonzero(variable)
        corr = np.abs(np.corrcoef(X[:, idx], rowvar=False)) if len(idx) > 1 else np.ones((1, 1))
        if y is not None:
            yc = np.asarray(y, dtype=float)
            relevance = np.abs([np.corrcoef(X[:, j], yc)[0, 1] for j in idx])
            order = np.argsort(-relevance, kind="stable")
        else:
            order = np.arange(len(idx))
        kept: List[int] = []
        for pos in order:
            if all(corr[pos, k] < self.threshold for k in kept):
                kept.append(pos)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[idx[kept]] = True
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


def make_filter(cfg: Dict) -> SelectKBest:
    return SelectKBest(score_func=SCORE_FUNCTIONS[cfg["score"]], k=cfg["k"])


def make_wrapper(estimator, cfg: Dict):
    from mlxtend.feature_selection import SequentialFeatureSelector

    return SequentialFeatureSelector(
        estimator, k_features=cfg["k_features"], forward=cfg["forward"], floating=cfg["floating"],
        scoring=cfg["scoring"], cv=StratifiedKFold(cfg["cv"], shuffle=True, random_state=cfg["random_state"]),
        n_jobs=cfg.get("n_jobs", 1),
    )


def hybrid_selector(estimator, selection_cfg: Dict) -> Pipeline:
    return Pipeline([
        ("basic", CorrelationFilter(**selection_cfg["basic"])),
        ("filter", make_filter(selection_cfg["filter"])),
        ("wrapper", make_wrapper(estimator, selection_cfg["wrapper"])),
    ])


class SelectionFunnel:
    def __init__(self, cfg: Dict, random_state: int):
        self.cfg = cfg
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y) -> "SelectionFunnel":
        cfg, rs = self.cfg, self.random_state
        y = np.asarray(y)
        self.stages_: Dict[str, List[str]] = {"initial": [str(c) for c in X.columns]}

        basic = CorrelationFilter(**cfg["basic"]).fit(X, y)
        cols = [str(c) for c in X.columns[basic.get_support()]]
        self.stages_["basic_statistics"] = cols

        flt = make_filter(cfg["filter"]).fit(X[cols], y)
        self.filter_scores_ = pd.Series(flt.scores_, index=cols).sort_values(ascending=False)
        cols = [str(c) for c in np.array(cols)[flt.get_support()]]
        self.stages_["filter"] = cols

        lr = LogisticRegression(max_iter=1000)
        sfs = make_wrapper(lr, cfg["wrapper"]).fit(X[cols].to_numpy(), y)
        self.sfs_ = sfs
        cols = [cols[i] for i in sfs.k_feature_idx_]
        self.stages_["hybrid_sfs"] = cols

        l1 = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=cfg["embedded"]["C"],
                                random_state=rs, max_iter=1000)
        emb = SelectFromModel(l1, threshold=1e-10).fit(X[cols], y)
        self.l1_coef_ = pd.Series(emb.estimator_.coef_[0], index=cols)
        cols = [str(c) for c in np.array(cols)[emb.get_support()]]
        self.stages_["embedded_l1"] = cols

        X_fit, X_val, y_fit, y_val = train_test_split(X[cols], y, test_size=cfg["tree"]["validation_size"],
                                                      stratify=y, random_state=rs)
        rf = RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1).fit(X_fit, y_fit)
        imp = permutation_importance(rf, X_val, y_val, scoring="roc_auc", n_repeats=cfg["tree"]["n_repeats"],
                                     random_state=rs, n_jobs=-1)
        self.permutation_importance_ = pd.DataFrame(
            {"mean": imp.importances_mean, "std": imp.importances_std}, index=cols).sort_values("mean",
                                                                                                ascending=False)
        self.stages_["tree_top"] = list(self.permutation_importance_.index[:cfg["tree"]["top_k"]])

        rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=cfg["wrapper"]["k_features"])
        rfe.fit(X[self.stages_["filter"]], y)
        self.rfe_selected_ = [str(c) for c in np.array(self.stages_["filter"])[rfe.get_support()]]
        return self

    def counts(self) -> pd.Series:
        return pd.Series({k: len(v) for k, v in self.stages_.items()}, name="n_features")
