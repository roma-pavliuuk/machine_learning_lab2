from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from src.extraction import TextExtractor
from src.features import CyclicTimeTransformer, InteractionTransformer, OOFWoEEncoder
from src.lab1 import lab1

LAB2_STEPS = {
    "cyclic_time": lambda **p: CyclicTimeTransformer(**p),
    "interactions": lambda **p: InteractionTransformer(**p),
    "oof_woe": lambda **p: OOFWoEEncoder(**p),
    "yeo_johnson": lambda **p: PowerTransformer(method="yeo-johnson", **p),
    "tfidf_svd": lambda **p: TextExtractor(**p),
}


def register_lab2_steps() -> None:
    for name, factory in LAB2_STEPS.items():
        existing = lab1._STEP_FACTORIES.get(name)
        if existing is not None and existing is not factory:
            raise ValueError(f"Крок {name!r} уже зареєстровано в ЛР №1 — конфлікт назв")
        lab1._STEP_FACTORIES[name] = factory


register_lab2_steps()

DynamicPipelineBuilder = lab1.DynamicPipelineBuilder
validate_config = lab1.validate_config


def load_config(path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_pipeline(config_path, estimator, selector: Optional[Pipeline] = None) -> Pipeline:
    steps = [("prep", DynamicPipelineBuilder(str(config_path)).build())]
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fairness_report(y_true, p_bad, group, *, privileged, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    granted = np.asarray(p_bad) < threshold
    group = np.asarray(group)
    priv = group == privileged
    rows = {}
    for name, mask in (("privileged", priv), ("unprivileged", ~priv)):
        good = mask & (y_true == 0)
        rows[name] = {"n": int(mask.sum()),
                      "grant_rate": float(granted[mask].mean()),
                      "tpr_good": float(granted[good].mean()) if good.any() else float("nan")}
    return {
        "grant_rate_privileged": rows["privileged"]["grant_rate"],
        "grant_rate_unprivileged": rows["unprivileged"]["grant_rate"],
        "tpr_good_privileged": rows["privileged"]["tpr_good"],
        "tpr_good_unprivileged": rows["unprivileged"]["tpr_good"],
        "DPR": rows["unprivileged"]["grant_rate"] / rows["privileged"]["grant_rate"],
        "EOR": rows["unprivileged"]["tpr_good"] / rows["privileged"]["tpr_good"],
    }


def readable_names(names) -> list:
    short = [n.split("__", 1)[1] for n in names]
    clashes = {s for s in short if short.count(s) > 1}
    return [n.replace("__", ":", 1) if s in clashes else s for n, s in zip(names, short)]


def transformed_frame(fitted_prep, X) -> pd.DataFrame:
    return pd.DataFrame(fitted_prep.transform(X), columns=readable_names(fitted_prep.get_feature_names_out()),
                        index=X.index)
