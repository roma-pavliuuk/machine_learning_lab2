from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def safe_name(name: str) -> str:
    name = name.replace(" / ", "_div_")
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    return name.strip("_")


def run_dfs(clients: pd.DataFrame, events: pd.DataFrame, *, client_index: str, event_index: str,
            event_key: str, settings: Dict) -> Tuple[pd.DataFrame, List[Dict], int]:
    import featuretools as ft

    es = ft.EntitySet(id="credit_applicants")
    es = es.add_dataframe(dataframe_name="clients", dataframe=clients.copy(), index=client_index)
    es = es.add_dataframe(dataframe_name="payments", dataframe=events.copy(), index=event_index)
    es = es.add_relationship("clients", client_index, "payments", event_key)

    matrix, definitions = ft.dfs(
        entityset=es,
        target_dataframe_name="clients",
        max_depth=settings["max_depth"],
        agg_primitives=settings["agg_primitives"],
        trans_primitives=settings["trans_primitives"],
        primitive_options=settings.get("primitive_options"),
        ignore_columns=settings.get("ignore_columns"),
        verbose=False,
    )

    lineage = []
    for feature in definitions:
        bases = sorted({f"{b.dataframe_name}.{b.get_name()}" for b in _base_features(feature)})
        lineage.append({"name": safe_name(feature.get_name()),
                        "definition": feature.get_name(),
                        "base_columns": bases})
    matrix = matrix.copy()
    matrix.columns = [safe_name(c) for c in matrix.columns]
    if matrix.columns.duplicated().any():
        raise ValueError("Нормалізація назв DFS дала дублікати")
    numeric = matrix.select_dtypes("number")
    n_infinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    return matrix, lineage, n_infinite


def _base_features(feature):
    deps = feature.get_dependencies(deep=True)
    leaves = [d for d in deps if not d.get_dependencies()]
    return leaves or [feature]
