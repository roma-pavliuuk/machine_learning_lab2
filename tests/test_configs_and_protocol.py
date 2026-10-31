import ast
import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.pipeline import DynamicPipelineBuilder

from conftest import ALL_CONFIGS, CONFIG_PAIRS, ROOT, load_yaml

SRC_FILES = ["features.py", "extraction.py", "selection.py", "pipeline.py", "autofe.py"]


@pytest.mark.parametrize("pair", CONFIG_PAIRS, ids=lambda p: p[0].stem)
def test_config_pair_differs_only_in_model_family(pair):
    lr, rf = (p.read_text(encoding="utf-8").splitlines() for p in pair)
    diff = [(a, b) for a, b in zip(lr, rf) if a != b]
    assert len(lr) == len(rf) and len(diff) == 1
    assert all(line.startswith("model_family:") for line in diff[0])


def test_dfs_group_in_sync_with_lineage():
    lineage = json.loads((ROOT / "data/processed/dfs_lineage.json").read_text(encoding="utf-8"))
    for path in ALL_CONFIGS[2:]:
        assert load_yaml(path)["features"]["dfs_features"] == [e["name"] for e in lineage]


def test_build_unfitted(config_path):
    with pytest.raises(NotFittedError):
        check_is_fitted(DynamicPipelineBuilder(config_path).build())


def test_run_verification_passes():
    result = subprocess.run([sys.executable, "run_verification.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("перевірки пройдено") == 4
    assert "жодна не походить від захищених колонок" in result.stdout


def test_protected_absent_after_fit(split):
    X_train, _, y_train, _ = split
    for path in ALL_CONFIGS[2:]:
        cfg = load_yaml(path)
        ct = DynamicPipelineBuilder(str(path)).build().fit(X_train, y_train)
        names = [n.split("__", 1)[1] for n in ct.get_feature_names_out()]
        for attr in cfg["protected_attributes"]:
            assert not any(n == attr or n.startswith(f"{attr}_") for n in names)


def test_full_pipeline_has_no_nan(split):
    X_train, X_test, y_train, _ = split
    for path in ALL_CONFIGS[2:]:
        ct = DynamicPipelineBuilder(str(path)).build().fit(X_train, y_train)
        assert not np.isnan(ct.transform(X_test)).any()


def test_no_dataset_literals_in_lab2_mechanism():
    forbidden = set()
    for path in ALL_CONFIGS:
        cfg = load_yaml(path)
        forbidden |= set(cfg["transform_policy"]) | set(cfg["features"])
        forbidden |= {c for cols in cfg["features"].values() for c in cols}
        forbidden |= set(cfg["protected_attributes"]) | {cfg["target"]}
    for name in SRC_FILES:
        tree = ast.parse((ROOT / "src" / name).read_text(encoding="utf-8"))
        strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not strings & forbidden, (name, strings & forbidden)


def test_readable_names_keep_group_on_clash():
    from src.pipeline import readable_names
    names = ["num__a_div_b", "dfs__a_div_b", "num__c"]
    assert readable_names(names) == ["num:a_div_b", "dfs:a_div_b", "c"]


def test_full_space_names_unique(split):
    from src.pipeline import readable_names
    X_train, _, y_train, _ = split
    ct = DynamicPipelineBuilder(str(ALL_CONFIGS[2])).build().fit(X_train, y_train)
    names = readable_names(ct.get_feature_names_out())
    assert len(names) == len(set(names))
