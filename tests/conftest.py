import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_PAIRS = [
    (ROOT / "config" / "stage2_manual_lr.yaml", ROOT / "config" / "stage2_manual_rf.yaml"),
    (ROOT / "config" / "config_lr.yaml", ROOT / "config" / "config_rf.yaml"),
]
ALL_CONFIGS = [p for pair in CONFIG_PAIRS for p in pair]
DATASET = ROOT / "data" / "processed" / "dataset.csv"


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(params=ALL_CONFIGS, ids=lambda p: p.name)
def config_path(request) -> str:
    return str(request.param)


@pytest.fixture(scope="session")
def dataset():
    if not DATASET.exists():
        pytest.skip("data/processed/dataset.csv ще не зібрано (scripts/build_dataset.py)")
    df = pd.read_csv(DATASET)
    return df.drop(columns="bad_credit"), df["bad_credit"]


@pytest.fixture(scope="session")
def split(dataset):
    X, y = dataset
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
