import json
from pathlib import Path

import yaml

from src.pipeline import DynamicPipelineBuilder

ROOT = Path(__file__).resolve().parent
CONFIGS = (
    "config/stage2_manual_lr.yaml", "config/stage2_manual_rf.yaml",
    "config/config_lr.yaml", "config/config_rf.yaml",
)


def test_idempotent(config_path):
    a = DynamicPipelineBuilder(config_path).build()
    b = DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b), "Збирання не ідемпотентне"


def test_protected_excluded(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ct = DynamicPipelineBuilder(config_path).build()
    used = {col for _, _, cols in ct.transformers for col in cols}
    leaked = used & set(cfg.get("protected_attributes", []))
    assert not leaked, f"Захищені ознаки в конвеєрі: {leaked}"


def test_protected_not_in_dfs_lineage(config_path, lineage_path=ROOT / "data/processed/dfs_lineage.json"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    protected = set(cfg.get("protected_attributes", []))
    lineage = json.loads(Path(lineage_path).read_text(encoding="utf-8"))
    tainted = [e["name"] for e in lineage
               if any(base.split(".", 1)[1] in protected for base in e["base_columns"])]
    assert not tainted, f"Ознаки DFS, побудовані з захищених колонок: {tainted}"
    return len(lineage)


if __name__ == "__main__":
    for path in CONFIGS:
        test_idempotent(path)
        test_protected_excluded(path)
        print(f"{path}: перевірки пройдено")
    n = test_protected_not_in_dfs_lineage(CONFIGS[-1])
    print(f"DFS: {n} згенерованих ознак — жодна не походить від захищених колонок")
