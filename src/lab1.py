import hashlib
import importlib.util
import os
import sys
from pathlib import Path

MODULE_NAME = "lab01_dynamic_pipeline_builder"
LAB1_DIR = Path(os.environ.get("LAB1_DIR", Path(__file__).resolve().parents[2] / "lab_01"))
LAB1_FILE = LAB1_DIR / "src" / "dynamic_pipeline_builder.py"
LAB1_SHA256 = "d0b8c633121e08f92e3f2435c2c5851b966967e83c78693155d5531a0b630b08"


def file_sha256(path: Path = LAB1_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    if not LAB1_FILE.exists():
        raise FileNotFoundError(
            f"Не знайдено препроцесор ЛР №1: {LAB1_FILE}. "
            f"Покладіть lab_01 поруч із lab_02 або задайте LAB1_DIR."
        )
    spec = importlib.util.spec_from_file_location(MODULE_NAME, LAB1_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


lab1 = _load()
