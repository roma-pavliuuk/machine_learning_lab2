import subprocess
import sys

from src.lab1 import LAB1_DIR, LAB1_SHA256, file_sha256, lab1
from src.pipeline import LAB2_STEPS, DynamicPipelineBuilder


def test_lab1_module_is_unchanged():
    assert file_sha256() == LAB1_SHA256, "Файл класу ЛР №1 змінено — клас має імпортуватися без змін"


def test_builder_is_the_lab1_class():
    assert DynamicPipelineBuilder is lab1.DynamicPipelineBuilder


def test_new_steps_registered_in_lab1_registry():
    for name in LAB2_STEPS:
        assert name in lab1._STEP_FACTORIES


def test_lab1_verification_protocol_still_passes():
    result = subprocess.run([sys.executable, "run_verification.py"], cwd=LAB1_DIR,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("перевірки пройдено") == 2


def test_lab1_configs_build_through_bridge():
    for name in ("config_lr.yaml", "config_rf.yaml"):
        ct = DynamicPipelineBuilder(str(LAB1_DIR / "config" / name)).build()
        assert ct.transformers
