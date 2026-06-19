# tests/test_kaggle_environment.py
"""Tests for Kaggle environment validation."""

from kaggle.environment import validate_kaggle_environment, detect_gpu


def test_detect_gpu():
    name, mem = detect_gpu()
    assert isinstance(name, str)
    assert isinstance(mem, int)


def test_validate_kaggle_environment():
    """Should return a report, even without GPU."""
    report = validate_kaggle_environment()
    assert hasattr(report, "is_valid")
    assert hasattr(report, "gpu_name")
    assert isinstance(report.issues, list)
    assert isinstance(report.warnings, list)