# tests/test_environment.py
"""Tests for environment metadata collection."""

from backend.environment import collect_environment_metadata, _detect_gpu


def test_collect_environment_metadata():
    """Should return a fully populated EnvironmentMetadata."""
    env = collect_environment_metadata()
    assert env.python_version != ""
    assert env.host_platform != ""
    # Qiskit may not be installed in CI, but should return a string
    assert isinstance(env.qiskit_version, str)
    assert isinstance(env.qiskit_aer_version, str)
    # GPU may be 'CPU' if none
    assert isinstance(env.gpu_name, str)
    assert env.gpu_memory_mb >= 0


def test_detect_gpu_returns_tuple():
    """GPU detection returns (str, int)."""
    name, mem = _detect_gpu()
    assert isinstance(name, str)
    assert isinstance(mem, int)
    assert mem >= 0