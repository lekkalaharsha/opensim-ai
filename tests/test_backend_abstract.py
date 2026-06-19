# tests/test_backend_abstract.py
"""Tests for abstract base classes."""

import pytest
from backend.abstract import (
    SimulationResult,
    EnvironmentMetadata,
    QuantumSimulatorBackend,
)


def test_simulation_result_defaults():
    """SimulationResult should initialize with sensible defaults."""
    sr = SimulationResult()
    assert sr.schema_version == "0.1.0"
    assert sr.success is False
    assert sr.fidelity is None
    assert sr.fidelity_method is None
    assert sr.error_message is None
    assert sr.environment is None
    assert sr.circuit_fingerprint == {}


def test_environment_metadata():
    """EnvironmentMetadata should store all fields."""
    env = EnvironmentMetadata(
        python_version="3.11.8",
        qiskit_version="1.0.2",
        qiskit_aer_version="0.14.1",
        gpu_name="Tesla T4",
        gpu_memory_mb=15360,
        cuda_version="12.2",
        host_platform="Linux",
    )
    assert env.python_version == "3.11.8"
    assert env.gpu_memory_mb == 15360


def test_cannot_instantiate_abc():
    """QuantumSimulatorBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        QuantumSimulatorBackend()


def test_concrete_backend_must_implement_all():
    """A partially implemented backend should raise TypeError."""
    class IncompleteBackend(QuantumSimulatorBackend):
        @property
        def name(self):
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteBackend()