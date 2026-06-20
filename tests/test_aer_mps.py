# tests/test_aer_mps.py
"""Tests for the Aer matrix product state backend."""

import json
import tempfile
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from backend.aer_mps import AerMPSBackend, _max_bond_entropy
from benchmark.circuit_library.ghz import generate_ghz_circuit
from benchmark.runner import run_single_benchmark, load_backend


def test_mps_backend_properties():
    """ABC properties match the MPS capability contract."""
    backend = AerMPSBackend()
    assert backend.name == "aer_mps"
    assert backend.supports_statevector is False
    assert backend.supports_fidelity is False


def test_mps_runs_ghz_successfully():
    """A GHZ circuit runs and records three-part timing."""
    backend = AerMPSBackend()
    result = backend.run(generate_ghz_circuit(n_qubits=5, depth=1, seed=42))

    assert result.success is True
    assert result.backend_name == "aer_mps"
    assert result.n_qubits == 5
    assert result.compilation_time_seconds > 0
    assert result.execution_time_seconds > 0
    assert abs(result.total_time_seconds
               - (result.compilation_time_seconds + result.execution_time_seconds)) < 1e-9
    assert result.environment is not None


def test_mps_ghz_entropy_is_one_bit():
    """GHZ is maximally entangled across any cut: entropy = 1 bit, exact."""
    backend = AerMPSBackend()
    result = backend.run(generate_ghz_circuit(n_qubits=6, depth=1, seed=0))

    assert result.entropy_method == "exact"
    assert result.entropy is not None
    assert abs(result.entropy - 1.0) < 1e-6


def test_mps_product_state_has_zero_entropy():
    """A product state (no two-qubit gates) has zero entanglement entropy."""
    circuit = QuantumCircuit(4, name="product")
    circuit.h(range(4))  # single-qubit gates only -> unentangled

    result = AerMPSBackend().run(circuit)

    assert result.success is True
    assert result.entropy is not None
    assert abs(result.entropy) < 1e-6


def test_mps_via_runner_writes_valid_json():
    """The MPS backend integrates with the runner and produces a valid record."""
    with tempfile.TemporaryDirectory() as tmpdir:
        circuit = generate_ghz_circuit(3, seed=0)
        result = run_single_benchmark(circuit, load_backend("aer_mps"), output_dir=tmpdir)

        assert result.success is True
        safe_name = result.circuit_name.replace(" ", "_").replace("/", "_")
        json_path = Path(tmpdir) / f"{safe_name}_{result.backend_name}.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["backend_name"] == "aer_mps"
        assert data["success"] is True
        assert data["circuit_fingerprint"]  # fingerprint populated by the runner


def test_max_bond_entropy_edge_cases():
    """Empty bonds yield 0; a trivial bond yields 0; a balanced cut yields 1 bit."""
    assert _max_bond_entropy([]) == 0.0
    assert _max_bond_entropy([[1.0]]) == 0.0
    import math
    assert abs(_max_bond_entropy([[math.sqrt(0.5), math.sqrt(0.5)]]) - 1.0) < 1e-9
