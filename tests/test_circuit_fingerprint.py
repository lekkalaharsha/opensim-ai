# tests/test_circuit_fingerprint.py
"""Tests for circuit fingerprint extraction."""

from benchmark.circuit_fingerprint import extract_circuit_fingerprint
from benchmark.circuit_library.ghz import generate_ghz_circuit
from benchmark.circuit_library.qft import generate_qft_circuit


def test_ghz_fingerprint():
    circuit = generate_ghz_circuit(5)
    fp = extract_circuit_fingerprint(circuit)
    assert fp["qubits"] == 5
    assert fp["depth"] > 0
    assert "gate_counts" in fp
    assert "h" in fp["gate_counts"]
    assert "cx" in fp["gate_counts"]
    ig = fp["interaction_graph"]
    assert ig["connected_components"] == 1
    assert ig["diameter"] > 0


def test_qft_fingerprint():
    circuit = generate_qft_circuit(4)
    fp = extract_circuit_fingerprint(circuit)
    assert fp["qubits"] == 4
    assert fp["depth"] > 0
    assert "interaction_graph" in fp