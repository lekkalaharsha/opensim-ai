# tests/test_entanglement.py
"""Tests for contiguous-cut entanglement entropy and cross-backend agreement."""

import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from benchmark.entanglement import max_contiguous_entropy
from benchmark.circuit_library.ghz import generate_ghz_circuit
from benchmark.circuit_library.random import generate_random_circuit
from backend.aer_statevector import AerStatevectorBackend
from backend.aer_mps import AerMPSBackend


def test_contiguous_entropy_none_statevector():
    """No statevector -> (None, None)."""
    assert max_contiguous_entropy(None, 5) == (None, None)


def test_contiguous_entropy_single_qubit():
    """A single qubit has no bonds -> zero entropy, exact."""
    assert max_contiguous_entropy(Statevector.from_label("0"), 1) == (0.0, "exact")


def test_contiguous_entropy_ghz_is_one_bit():
    """GHZ is maximally entangled across any contiguous cut -> 1 bit."""
    sv = Statevector(generate_ghz_circuit(5, depth=1, seed=0))
    entropy, method = max_contiguous_entropy(sv, 5)
    assert method == "exact"
    assert abs(entropy - 1.0) < 1e-9


def test_contiguous_entropy_product_state_is_zero():
    """A product state (single-qubit gates only) has zero entanglement."""
    qc = QuantumCircuit(4)
    qc.h(range(4))
    entropy, method = max_contiguous_entropy(Statevector(qc), 4)
    assert method == "exact"
    assert abs(entropy) < 1e-9


def test_statevector_and_mps_entropy_agree():
    """The two backends report the same contiguous-cut entropy for a circuit.

    This is the consistency guarantee behind standardizing on contiguous cuts:
    the statevector partial-trace entropy and the MPS Schmidt-coefficient
    entropy describe the same bipartitions and must match.
    """
    circuit = generate_random_circuit(6, depth=4, seed=7)

    sv_result = AerStatevectorBackend().run(circuit)
    sv_entropy, _ = max_contiguous_entropy(sv_result.statevector, circuit.num_qubits)
    mps_entropy = AerMPSBackend().run(circuit).entropy

    assert sv_entropy is not None and mps_entropy is not None
    # A genuinely entangled circuit, so the test is non-trivial.
    assert sv_entropy > 0.1
    assert math.isclose(sv_entropy, mps_entropy, rel_tol=1e-6, abs_tol=1e-6)
