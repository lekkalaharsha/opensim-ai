"""Clifford‑only circuit generator.

Uses the stabilizer formalism to create circuits consisting only of
Clifford gates (H, S, CX). These circuits are efficiently simulable
classically but serve as a stress test for MPS and statevector backends.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford


def generate_clifford_circuit(n_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """Generate a random Clifford circuit.

    Args:
        n_qubits: Number of qubits.
        depth: Number of gate layers (each layer adds one random Clifford
            operation on a random subset of qubits).
        seed: Random seed for reproducibility.

    Returns:
        Qiskit QuantumCircuit consisting only of Clifford gates.
    """
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(n_qubits)
    circuit.name = f"clifford_{n_qubits}q_d{depth}"

    if n_qubits < 2:
        # Cannot generate multi-qubit Clifford gates on < 2 qubits.
        # Return an empty circuit, consistent with other generators.
        return circuit

    for _ in range(depth):
        # Pick a random subset of qubits (2–4) to apply a Clifford to
        subset_size = rng.integers(2, min(5, n_qubits + 1))
        qubits = rng.choice(n_qubits, size=subset_size, replace=False)
        qubits = [int(q) for q in qubits]

        # Generate a random Clifford on that many qubits
        cliff = random_clifford(int(subset_size), seed=rng.integers(0, 2**31 - 1))
        # Convert to circuit and append to our main circuit
        cliff_circ = cliff.to_circuit()
        circuit.append(cliff_circ.to_gate(), qubits)

    return circuit