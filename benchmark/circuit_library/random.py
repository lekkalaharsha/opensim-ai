"""Random circuit generator with full seed control.

Generates random circuits with SU(4) gates on random qubit pairs,
suitable for benchmarking worst-case entanglement growth.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit as qiskit_random_circuit


def generate_random_circuit(n_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """Create a random circuit with SU(4) two-qubit gates.

    Args:
        n_qubits: Number of qubits.
        depth: Number of gate layers.
        seed: Random seed for reproducibility.

    Returns:
        Qiskit QuantumCircuit with random gates.
    """
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(n_qubits)
    circuit.name = f"random_{n_qubits}q_d{depth}"

    if n_qubits == 0:
        return circuit

    for _ in range(depth):
        # Pick random pairs of qubits (allow repetition within a layer)
        qubit_pairs = rng.choice(n_qubits, size=(n_qubits, 2), replace=True)
        for pair in qubit_pairs:
            if pair[0] != pair[1]:
                # Generate a random SU(4) gate
                su4 = qiskit_random_circuit(
                    num_qubits=2,
                    depth=2,
                    seed=rng.integers(0, 2**31 - 1)
                ).to_gate()
                su4.label = "SU4"
                circuit.append(su4, [int(pair[0]), int(pair[1])])
        circuit.barrier()

    return circuit