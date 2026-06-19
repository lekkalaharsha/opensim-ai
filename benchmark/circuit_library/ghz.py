"""Greenberger–Horne–Zeilinger (GHZ) circuit generator.

Produces the maximally entangled state (|0...0> + |1...1>)/√2.
"""

from qiskit import QuantumCircuit


def generate_ghz_circuit(n_qubits: int, depth: int = 1, seed: int = 42) -> QuantumCircuit:
    """Create a GHZ circuit.

    Args:
        n_qubits: Number of qubits (must be >= 2 for entanglement).
        depth: Number of repetitions of the GHZ pattern (default 1).
        seed: Random seed (not used by this deterministic circuit, but kept
            for API consistency).

    Returns:
        Qiskit QuantumCircuit with name 'ghz_Nq_dD'.
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.name = f"ghz_{n_qubits}q_d{depth}"

    if n_qubits == 0:
        return circuit

    for _ in range(depth):
        circuit.h(0)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        if depth > 1:
            circuit.barrier()

    return circuit