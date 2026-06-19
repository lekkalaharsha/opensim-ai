"""QAOA ansatz circuit generator.

Creates a parameterised QAOA circuit for MaxCut on a random graph where each
node has a maximum degree of 3.
The depth parameter corresponds to the number of layers (p).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def generate_qaoa_circuit(n_qubits: int, depth: int = 1, seed: int = 42) -> QuantumCircuit:
    """Generate a QAOA ansatz circuit for MaxCut.

    The circuit is built on a random graph where nodes have a maximum degree of 3.

    Args:
        n_qubits: Number of qubits (nodes in the graph).
        depth: Number of QAOA layers (p). Higher p = more expressive.
        seed: Random seed for graph generation.

    Returns:
        Qiskit QuantumCircuit with placeholder parameters (β, γ per layer).
    """
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(n_qubits)
    circuit.name = f"qaoa_{n_qubits}q_p{depth}"

    edges = set()
    # This heuristic attempts to generate a 3-regular graph. It requires
    # n_qubits >= 2 and works best for even n_qubits >= 4. For other cases,
    # it will generate a graph that may not be 3-regular.
    if n_qubits >= 2:
        degrees = [0] * n_qubits
        # Use a while loop with a safety break to avoid infinite loops.
        max_attempts = n_qubits * 10
        attempts = 0
        target_edges = (3 * n_qubits) // 2
        while len(edges) < target_edges and attempts < max_attempts:
            attempts += 1
            q1, q2 = rng.choice(n_qubits, size=2, replace=False)
            if degrees[q1] < 3 and degrees[q2] < 3:
                edge = tuple(sorted((int(q1), int(q2))))
                if edge not in edges:
                    edges.add(edge)
                    degrees[q1] += 1
                    degrees[q2] += 1

    # Initial layer: Hadamard on all qubits
    circuit.h(range(n_qubits))

    # QAOA layers
    for layer in range(depth):
        gamma = Parameter(f"γ_{layer}")
        beta = Parameter(f"β_{layer}")

        # Cost Hamiltonian: ZZ interactions for each edge
        for (u, v) in edges:
            circuit.cx(u, v)
            circuit.rz(2 * gamma, v)   # e^{-iγ ZZ} equivalent
            circuit.cx(u, v)

        # Mixer Hamiltonian: X rotations
        for i in range(n_qubits):
            circuit.rx(2 * beta, i)

    return circuit