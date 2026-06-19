"""Hardware‑efficient variational ansatz circuit generator.

Produces a layered ansatz with Ry and Rz single‑qubit rotations followed
by a ring of CNOT entanglers. This structure is commonly used for
variational quantum algorithms.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def generate_variational_circuit(n_qubits: int, depth: int = 1, seed: int = 42) -> QuantumCircuit:
    """Create a hardware‑efficient variational ansatz.

    Args:
        n_qubits: Number of qubits.
        depth: Number of layers (each layer = Ry, Rz, CNOT ring).
        seed: Random seed (unused; included for API consistency).

    Returns:
        Qiskit QuantumCircuit with placeholder parameters.
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.name = f"var_{n_qubits}q_d{depth}"

    param_counter = 0

    for layer in range(depth):
        # Single‑qubit rotations
        for i in range(n_qubits):
            circuit.ry(Parameter(f"θ_{param_counter}"), i)
            param_counter += 1
            circuit.rz(Parameter(f"φ_{param_counter}"), i)
            param_counter += 1

        # Entangling ring of CNOTs (linear connectivity)
        for i in range(n_qubits - 1):
            circuit.cx(i, i + 1)
        # Optional: close the ring for periodic boundary
        if n_qubits > 2:
            circuit.cx(n_qubits - 1, 0)

    return circuit