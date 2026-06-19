"""Quantum Fourier Transform circuit generator.

Implements the standard QFT (without final swap reversal, suitable for
simulation benchmarks). Circuit depth scales as O(n²).
"""

from math import pi
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT


def generate_qft_circuit(n_qubits: int, depth: Optional[int] = None, seed: int = 42) -> QuantumCircuit:
    """Create a QFT circuit.

    Args:
        n_qubits: Number of qubits.
        depth: Ignored; included for API consistency. QFT depth is fixed.
        seed: Random seed (unused).

    Returns:
        Qiskit QuantumCircuit implementing the QFT.
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.name = f"qft_{n_qubits}q"

    # Append standard QFT (includes SWAPs at the end)
    qft = QFT(num_qubits=n_qubits, do_swaps=True)
    circuit.append(qft, range(n_qubits))

    return circuit