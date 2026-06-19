"""Qiskit Aer Matrix Product State (MPS) backend wrapper.

Planned for Milestone 1. Implements QuantumSimulatorBackend for
Aer's matrix_product_state method.
"""

from qiskit import QuantumCircuit
from backend.abstract import QuantumSimulatorBackend, SimulationResult


class AerMPSBackend(QuantumSimulatorBackend):
    """Backend wrapping Qiskit Aer's MPS simulator."""

    @property
    def name(self) -> str:
        return "aer_mps"

    @property
    def supports_statevector(self) -> bool:
        return False   # MPS typically returns samples, not full statevector

    @property
    def supports_fidelity(self) -> bool:
        return False

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> SimulationResult:
        # TODO: implement in Milestone 1
        raise NotImplementedError("MPS backend not yet implemented.")