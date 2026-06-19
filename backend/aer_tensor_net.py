"""Qiskit Aer tensor network backend wrapper.

Future extension. Will wrap Aer's tensor_network simulation method.
"""

from qiskit import QuantumCircuit
from backend.abstract import QuantumSimulatorBackend, SimulationResult


class AerTensorNetworkBackend(QuantumSimulatorBackend):
    """Backend wrapping Qiskit Aer's tensor network simulator."""

    @property
    def name(self) -> str:
        return "aer_tensor_network"

    @property
    def supports_statevector(self) -> bool:
        return False

    @property
    def supports_fidelity(self) -> bool:
        return False

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> SimulationResult:
        # TODO: implement when Aer's tensor network method stabilises
        raise NotImplementedError("Tensor network backend not yet implemented.")