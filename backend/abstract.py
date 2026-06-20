"""Core abstractions for quantum simulator backends.

This module defines the standard data structures and interfaces that every
simulator backend must implement. It ensures consistent metrics collection
and result formatting across different simulation methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from qiskit.circuit import QuantumCircuit
import numpy as np


@dataclass
class EnvironmentMetadata:
    """Hardware and software environment information for reproducibility.

    Attributes:
        python_version: Python interpreter version (e.g., '3.11.8').
        qiskit_version: Installed Qiskit package version.
        qiskit_aer_version: Installed Qiskit Aer package version.
        gpu_name: GPU model name (e.g., 'Tesla T4'), or 'CPU' if none.
        gpu_memory_mb: Total GPU memory in megabytes.
        cuda_version: CUDA toolkit version, if available.
        host_platform: Operating system platform string.
    """
    python_version: str = ""
    qiskit_version: str = ""
    qiskit_aer_version: str = ""
    gpu_name: str = ""
    gpu_memory_mb: int = 0
    cuda_version: str = ""
    host_platform: str = ""


@dataclass
class SimulationResult:
    """Standardized benchmark result returned by any backend.

    This dataclass is the single source of truth for all metrics collected
    during a single circuit simulation run. Every backend must populate
    these fields consistently.

    Attributes:
        schema_version: Version of the result schema (currently '0.1.0').
        backend_name: Identifier of the backend used (e.g., 'aer_statevector').
        circuit_name: Name of the circuit executed.
        n_qubits: Number of qubits in the circuit.
        depth: Circuit depth (number of layers of gates).
        compilation_time_seconds: Time spent in transpilation/compilation.
        execution_time_seconds: Time spent in actual simulation (backend.run).
        total_time_seconds: Sum of compilation and execution time.
        peak_gpu_memory_mb: Maximum GPU memory observed during simulation.
        fidelity: State fidelity against ideal result, if computable.
        fidelity_method: Method used for fidelity ('exact', 'xeb', or None).
        entropy: Maximum bipartite entanglement entropy, if computed.
        entropy_method: Method used for entropy ('exact', 'approximate', or None).
        statevector: The final statevector, if the backend supports it. Note: this
            field is for in-memory use and is not serialized to the output JSON
            file due to its potential size.
        success: Whether the simulation completed without error.
        error_message: Error description if success is False.
        environment: Hardware/software environment metadata.
        circuit_fingerprint: Structural features of the circuit (ML input).
    """
    schema_version: str = "0.1.0"
    backend_name: str = ""
    circuit_name: str = ""
    n_qubits: int = 0
    depth: int = 0
    compilation_time_seconds: float = 0.0
    execution_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    fidelity: Optional[float] = None
    fidelity_method: Optional[str] = None
    entropy: Optional[float] = None         # max contiguous-cut entropy (primary)
    entropy_method: Optional[str] = None
    entropy_middle: Optional[float] = None  # entropy at center bond k=(n-1)//2
    entropy_avg: Optional[float] = None     # mean over all n-1 contiguous cuts
    entropy_var: Optional[float] = None     # variance over all n-1 contiguous cuts
    statevector: Optional[np.ndarray] = None
    success: bool = False
    error_message: Optional[str] = None
    environment: Optional[EnvironmentMetadata] = None
    circuit_fingerprint: Dict[str, Any] = field(default_factory=dict)


class QuantumSimulatorBackend(ABC):
    """Abstract base class for all quantum circuit simulator backends.

    Any new backend (statevector, MPS, tensor network, etc.) must implement
    this interface. This ensures the benchmark runner can treat all backends
    uniformly.

    Usage:
        class MyBackend(QuantumSimulatorBackend):
            def run(self, circuit, shots=0):
                # ... implementation ...
                return SimulationResult(...)

            @property
            def name(self):
                return "my_backend"
    """

    @abstractmethod
    def run(self, circuit: QuantumCircuit, shots: int = 0) -> SimulationResult:
        """Execute a quantum circuit and return a standardized result.

        Args:
            circuit: A Qiskit QuantumCircuit.
            shots: Number of measurement shots (0 = statevector mode).

        Returns:
            A SimulationResult containing all performance metrics and metadata.

        Raises:
            MemoryError: If the simulation runs out of GPU memory (should be
                caught internally and returned as success=False).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        pass

    @property
    @abstractmethod
    def supports_statevector(self) -> bool:
        """Whether this backend can return a full statevector."""
        pass

    @property
    @abstractmethod
    def supports_fidelity(self) -> bool:
        """Whether this backend can compute exact fidelity."""
        pass