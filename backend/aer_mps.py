"""Qiskit Aer Matrix Product State (MPS) backend wrapper.

Implements the QuantumSimulatorBackend ABC for Aer's ``matrix_product_state``
method. Collects compilation time, execution time, and peak GPU memory like the
statevector backend, and additionally records the maximum bipartite
entanglement entropy directly from the MPS Schmidt coefficients.

The MPS representation stores, at each of the ``n - 1`` bonds along the qubit
chain, the Schmidt (singular) values of the contiguous bipartition at that cut.
For an untruncated MPS (Aer's default) these are exact, so the von Neumann
entropy ``S = -Σ pᵢ log₂ pᵢ`` with ``pᵢ = λᵢ²`` is the exact entanglement
entropy across that contiguous cut. The maximum over all bonds is reported.
"""

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from backend.abstract import QuantumSimulatorBackend, SimulationResult
from backend.environment import collect_environment_metadata
from backend.config import AerConfig
from backend.gpu_poll import GpuMemoryPoller


def _max_bond_entropy(lambdas: Sequence[Sequence[float]]) -> float:
    """Return the maximum von Neumann entropy (bits) over all MPS bonds.

    Args:
        lambdas: Per-bond Schmidt coefficient vectors from the Aer MPS, where
            each vector is normalized so that the sum of squares is 1.

    Returns:
        The largest bipartite entanglement entropy across contiguous cuts, in
        bits. Returns 0.0 when there are no bonds (e.g. a single qubit).
    """
    max_entropy = 0.0
    for lam in lambdas:
        probs = np.asarray(lam, dtype=float) ** 2
        probs = probs[probs > 1e-15]
        if probs.size == 0:
            continue
        entropy_bits = float(-np.sum(probs * np.log2(probs)))
        if entropy_bits > max_entropy:
            max_entropy = entropy_bits
    return max_entropy


class AerMPSBackend(QuantumSimulatorBackend):
    """Backend wrapping Qiskit Aer's matrix product state simulator.

    Records, per run:
    - compilation (transpile) time
    - execution time
    - peak GPU memory via :class:`GpuMemoryPoller`
    - maximum bipartite entanglement entropy from the MPS Schmidt coefficients

    Does not return a full statevector or compute fidelity: MPS targets the
    large, low-entanglement regime where materializing a statevector is
    infeasible, so those capabilities are reported as unsupported.
    """

    def __init__(self) -> None:
        """Initialize the Aer MPS backend, honoring the shared Aer config."""
        self.config = AerConfig()
        self._simulator = AerSimulator(method="matrix_product_state")
        self._simulator.set_options(**self.config.to_aer_options())

    # ------------------------------------------------------------------
    # ABC properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "aer_mps"

    @property
    def supports_statevector(self) -> bool:
        return False

    @property
    def supports_fidelity(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def run(self, circuit: QuantumCircuit, shots: int = 0) -> SimulationResult:
        """Execute a circuit on the Aer MPS simulator.

        Args:
            circuit: Qiskit QuantumCircuit to simulate.
            shots: Number of measurement shots (0 = no sampling; the MPS
                structure is saved regardless).

        Returns:
            SimulationResult with timing, peak memory, and entanglement entropy
            populated. ``MemoryError`` and other failures are caught and
            returned as ``success=False``.
        """
        env = collect_environment_metadata()

        # --- compilation timing ---
        compilation_start = time.perf_counter()
        try:
            transpiled = transpile(circuit, self._simulator)
        except Exception as exc:
            compilation_time = time.perf_counter() - compilation_start
            return self._failure(circuit, env, compilation_time, f"Compilation error: {exc}")
        compilation_time = time.perf_counter() - compilation_start

        # Save the MPS structure so we can read the Schmidt coefficients back.
        transpiled = transpiled.copy()
        transpiled.save_matrix_product_state(label="mps")

        # --- execution with memory polling ---
        with GpuMemoryPoller() as poller:
            try:
                exec_start = time.perf_counter()
                result = self._simulator.run(transpiled, shots=shots).result()
                execution_time = time.perf_counter() - exec_start
            except MemoryError as mem_err:
                return self._failure(
                    circuit, env, compilation_time, f"Out of memory: {mem_err}",
                    peak_mb=float(poller.peak_mb),
                )
            except Exception as exc:
                return self._failure(
                    circuit, env, compilation_time, f"Execution error: {exc}",
                    peak_mb=float(poller.peak_mb),
                )

        # --- entanglement entropy from MPS Schmidt coefficients ---
        entropy: Optional[float] = None
        entropy_method: Optional[str] = None
        try:
            _, lambdas = result.data(0)["mps"]
            entropy = _max_bond_entropy(lambdas)
            entropy_method = "exact"
        except Exception:
            entropy = None
            entropy_method = None

        return SimulationResult(
            schema_version="0.1.0",
            backend_name=self.name,
            circuit_name=circuit.name or "unnamed",
            n_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            compilation_time_seconds=compilation_time,
            execution_time_seconds=execution_time,
            total_time_seconds=compilation_time + execution_time,
            peak_gpu_memory_mb=float(poller.peak_mb),
            entropy=entropy,
            entropy_method=entropy_method,
            success=True,
            environment=env,
        )

    def _failure(
        self,
        circuit: QuantumCircuit,
        env,
        compilation_time: float,
        message: str,
        peak_mb: float = 0.0,
    ) -> SimulationResult:
        """Build a failed SimulationResult with consistent metadata."""
        return SimulationResult(
            schema_version="0.1.0",
            backend_name=self.name,
            circuit_name=circuit.name or "unnamed",
            n_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            compilation_time_seconds=compilation_time,
            peak_gpu_memory_mb=peak_mb,
            success=False,
            error_message=message,
            environment=env,
        )
