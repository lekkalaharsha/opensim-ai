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

This is the same contiguous-cut definition the statevector backend uses (via
``benchmark.entanglement.max_contiguous_entropy``), so entropy is directly
comparable across the two backends — here it is read straight from the Schmidt
coefficients with no statevector materialized.
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


def _all_bond_entropies(lambdas: Sequence[Sequence[float]]) -> List[float]:
    """Return per-bond von Neumann entropy (bits) for every MPS bond.

    Args:
        lambdas: Per-bond Schmidt coefficient vectors from the Aer MPS.

    Returns:
        List of entropy values, one per bond (length = n_qubits - 1).
    """
    result = []
    for lam in lambdas:
        probs = np.asarray(lam, dtype=float) ** 2
        probs = probs[probs > 1e-15]
        if probs.size == 0:
            result.append(0.0)
        else:
            result.append(float(-np.sum(probs * np.log2(probs))))
    return result


def _max_bond_entropy(lambdas: Sequence[Sequence[float]]) -> float:
    """Return the maximum von Neumann entropy (bits) over all MPS bonds."""
    entropies = _all_bond_entropies(lambdas)
    return max(entropies) if entropies else 0.0


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
        entropy_middle: Optional[float] = None
        entropy_avg: Optional[float] = None
        entropy_var: Optional[float] = None
        entropy_method: Optional[str] = None
        try:
            _, lambdas = result.data(0)["mps"]
            bond_ents = _all_bond_entropies(lambdas)
            if bond_ents:
                entropy = max(bond_ents)
                entropy_middle = bond_ents[len(bond_ents) // 2]
                entropy_avg = sum(bond_ents) / len(bond_ents)
                entropy_var = float(np.var(bond_ents))
                entropy_method = "exact"
        except Exception:
            pass

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
            entropy_middle=entropy_middle,
            entropy_avg=entropy_avg,
            entropy_var=entropy_var,
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
