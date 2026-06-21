"""Qiskit Aer 'automatic' backend wrapper — the baseline to beat.

Aer already auto-selects a simulation method (statevector / MPS / stabilizer /
density-matrix) per circuit using built-in heuristics. This wraps that as a
backend so its achieved runtime is measured on the same circuits as our
explicit statevector/MPS backends. The ML selector must beat THIS, not just
our hand-written rule.

ponytail: runs in shots mode on purpose. Forcing save_statevector() would pin
Aer to a statevector method and it could never pick MPS — defeating the point.
No fidelity is computed (sampling mode); fidelity stays null.
"""

import time

from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from backend.abstract import QuantumSimulatorBackend, SimulationResult
from backend.environment import collect_environment_metadata
from backend.config import AerConfig
from backend.gpu_poll import GpuMemoryPoller


class AerAutomaticBackend(QuantumSimulatorBackend):
    """Backend wrapping Aer's automatic method selection (baseline)."""

    def __init__(self, shots: int = 1024) -> None:
        self.config = AerConfig()
        self._shots = shots
        self._simulator = AerSimulator(method="automatic")
        self._simulator.set_options(**self.config.to_aer_options())

    @property
    def name(self) -> str:
        return "aer_automatic"

    @property
    def supports_statevector(self) -> bool:
        return False  # sampling mode — no full statevector returned

    @property
    def supports_fidelity(self) -> bool:
        return False

    def run(self, circuit: QuantumCircuit, shots: int = 0) -> SimulationResult:
        env = collect_environment_metadata()
        shots = shots or self._shots

        # Automatic selection only kicks in with something to measure. Add a
        # full measurement if the circuit has no classical readout of its own.
        circ = circuit
        if not circuit.cregs:
            circ = circuit.copy()
            circ.measure_all()

        compilation_start = time.perf_counter()
        try:
            transpiled = transpile(circ, self._simulator)
        except Exception as exc:
            return SimulationResult(
                schema_version="0.1.0", backend_name=self.name,
                circuit_name=circuit.name or "unnamed", n_qubits=circuit.num_qubits,
                depth=circuit.depth(),
                compilation_time_seconds=time.perf_counter() - compilation_start,
                success=False, error_message=f"Compilation error: {exc}", environment=env,
            )
        compilation_time = time.perf_counter() - compilation_start

        with GpuMemoryPoller() as poller:
            try:
                exec_start = time.perf_counter()
                result = self._simulator.run(transpiled, shots=shots).result()
                # Aer returns success=False (not an exception) when the backend
                # fails — e.g. a GPU kernel missing for this compute capability.
                # Promote it so it's recorded as a failed run, not a fake OK.
                if not result.success:
                    status = result.results[0].status if result.results else result.status
                    raise RuntimeError(status)
                execution_time = time.perf_counter() - exec_start
            except MemoryError as mem_err:
                return SimulationResult(
                    schema_version="0.1.0", backend_name=self.name,
                    circuit_name=circuit.name or "unnamed", n_qubits=circuit.num_qubits,
                    depth=circuit.depth(), compilation_time_seconds=compilation_time,
                    peak_gpu_memory_mb=float(poller.peak_mb), success=False,
                    error_message=f"Out of memory: {mem_err}", environment=env,
                )
            except Exception as exc:
                return SimulationResult(
                    schema_version="0.1.0", backend_name=self.name,
                    circuit_name=circuit.name or "unnamed", n_qubits=circuit.num_qubits,
                    depth=circuit.depth(), compilation_time_seconds=compilation_time,
                    peak_gpu_memory_mb=float(poller.peak_mb), success=False,
                    error_message=f"Execution error: {exc}", environment=env,
                )

        return SimulationResult(
            schema_version="0.1.0", backend_name=self.name,
            circuit_name=circuit.name or "unnamed", n_qubits=circuit.num_qubits,
            depth=circuit.depth(), compilation_time_seconds=compilation_time,
            execution_time_seconds=execution_time,
            total_time_seconds=compilation_time + execution_time,
            peak_gpu_memory_mb=float(poller.peak_mb),
            fidelity=None, fidelity_method=None,
            success=True, error_message=None,
            environment=env,
        )
