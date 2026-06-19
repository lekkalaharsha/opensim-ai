"""Qiskit Aer statevector backend wrapper.

Implements the QuantumSimulatorBackend ABC for Aer's statevector simulator.
Collects compilation time, execution time, peak GPU memory (via pynvml polling),
and exact fidelity.
"""

import time
import os
import threading
from typing import Optional

import numpy as np
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from backend.abstract import QuantumSimulatorBackend, SimulationResult
from backend.environment import collect_environment_metadata
from backend.config import AerConfig


class AerStatevectorBackend(QuantumSimulatorBackend):
    """Backend wrapping Qiskit Aer's statevector simulator.

    Runs on GPU if available (device='GPU') and records:
    - compilation (transpile) time
    - execution time
    - peak GPU memory via pynvml in a background thread
    - exact fidelity by comparing against an ideal statevector simulation
    """

    def __init__(self) -> None:
        """Initialize the Aer statevector backend with GPU acceleration."""
        self.config = AerConfig()
        self._simulator = AerSimulator(method="statevector")
        self._simulator.set_options(**self.config.to_aer_options())

    # ------------------------------------------------------------------
    # ABC properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "aer_statevector"

    @property
    def supports_statevector(self) -> bool:
        return True

    @property
    def supports_fidelity(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def run(self, circuit: QuantumCircuit, shots: int = 0) -> SimulationResult:
        """Execute a circuit on the Aer statevector simulator.

        Args:
            circuit: Qiskit QuantumCircuit to simulate.
            shots: Number of measurement shots (0 = return statevector).

        Returns:
            SimulationResult with all metrics populated.
        """
        # --- environment metadata (collected once per run) ---
        env = collect_environment_metadata()

        # --- GPU memory polling setup ---
        peak_memory = [0]                # list for in-place mutation by thread
        stop_polling = threading.Event()

        def poll_memory() -> None:
            """Poll GPU memory usage every 100 ms until stopped."""
            try:
                # Respect CUDA_VISIBLE_DEVICES to poll the correct GPU
                device_index_str = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(',')[0].strip()
                try:
                    device_index = int(device_index_str)
                except (ValueError, IndexError):
                    device_index = 0

                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                while not stop_polling.is_set():
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_mb = int(info.used) // (1024 * 1024)
                    if used_mb > peak_memory[0]:
                        peak_memory[0] = used_mb
                    time.sleep(0.1)
                pynvml.nvmlShutdown()
            except Exception:
                pass                        # silently ignore if pynvml unavailable

        memory_thread = threading.Thread(target=poll_memory, daemon=True)

        # --- compilation timing ---
        compilation_start = time.perf_counter()
        try:
            transpiled = transpile(circuit, self._simulator)
        except Exception as exc:
            # compilation failure → partial result
            compilation_time = time.perf_counter() - compilation_start
            return SimulationResult(
                schema_version="0.1.0",
                backend_name=self.name,
                circuit_name=circuit.name or "unnamed",
                n_qubits=circuit.num_qubits,
                depth=circuit.depth(),
                compilation_time_seconds=compilation_time,
                success=False,
                error_message=f"Compilation error: {exc}",
                environment=env,
            )
        compilation_time = time.perf_counter() - compilation_start

        # --- execution with memory polling ---
        try:
            memory_thread.start()
            exec_start = time.perf_counter()
            # Add save_statevector for Aer 0.17.x compatibility
            if shots == 0:
                transpiled = transpiled.copy()
                transpiled.save_statevector()
            job = self._simulator.run(transpiled, shots=shots)
            result = job.result()
            execution_time = time.perf_counter() - exec_start
        except MemoryError as mem_err:
            stop_polling.set()
            memory_thread.join(timeout=1)
            return SimulationResult(
                schema_version="0.1.0",
                backend_name=self.name,
                circuit_name=circuit.name or "unnamed",
                n_qubits=circuit.num_qubits,
                depth=circuit.depth(),
                compilation_time_seconds=compilation_time,
                success=False,
                error_message=f"Out of memory: {mem_err}",
                environment=env,
            )
        except Exception as exc:
            stop_polling.set()
            memory_thread.join(timeout=1)
            return SimulationResult(
                schema_version="0.1.0",
                backend_name=self.name,
                circuit_name=circuit.name or "unnamed",
                n_qubits=circuit.num_qubits,
                depth=circuit.depth(),
                compilation_time_seconds=compilation_time,
                success=False,
                error_message=f"Execution error: {exc}",
                environment=env,
            )
        finally:
            stop_polling.set()
            memory_thread.join(timeout=1)

        # --- statevector & fidelity ---
        statevector = None
        try:
            statevector = result.get_statevector() if shots == 0 else None
        except Exception:
            statevector = None

        fidelity = None
        fidelity_method = None

        if statevector is not None:
            # Compute exact fidelity by running the same circuit with a 'perfect'
            # statevector simulation (CPU, no noise) as reference.
            try:
                ref_sim = AerSimulator(method="statevector", device="CPU")   # force CPU reference
                ref_job = ref_sim.run(transpiled, shots=0)
                ref_sv = ref_job.result().get_statevector()
                if ref_sv is not None:
                    fidelity = float(
                        np.abs(np.dot(np.conj(ref_sv.data), statevector.data)) ** 2
                    )
                    fidelity_method = "exact"
            except Exception:
                fidelity = None
                fidelity_method = None

        return SimulationResult(
            schema_version="0.1.0",
            backend_name=self.name,
            circuit_name=circuit.name or "unnamed",
            n_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            compilation_time_seconds=compilation_time,
            execution_time_seconds=execution_time,
            total_time_seconds=compilation_time + execution_time,
            peak_gpu_memory_mb=float(peak_memory[0]),
            fidelity=fidelity,
            fidelity_method=fidelity_method,
            statevector=statevector,
            success=True,
            environment=env,
        )