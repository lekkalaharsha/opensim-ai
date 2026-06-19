# benchmark/runner.py
"""Benchmark runner for executing circuits on quantum simulator backends.

Provides `run_single_benchmark` for one-off runs and `run_sweep` for
batch processing based on a YAML configuration.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from qiskit import QuantumCircuit

from backend.abstract import QuantumSimulatorBackend, SimulationResult
from benchmark.circuit_library.ghz import generate_ghz_circuit
from benchmark.circuit_library.qft import generate_qft_circuit
from benchmark.circuit_library.random import generate_random_circuit
from benchmark.circuit_library.qaoa import generate_qaoa_circuit
from benchmark.circuit_library.variational import generate_variational_circuit
from benchmark.circuit_library.clifford import generate_clifford_circuit


# Registry of circuit generators accessible by name
CIRCUIT_GENERATORS = {
    "ghz": generate_ghz_circuit,
    "qft": generate_qft_circuit,
    "random": generate_random_circuit,
    "qaoa": generate_qaoa_circuit,
    "variational": generate_variational_circuit,
    "clifford": generate_clifford_circuit,
}


def run_single_benchmark(
    circuit: QuantumCircuit,
    backend: QuantumSimulatorBackend,
    output_dir: str = "data/raw",
) -> SimulationResult:
    """Execute a single benchmark and save the result as JSON.

    Args:
        circuit: The quantum circuit to simulate.
        backend: An instance of a QuantumSimulatorBackend.
        output_dir: Directory to save the JSON output file.

    Returns:
        SimulationResult with all metrics populated.
    """
    # Run the simulation with exception safety
    try:
        result = backend.run(circuit)
    except MemoryError as mem_err:
        result = SimulationResult(
            schema_version="0.1.0",
            backend_name=backend.name,
            circuit_name=circuit.name or "unnamed",
            n_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            success=False,
            error_message=f"Out of memory: {mem_err}",
        )
    except Exception as exc:
        result = SimulationResult(
            schema_version="0.1.0",
            backend_name=backend.name,
            circuit_name=circuit.name or "unnamed",
            n_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            success=False,
            error_message=f"Execution error: {exc}",
        )

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Serialize to JSON-friendly dict
    result_dict = {
        "schema_version": result.schema_version,
        "backend_name": result.backend_name,
        "circuit_name": result.circuit_name,
        "n_qubits": result.n_qubits,
        "depth": result.depth,
        "compilation_time_seconds": result.compilation_time_seconds,
        "execution_time_seconds": result.execution_time_seconds,
        "total_time_seconds": result.total_time_seconds,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        "fidelity": result.fidelity,
        "fidelity_method": result.fidelity_method,
        "entropy": result.entropy,
        "entropy_method": result.entropy_method,
        "success": result.success,
        "error_message": result.error_message,
        "environment": {
            "python_version": result.environment.python_version if result.environment else "",
            "qiskit_version": result.environment.qiskit_version if result.environment else "",
            "qiskit_aer_version": result.environment.qiskit_aer_version if result.environment else "",
            "gpu_name": result.environment.gpu_name if result.environment else "",
            "gpu_memory_mb": result.environment.gpu_memory_mb if result.environment else 0,
            "cuda_version": result.environment.cuda_version if result.environment else "",
            "host_platform": result.environment.host_platform if result.environment else "",
        },
        "circuit_fingerprint": result.circuit_fingerprint,
    }

    # File name includes circuit name and backend to avoid collisions
    safe_circuit_name = result.circuit_name.replace(" ", "_").replace("/", "_")
    file_name = f"{safe_circuit_name}_{result.backend_name}.json"
    file_path = output_path / file_name

    with open(file_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    return result


def run_sweep(config_path: str, output_dir: str = "data/raw") -> List[SimulationResult]:
    """Run a batch of benchmarks as defined in a YAML sweep configuration."""
    import yaml
    from backend.abstract import QuantumSimulatorBackend
    from backend.aer_statevector import AerStatevectorBackend

    # MPS backend may not be implemented yet; handle gracefully
    try:
        from backend.aer_mps import AerMPSBackend
        _HAS_MPS = True
    except ImportError:
        _HAS_MPS = False
        AerMPSBackend = None  # type: ignore

    # Use a union type via the ABC to satisfy type checker
    BACKEND_MAP: dict[str, type[QuantumSimulatorBackend]] = {
        "aer_statevector": AerStatevectorBackend,
    }
    if _HAS_MPS and AerMPSBackend is not None:
        BACKEND_MAP["aer_mps"] = AerMPSBackend

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = []
    seed_base = config.get("seed_base", 42)

    for circuit_name in config.get("circuits", []):
        generator = CIRCUIT_GENERATORS.get(circuit_name)
        if generator is None:
            print(f"Warning: Unknown circuit type '{circuit_name}', skipping.")
            continue

        for n_qubits in config.get("qubits", []):
            for depth in config.get("depths", []):
                for backend_name in config.get("backends", []):
                    backend_cls = BACKEND_MAP.get(backend_name)
                    if backend_cls is None:
                        print(f"Warning: Unknown backend '{backend_name}', skipping.")
                        continue
                    for rep in range(config.get("repetitions", 1)):
                        seed = seed_base + rep
                        try:
                            circuit = generator(n_qubits, depth, seed)
                        except Exception as e:
                            print(f"Error generating {circuit_name} (n={n_qubits}, d={depth}): {e}")
                            continue
                        backend = backend_cls()
                        print(f"Running: {circuit.name} on {backend_name} (rep {rep})...")
                        result = run_single_benchmark(circuit, backend, output_dir)
                        results.append(result)
                        if result.success:
                            print(f"  -> OK: {result.total_time_seconds:.3f}s, {result.peak_gpu_memory_mb:.0f}MB")
                        else:
                            print(f"  -> FAIL: {result.error_message}")
    return results