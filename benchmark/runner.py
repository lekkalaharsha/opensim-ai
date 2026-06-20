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
from benchmark.schema import validate_record
from benchmark.circuit_library.clifford import generate_clifford_circuit
from benchmark.circuit_fingerprint import extract_circuit_fingerprint
from benchmark.entanglement import contiguous_entropy_features


# Registry of circuit generators accessible by name
CIRCUIT_GENERATORS = {
    "ghz": generate_ghz_circuit,
    "qft": generate_qft_circuit,
    "random": generate_random_circuit,
    "qaoa": generate_qaoa_circuit,
    "variational": generate_variational_circuit,
    "clifford": generate_clifford_circuit,
}


def load_circuit_generator(name: str):
    """Return the circuit generator function registered under ``name``.

    Args:
        name: Circuit family identifier (e.g., 'ghz', 'qft').

    Returns:
        The generator callable, invoked as ``generator(n_qubits, depth, seed)``.

    Raises:
        ValueError: If no generator is registered under ``name``.
    """
    generator = CIRCUIT_GENERATORS.get(name)
    if generator is None:
        raise ValueError(
            f"Unknown circuit type: {name!r}. Available: {sorted(CIRCUIT_GENERATORS)}"
        )
    return generator


def load_backend(name: str) -> QuantumSimulatorBackend:
    """Instantiate a simulator backend by its registered name.

    The MPS backend is registered only when it can be imported, so callers on
    environments without it still get a clear ``ValueError`` rather than an
    ``ImportError`` at module load.

    Args:
        name: Backend identifier (e.g., 'aer_statevector', 'aer_mps').

    Returns:
        A new backend instance implementing ``QuantumSimulatorBackend``.

    Raises:
        ValueError: If no backend is registered under ``name``.
    """
    from backend.aer_statevector import AerStatevectorBackend

    backend_map: dict[str, type[QuantumSimulatorBackend]] = {
        "aer_statevector": AerStatevectorBackend,
    }
    try:
        from backend.aer_mps import AerMPSBackend
        backend_map["aer_mps"] = AerMPSBackend
    except ImportError:
        pass

    backend_cls = backend_map.get(name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown backend: {name!r}. Available: {sorted(backend_map)}"
        )
    return backend_cls()


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

    # Populate the ML feature fingerprint from the circuit's structure if the
    # backend did not already supply one. This is a property of the circuit,
    # so it is recorded even for failed runs.
    if not result.circuit_fingerprint:
        try:
            result.circuit_fingerprint = extract_circuit_fingerprint(circuit)
        except Exception:
            pass

    # Populate entanglement entropy from the statevector when the backend
    # returned one and did not compute entropy itself. We use the contiguous-cut
    # definition so statevector and MPS entropy are directly comparable (the MPS
    # backend reads the same cuts from its Schmidt coefficients). The MPS backend
    # sets its own entropy, so the guard leaves that untouched. This lives in
    # benchmark/ (not the backend) because backend/ must never import from
    # benchmark/ (no-circular-imports rule).
    if result.entropy is None and result.statevector is not None:
        try:
            e_max, e_mid, e_avg = contiguous_entropy_features(
                result.statevector, result.n_qubits
            )
            result.entropy = e_max
            result.entropy_middle = e_mid
            result.entropy_avg = e_avg
            result.entropy_method = "exact"
        except Exception:
            pass

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
        "entropy_middle": result.entropy_middle,
        "entropy_avg": result.entropy_avg,
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

    # Validate the result before saving
    # Validate the result before saving (validate_record returns list of issues)
    validation_errors = validate_record(result_dict)
    if validation_errors:
        result_dict["validation_errors"] = validation_errors
        print(f"WARN Validation warnings for {result.circuit_name}: {validation_errors}")
        # Still save, but with warnings logged
        
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