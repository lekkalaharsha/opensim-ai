# tests/test_integration.py
"""Milestone 0 integration test – 15-point gate.

Validates the full pipeline: circuit generation → backend execution → result JSON.
Must pass before advancing to Phase 1.
"""

import json
import os
from pathlib import Path

import pytest

from benchmark.circuit_library.ghz import generate_ghz_circuit
from backend.aer_statevector import AerStatevectorBackend
from benchmark.runner import run_single_benchmark


def test_milestone_0_integration():
    """All 15 gate conditions for Milestone 0."""
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    result = run_single_benchmark(
        generate_ghz_circuit(n_qubits=5, depth=1, seed=42),
        AerStatevectorBackend(),
        output_dir=output_dir,
    )

    # 1. No exception
    assert result is not None, "Result is None"

    # 2. Schema version
    assert result.schema_version == "0.1.0", f"Expected 0.1.0, got {result.schema_version}"

    # 3. Backend name
    assert result.backend_name == "aer_statevector", f"Expected aer_statevector, got {result.backend_name}"

    # 4. Qubit count
    assert result.n_qubits == 5, f"Expected 5 qubits, got {result.n_qubits}"

    # 5. Success
    assert result.success is True, "Simulation failed"

    # 6. Fidelity
    assert result.fidelity is not None, "Fidelity is None"
    assert result.fidelity > 0.99, f"Fidelity too low: {result.fidelity}"

    # 7. Fidelity method
    assert result.fidelity_method == "exact", f"Expected 'exact', got {result.fidelity_method}"

    # 8. Compilation time > 0
    assert result.compilation_time_seconds > 0, "Compilation time not recorded"

    # 9. Execution time > 0
    assert result.execution_time_seconds > 0, "Execution time not recorded"

    # 10. Total time = compile + execute
    expected_total = result.compilation_time_seconds + result.execution_time_seconds
    assert abs(result.total_time_seconds - expected_total) < 0.001, (
        f"Total time mismatch: {result.total_time_seconds} vs {expected_total}"
    )

    # 11. Environment metadata present
    assert result.environment is not None, "Environment metadata missing"
    env = result.environment
    assert env.python_version != "", "Python version empty"
    assert env.qiskit_version != "", "Qiskit version empty"

    # 12. GPU name (may be CPU if no GPU, but should be non-empty)
    assert env.gpu_name != "", "GPU name empty"

    # 13. JSON output file exists
    safe_name = result.circuit_name.replace(" ", "_").replace("/", "_")
    json_path = Path(output_dir) / f"{safe_name}_{result.backend_name}.json"
    assert json_path.exists(), f"JSON file not found: {json_path}"

    # 14. JSON is valid
    with open(json_path) as f:
        data = json.load(f)
    assert data["schema_version"] == "0.1.0", "Wrong schema_version in JSON"

    # 15. Environment metadata in JSON
    json_env = data.get("environment", {})
    assert json_env.get("python_version") != "", "Python version missing in JSON"

    print("✅ PASSED: Milestone 0")
    print(f"   Fidelity: {result.fidelity:.6f}")
    print(f"   Time: {result.total_time_seconds:.4f}s")
    print(f"   GPU Memory: {result.peak_gpu_memory_mb:.1f}MB")
    print(f"   GPU: {env.gpu_name}")