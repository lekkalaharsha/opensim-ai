# tests/test_runner.py
"""Tests for the benchmark runner."""

import json
import tempfile
from pathlib import Path

from benchmark.runner import run_single_benchmark
from benchmark.circuit_library.ghz import generate_ghz_circuit
from backend.aer_statevector import AerStatevectorBackend


def test_run_single_benchmark_writes_json():
    """Runner should write a valid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        circuit = generate_ghz_circuit(3, seed=0)
        result = run_single_benchmark(circuit, AerStatevectorBackend(), output_dir=tmpdir)

        assert result.success is True
        # Check file exists
        safe_name = result.circuit_name.replace(" ", "_").replace("/", "_")
        json_path = Path(tmpdir) / f"{safe_name}_{result.backend_name}.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["schema_version"] == "0.1.0"
        assert data["n_qubits"] == 3
        assert data["success"] is True


def test_runner_handles_oom_gracefully():
    """Runner should not crash when memory is exhausted (test with large circuit on CPU)."""
    # We simulate OOM by using a backend that raises MemoryError
    import numpy as np
    from backend.abstract import QuantumSimulatorBackend, SimulationResult

    class OOMBackend(QuantumSimulatorBackend):
        @property
        def name(self): return "oom_test"
        @property
        def supports_statevector(self): return False
        @property
        def supports_fidelity(self): return False
        def run(self, circuit, shots=0):
            raise MemoryError("Simulated OOM")

    circuit = generate_ghz_circuit(2, seed=0)
    result = run_single_benchmark(circuit, OOMBackend(), output_dir=tempfile.mkdtemp())
    assert result.success is False
    assert "OOM" in result.error_message.upper()