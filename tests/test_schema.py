# tests/test_schema.py
"""Tests for JSON schema validation."""

import json
import tempfile
from pathlib import Path
from benchmark.schema import validate_record, validate_file, validate_directory


def valid_record():
    return {
        "schema_version": "0.1.0",
        "backend_name": "aer_statevector",
        "circuit_name": "test",
        "n_qubits": 2,
        "depth": 1,
        "compilation_time_seconds": 0.1,
        "execution_time_seconds": 0.2,
        "total_time_seconds": 0.3,
        "peak_gpu_memory_mb": 100,
        "success": True,
        "environment": {
            "python_version": "3.11",
            "qiskit_version": "1.0",
            "qiskit_aer_version": "0.14",
            "gpu_name": "T4",
            "gpu_memory_mb": 15000,
            "cuda_version": "12.2",
            "host_platform": "Linux",
        },
    }


def test_validate_valid_record():
    errors = validate_record(valid_record())
    assert errors == []


def test_validate_missing_key():
    rec = valid_record()
    del rec["schema_version"]
    errors = validate_record(rec)
    assert any("schema_version" in e for e in errors)


def test_validate_wrong_schema():
    rec = valid_record()
    rec["schema_version"] = "0.0.9"
    errors = validate_record(rec)
    assert any("Unsupported" in e for e in errors)


def test_validate_missing_environment():
    rec = valid_record()
    del rec["environment"]
    errors = validate_record(rec)
    assert any("environment" in e for e in errors)


def test_validate_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.json"
        with open(path, "w") as f:
            json.dump(valid_record(), f)
        errors = validate_file(str(path))
        assert errors == []


def test_validate_directory():
    with tempfile.TemporaryDirectory() as d:
        # Write one valid, one invalid
        p1 = Path(d) / "valid.json"
        with open(p1, "w") as f:
            json.dump(valid_record(), f)
        p2 = Path(d) / "invalid.json"
        with open(p2, "w") as f:
            json.dump({"bad": "data"}, f)
        results = validate_directory(d)
        assert "valid.json" in results
        assert results["valid.json"] == []
        assert len(results["invalid.json"]) > 0