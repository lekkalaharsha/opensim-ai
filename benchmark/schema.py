# benchmark/schema.py
"""JSON schema validation for benchmark output files.

Ensures every record written to data/raw/ contains the mandatory fields and
types expected by downstream processing.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


REQUIRED_TOP_KEYS = [
    "schema_version",
    "backend_name",
    "circuit_name",
    "n_qubits",
    "depth",
    "compilation_time_seconds",
    "execution_time_seconds",
    "total_time_seconds",
    "peak_gpu_memory_mb",
    "success",
]

ENVIRONMENT_KEYS = [
    "python_version",
    "qiskit_version",
    "qiskit_aer_version",
    "gpu_name",
    "gpu_memory_mb",
    "cuda_version",
    "host_platform",
]


def validate_record(record: Dict[str, Any]) -> List[str]:
    """Validate a single benchmark record dictionary.

    Args:
        record: The parsed JSON object of a benchmark result.

    Returns:
        List of validation error messages. Empty list means valid.
    """
    errors = []

    # Required top-level keys
    for key in REQUIRED_TOP_KEYS:
        if key not in record:
            errors.append(f"Missing required key: '{key}'")

    if "schema_version" in record and record["schema_version"] != "0.1.0":
        errors.append(f"Unsupported schema_version: {record['schema_version']}")

    # success must be boolean
    if "success" in record and not isinstance(record["success"], bool):
        errors.append("'success' must be boolean")

    # timing values must be non-negative numbers
    for time_key in ["compilation_time_seconds", "execution_time_seconds", "total_time_seconds"]:
        if time_key in record:
            val = record[time_key]
            if not isinstance(val, (int, float)) or val < 0:
                errors.append(f"'{time_key}' must be a non-negative number")

    # environment block
    env = record.get("environment")
    if env is None:
        errors.append("Missing 'environment' block")
    elif isinstance(env, dict):
        for key in ENVIRONMENT_KEYS:
            if key not in env:
                errors.append(f"Missing environment key: '{key}'")
    else:
        errors.append("'environment' must be a dictionary")

    # fidelity and entropy are optional but must be numbers or null
    for opt_key in ["fidelity", "entropy"]:
        if opt_key in record and record[opt_key] is not None:
            if not isinstance(record[opt_key], (int, float)):
                errors.append(f"'{opt_key}' must be a number or null")

    return errors


def validate_file(file_path: str) -> List[str]:
    """Validate a benchmark JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of validation error messages. Empty list means valid.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return [f"File error: {e}"]

    if not isinstance(data, dict):
        return ["Top-level JSON must be a dictionary"]

    return validate_record(data)


def validate_directory(raw_dir: str) -> Dict[str, List[str]]:
    """Validate all JSON files in a directory.

    Args:
        raw_dir: Path to directory containing benchmark JSON files.

    Returns:
        Dictionary mapping filename to list of validation errors.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return {"error": ["Directory not found"]}

    results = {}
    for json_file in sorted(raw_path.glob("*.json")):
        errors = validate_file(str(json_file))
        if errors:
            results[json_file.name] = errors
        else:
            results[json_file.name] = []

    return results