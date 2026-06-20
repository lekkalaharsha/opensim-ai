"""Kaggle environment validation.

Before any sweep starts we verify the GPU is available, versions match,
and all dependencies are installed.
"""

import sys
import subprocess
from dataclasses import dataclass, field
from typing import Tuple
import importlib.metadata


@dataclass
class KaggleEnvironmentReport:
    """Result of the Kaggle environment check."""
    is_valid: bool
    gpu_name: str
    gpu_memory_mb: int
    python_version: str
    qiskit_version: str
    qiskit_aer_version: str
    cuda_available: bool
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def detect_gpu() -> Tuple[str, int]:
    """Return (gpu_name, memory_mb).  ('CPU', 0) if no GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_mb = mem.total // (1024 * 1024)
        pynvml.nvmlShutdown()
        return name, mem_mb
    except Exception:
        return "CPU", 0


def get_package_version(pkg: str) -> str:
    """Safely return installed version of pkg, or 'NOT INSTALLED'."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def validate_kaggle_environment() -> KaggleEnvironmentReport:
    """Check the Kaggle environment is ready for benchmarking.

    Returns a report with a boolean 'is_valid' and lists of issues/warnings.
    """
    issues, warnings = [], []

    gpu_name, gpu_mem = detect_gpu()
    if gpu_name == "CPU":
        issues.append("No GPU detected. Enable GPU accelerator in notebook settings.")
    elif "T4" not in gpu_name and "P100" not in gpu_name:
        warnings.append(f"Unexpected GPU: {gpu_name}. Expected Tesla T4 or P100.")
    if gpu_mem < 14000 and gpu_name != "CPU":
        warnings.append(f"Low GPU memory: {gpu_mem} MB. T4 has ~15 GB.")

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 10):
        issues.append(f"Python {py_ver} too old. 3.10+ required.")

    qiskit_ver = get_package_version("qiskit")
    if qiskit_ver == "NOT INSTALLED":
        issues.append("Qiskit not installed.")
    elif qiskit_ver < "1.0":
        issues.append(f"Qiskit {qiskit_ver} too old. Need >=1.0.0.")

    aer_ver = get_package_version("qiskit-aer")
    if aer_ver == "NOT INSTALLED":
        issues.append("qiskit-aer not installed.")

    cuda_ok = False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        cuda_ok = True
    except Exception:
        warnings.append("nvidia-smi not found; CUDA may be unavailable.")

    return KaggleEnvironmentReport(
        is_valid=len(issues) == 0,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        python_version=py_ver,
        qiskit_version=qiskit_ver,
        qiskit_aer_version=aer_ver,
        cuda_available=cuda_ok,
        issues=issues,
        warnings=warnings,
    )