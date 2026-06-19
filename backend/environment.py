"""Environment metadata collection for reproducibility.

Every benchmark record must include detailed information about the
hardware and software environment in which it was executed. This module
provides a single function that gathers all relevant metadata.
"""

import platform
import sys
import os
import subprocess
from typing import Tuple

from backend.abstract import EnvironmentMetadata


def _detect_gpu() -> Tuple[str, int]:
    """Detect GPU name and total memory using pynvml.

    Returns:
        Tuple of (gpu_name, memory_mb). Returns ("CPU", 0) if no GPU is found
        or if pynvml is not available.
    """
    try:
        # Respect CUDA_VISIBLE_DEVICES to report on the correct GPU
        device_index_str = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(',')[0].strip()
        try:
            device_index = int(device_index_str)
        except (ValueError, IndexError):
            device_index = 0

        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_mb = int(memory_info.total) // (1024 * 1024)
        pynvml.nvmlShutdown()
        return name, memory_mb
    except Exception:
        return "CPU", 0


def _detect_cuda_version() -> str:
    """Attempt to detect CUDA version via nvidia-smi.

    Returns:
        CUDA version string, or empty string if not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "CUDA Version" in line:
                    # Line format: "| CUDA Version: X.Y |"
                    parts = line.split("CUDA Version:")[-1].strip().split()
                    if parts:
                        return parts[0]
    except Exception:
        pass
    return ""


def collect_environment_metadata() -> EnvironmentMetadata:
    """Collect hardware and software environment information.

    This function gathers:
    - Python version
    - Qiskit and Qiskit Aer versions
    - GPU name and total memory (via pynvml, if available)
    - CUDA version (via nvidia-smi, if available)
    - Host platform string

    Returns:
        An EnvironmentMetadata instance populated with all available data.
    """
    # Python version
    python_version = platform.python_version()

    # Qiskit versions (safe import because they may not be installed)
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        # Fallback for older Python
        from importlib_metadata import version, PackageNotFoundError

    try:
        qiskit_version = version("qiskit")
    except PackageNotFoundError:
        qiskit_version = "NOT INSTALLED"

    try:
        qiskit_aer_version = version("qiskit-aer")
    except PackageNotFoundError:
        qiskit_aer_version = "NOT INSTALLED"

    # GPU information
    gpu_name, gpu_memory_mb = _detect_gpu()
    cuda_version = _detect_cuda_version() if gpu_name != "CPU" else ""

    # Host platform
    host_platform = platform.platform()

    return EnvironmentMetadata(
        python_version=python_version,
        qiskit_version=qiskit_version,
        qiskit_aer_version=qiskit_aer_version,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        cuda_version=cuda_version,
        host_platform=host_platform,
    )