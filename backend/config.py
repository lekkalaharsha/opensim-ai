"""GPU and fusion configuration for Qiskit Aer backends.

Provides a central place for tuning Aer's simulation parameters, such as
GPU device selection, memory limits, fusion thresholds, and precision.
Configuration can be overridden via environment variables.
"""

import os


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _gpu_available() -> bool:
    """Check if the installed Qiskit Aer build can run GPU simulations.

    A physical GPU being present (detectable via pynvml) is not sufficient:
    the CPU-only ``qiskit-aer`` wheel cannot use it. We therefore ask Aer
    directly which devices it supports, which only reports ``GPU`` when the
    ``qiskit-aer-gpu`` build is installed AND a usable CUDA device exists.
    """
    try:
        from qiskit_aer import AerSimulator
        return "GPU" in AerSimulator().available_devices()
    except Exception:
        return False


class AerConfig:
    """Configuration for Qiskit Aer GPU backends.

    Attributes:
        device: Target device ('GPU' or 'CPU').
        gpu_memory_mb: Maximum GPU memory to use (MB). If 0, use all available.
        precision: Floating-point precision ('double' or 'single').
        fusion_threshold: Max qubits for gate fusion (higher = more fusion).
        fusion_enabled: Whether to enable gate fusion.
        batched_contraction: Whether to use batched tensor contraction.
        max_parallel_threads: Max CPU threads for hybrid operations.
        cuda_visible_devices: CUDA device indices (e.g., '0,1').
    """

    def __init__(self):
        default_device = "GPU" if _gpu_available() else "CPU"
        self.device = os.getenv("OPENQSIM_DEVICE", default_device)

        # GPU memory limit (MB)
        self.gpu_memory_mb = _env_int("OPENQSIM_GPU_MEMORY_MB", 0)

        # Single precision halves GPU memory usage and speeds up simulation;
        # fidelity remains >0.999 for typical circuits. Use double only when
        # OPENQSIM_PRECISION=double is set explicitly.
        self.precision = os.getenv("OPENQSIM_PRECISION", "single")

        # Fusion settings
        self.fusion_enabled = _env_bool("OPENQSIM_FUSION_ENABLED", True)
        self.fusion_threshold = _env_int("OPENQSIM_FUSION_THRESHOLD", 5)

        # Batched contraction for tensor networks
        self.batched_contraction = _env_bool("OPENQSIM_BATCHED_CONTRACTION", True)

        # Parallelism — 8 threads covers P100/T4 CPU cores; 0 = Aer default
        self.max_parallel_threads = _env_int("OPENQSIM_MAX_THREADS", 8)

        # CUDA device visibility
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    def to_aer_options(self) -> dict:
        """Convert configuration to Qiskit Aer backend options dict.

        Returns:
            Dictionary suitable for passing to AerSimulator.set_options().
        """
        options = {
            "device": self.device,
            "precision": self.precision,
            "fusion_enable": self.fusion_enabled,
            "fusion_threshold": self.fusion_threshold,
        }
        if self.gpu_memory_mb > 0:
            options["gpu_memory_mb"] = self.gpu_memory_mb
        if self.max_parallel_threads > 0:
            options["max_parallel_threads"] = self.max_parallel_threads
        return options

    def to_dict(self) -> dict:
        """Return all configuration values as a dictionary.

        Useful for logging or embedding in benchmark metadata.
        """
        return {
            "device": self.device,
            "gpu_memory_mb": self.gpu_memory_mb,
            "precision": self.precision,
            "fusion_enabled": self.fusion_enabled,
            "fusion_threshold": self.fusion_threshold,
            "batched_contraction": self.batched_contraction,
            "max_parallel_threads": self.max_parallel_threads,
            "cuda_visible_devices": self.cuda_visible_devices,
        }


# Global singleton for convenience
config = AerConfig()