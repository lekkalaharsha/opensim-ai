"""Background GPU memory polling for benchmark backends.

Peak GPU memory is observed, never estimated (project rule): a daemon thread
samples ``pynvml`` at a fixed interval while a simulation runs. Exposed as a
context manager so backends can wrap just the execution region:

    with GpuMemoryPoller() as poller:
        run_the_simulation()
    peak_mb = poller.peak_mb

If ``pynvml`` is unavailable or no GPU is present, polling silently yields a
peak of 0 rather than raising.
"""

import os
import time
import threading


class GpuMemoryPoller:
    """Context manager that records peak GPU memory (MB) during its scope."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        """Initialize the poller.

        Args:
            interval_seconds: Sampling period in seconds (default 100 ms).
        """
        self.interval_seconds = interval_seconds
        self._peak_mb = 0
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    @property
    def peak_mb(self) -> int:
        """Maximum observed GPU memory usage in megabytes (0 if unobserved)."""
        return self._peak_mb

    def _poll(self) -> None:
        """Sample GPU memory until stopped; swallow errors if pynvml is absent."""
        try:
            # Respect CUDA_VISIBLE_DEVICES to poll the correct GPU.
            device_index_str = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            try:
                device_index = int(device_index_str)
            except (ValueError, IndexError):
                device_index = 0

            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            while not self._stop.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mb = int(info.used) // (1024 * 1024)
                if used_mb > self._peak_mb:
                    self._peak_mb = used_mb
                time.sleep(self.interval_seconds)
            pynvml.nvmlShutdown()
        except Exception:
            pass  # No GPU / pynvml unavailable: peak stays 0.

    def __enter__(self) -> "GpuMemoryPoller":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
