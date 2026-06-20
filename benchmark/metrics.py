# benchmark/metrics.py
"""Metrics computation utilities for benchmark results.

Functions to analyze and compare SimulationResult objects, including
regret calculation, best backend selection, and summary statistics.
"""

from typing import List, Dict, Optional
import numpy as np
from backend.abstract import SimulationResult


def optimal_backend(results: List[SimulationResult]) -> str:
    """Return the backend name that achieved the minimum total time.

    Args:
        results: List of SimulationResult objects for the same circuit.

    Returns:
        Backend name with the lowest total_time_seconds.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return "unknown"
    best = min(successful, key=lambda r: r.total_time_seconds)
    return best.backend_name


def compute_regret(results: List[SimulationResult], backend_name: str) -> Optional[float]:
    """Compute the regret of a specific backend relative to the optimal.

    Regret = total_time(backend) / total_time(optimal_backend).

    Args:
        results: All results for a given circuit (should include the backend in question).
        backend_name: The backend to evaluate.

    Returns:
        Regret value >= 1.0, or None if the backend result or optimal is unavailable.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return None
    target = next((r for r in successful if r.backend_name == backend_name), None)
    if target is None:
        return None
    best_time = min(r.total_time_seconds for r in successful)
    if best_time <= 0:
        return None
    return target.total_time_seconds / best_time


def average_regret(results_by_circuit: Dict[str, List[SimulationResult]], backend_name: str) -> float:
    """Compute average regret of a backend across multiple circuits.

    Args:
        results_by_circuit: Mapping of circuit_id -> list of SimulationResult.
        backend_name: The backend to evaluate.

    Returns:
        Average regret across circuits where the backend completed successfully.
    """
    regrets = []
    for circuit_id, circuit_results in results_by_circuit.items():
        regret = compute_regret(circuit_results, backend_name)
        if regret is not None:
            regrets.append(regret)
    if not regrets:
        return float("inf")
    return float(np.mean(regrets))


def summarize_results(results: List[SimulationResult]) -> dict:
    """Generate summary statistics from a list of benchmark results.

    Args:
        results: List of SimulationResult objects.

    Returns:
        Dictionary with counts, success rate, timing stats, memory stats.
    """
    total = len(results)
    successful = [r for r in results if r.success]
    failed = total - len(successful)

    times = [r.total_time_seconds for r in successful] if successful else [0]
    memories = [r.peak_gpu_memory_mb for r in successful if r.peak_gpu_memory_mb > 0] or [0]

    return {
        "total_runs": total,
        "successful": len(successful),
        "failed": failed,
        "avg_time_s": float(np.mean(times)),
        "median_time_s": float(np.median(times)),
        "min_time_s": float(np.min(times)),
        "max_time_s": float(np.max(times)),
        "avg_peak_gpu_memory_mb": float(np.mean(memories)),
        "max_peak_gpu_memory_mb": float(np.max(memories)),
    }


def determine_winner(
    sv_result: SimulationResult,
    mps_result: SimulationResult,
    fidelity_threshold: float = 0.999,
) -> str:
    """Determine which backend won for a given circuit.

    A backend only qualifies if it succeeded AND its fidelity (when measured)
    is >= fidelity_threshold.  If only one qualifies, it wins outright.  If
    neither qualifies, returns "undecidable".  If both qualify, the faster one
    wins; equal times favour the statevector backend.

    Args:
        sv_result: SimulationResult from the statevector backend.
        mps_result: SimulationResult from the MPS backend.
        fidelity_threshold: Minimum acceptable fidelity (default 0.999).

    Returns:
        Backend name string: 'aer_statevector', 'aer_mps', or 'undecidable'.
    """
    sv_fid = sv_result.fidelity
    mps_fid = mps_result.fidelity

    sv_ok = sv_result.success and (sv_fid is None or sv_fid >= fidelity_threshold)
    mps_ok = mps_result.success and (mps_fid is None or mps_fid >= fidelity_threshold)

    if not sv_ok and not mps_ok:
        return "undecidable"
    if not sv_ok:
        return mps_result.backend_name
    if not mps_ok:
        return sv_result.backend_name

    return (
        sv_result.backend_name
        if sv_result.total_time_seconds <= mps_result.total_time_seconds
        else mps_result.backend_name
    )


def backend_comparison_table(results: List[SimulationResult]) -> List[Dict]:
    """Create a comparison table across backends for a set of results.

    Args:
        results: List of results (can be across different circuits/backends).

    Returns:
        List of dictionaries with per-backend aggregated stats.
    """
    from collections import defaultdict
    backend_groups = defaultdict(list)
    for r in results:
        backend_groups[r.backend_name].append(r)

    table = []
    for bname, group in backend_groups.items():
        summary = summarize_results(group)
        summary["backend"] = bname
        table.append(summary)
    return table