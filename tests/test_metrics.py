# tests/test_metrics.py
"""Tests for the metrics module."""

from benchmark.metrics import (
    optimal_backend,
    compute_regret,
    average_regret,
    summarize_results,
)
from backend.abstract import SimulationResult


def make_result(backend, time, success=True):
    return SimulationResult(
        backend_name=backend,
        total_time_seconds=time,
        success=success,
    )


def test_optimal_backend():
    results = [
        make_result("aer_statevector", 10.0),
        make_result("aer_mps", 5.0),
        make_result("aer_statevector", 10.0),
    ]
    assert optimal_backend(results) == "aer_mps"


def test_optimal_backend_ignores_failures():
    results = [
        make_result("aer_statevector", 1.0),
        make_result("aer_mps", 0.5, success=False),
    ]
    assert optimal_backend(results) == "aer_statevector"


def test_compute_regret():
    results = [
        make_result("aer_statevector", 10.0),
        make_result("aer_mps", 5.0),
    ]
    regret = compute_regret(results, "aer_statevector")
    assert regret == 2.0


def test_average_regret():
    results_by_circuit = {
        "c1": [make_result("aer_statevector", 10.0), make_result("aer_mps", 5.0)],
        "c2": [make_result("aer_statevector", 8.0), make_result("aer_mps", 4.0)],
    }
    avg = average_regret(results_by_circuit, "aer_statevector")
    assert avg == 2.0


def test_summarize_results():
    results = [
        make_result("aer_statevector", 10.0),
        make_result("aer_mps", 5.0),
        make_result("aer_statevector", 15.0, success=False),
    ]
    summary = summarize_results(results)
    assert summary["total_runs"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["min_time_s"] == 5.0
    assert summary["max_time_s"] == 10.0