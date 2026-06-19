# tests/test_nim_advisor.py
"""Test for NVIDIA NIM LLM advisor (requires API key)."""

import os
import sys
import pytest

# Skip if no API key
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    pytest.skip("NVIDIA_API_KEY not set, skipping NIM advisor tests", allow_module_level=True)

from backend.llm_advisor import NIMBackendAdvisor, AdvisorRecommendation


@pytest.fixture
def advisor():
    return NIMBackendAdvisor(api_key=NVIDIA_API_KEY, model="meta/llama-3.1-8b-instruct")


def test_recommend_backend(advisor):
    """Advisor should return a valid recommendation."""
    fingerprint = {
        "qubits": 20,
        "depth": 40,
        "gate_counts": {"h": 20, "cx": 120},
        "interaction_graph": {"diameter": 5},
    }
    rec = advisor.recommend_backend(
        qasm='OPENQASM 2.0; include "qelib1.inc"; qreg q[20]; h q[0]; cx q[0], q[1];',
        fingerprint=fingerprint,
    )
    assert isinstance(rec, AdvisorRecommendation)
    assert rec.recommended_backend in ("aer_statevector", "aer_mps")
    assert 0.0 <= rec.confidence <= 1.0
    assert len(rec.reasoning) > 0
    assert rec.model_used == "meta/llama-3.1-8b-instruct"
    assert rec.latency_seconds > 0


def test_explain_result(advisor):
    """Advisor should explain a benchmark result without error."""
    from backend.abstract import SimulationResult, EnvironmentMetadata
    result = SimulationResult(
        circuit_name="ghz_5q",
        n_qubits=5,
        backend_name="aer_statevector",
        total_time_seconds=0.123,
        peak_gpu_memory_mb=512,
        success=True,
        fidelity=0.999,
        environment=EnvironmentMetadata(gpu_name="Tesla T4"),
    )
    explanation = advisor.explain_result(result)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_batch_analyze(advisor):
    """Batch analysis should return one recommendation per circuit."""
    circuits = [
        {
            "qasm": "...",
            "fingerprint": {"qubits": 10, "depth": 5, "gate_counts": {}, "interaction_graph": {}},
        },
        {
            "qasm": "...",
            "fingerprint": {"qubits": 25, "depth": 10, "gate_counts": {}, "interaction_graph": {}},
        },
    ]
    recs = advisor.batch_analyze(circuits)
    assert len(recs) == 2
    for rec in recs:
        assert isinstance(rec, AdvisorRecommendation)