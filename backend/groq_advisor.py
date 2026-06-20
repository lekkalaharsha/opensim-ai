"""Groq-backed LLM advisor for quantum circuit backend selection.

Implements the same ``recommend_backend`` interface as ``NIMBackendAdvisor``
using Groq's OpenAI-compatible endpoint.  Intended as a rate-limit fallback
when NVIDIA NIM returns 429 errors.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests

from backend.model_catalog import GROQ_DEFAULT, GROQ_ENDPOINT


@dataclass
class AdvisorRecommendation:
    """Structured recommendation returned by an advisor."""
    recommended_backend: str
    confidence: float
    reasoning: str
    model_used: str
    latency_seconds: float
    raw_response: str


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, max_calls_per_minute: int = 30):
        self.min_interval = 60.0 / max_calls_per_minute
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class GroqBackendAdvisor:
    """LLM advisor backed by Groq inference.

    Uses the OpenAI-compatible Groq endpoint.  Intended as a rate-limit
    fallback when NVIDIA NIM returns 429 errors.

    Usage::

        advisor = GroqBackendAdvisor(api_key="gsk_...")
        rec = advisor.recommend_backend(qasm, fingerprint)
        print(rec.recommended_backend, rec.confidence)
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model or GROQ_DEFAULT.model_id
        self.endpoint = GROQ_ENDPOINT
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._rate_limiter = RateLimiter(max_calls_per_minute=30)

    def _call_api(self, messages: list, max_tokens: int = 200) -> str:
        self._rate_limiter.wait()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.endpoint, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Groq API call failed: {e}") from e
                time.sleep(2 ** attempt)
        raise RuntimeError("Groq API call failed after retries")

    def recommend_backend(
        self,
        qasm: str,
        fingerprint: Dict[str, Any],
        available_backends: Optional[list] = None,
    ) -> AdvisorRecommendation:
        """Recommend the best simulation backend for the given circuit.

        Args:
            qasm: QASM string of the circuit.
            fingerprint: Circuit fingerprint dict from circuit_fingerprint.py.
            available_backends: Backend names to choose from.

        Returns:
            AdvisorRecommendation with backend, confidence, and reasoning.
        """
        if available_backends is None:
            available_backends = ["aer_statevector", "aer_mps"]

        system_prompt = (
            "You are a quantum computing simulation expert. "
            "Given a quantum circuit's structure and metrics, "
            "recommend the most efficient simulator backend. "
            "Answer in JSON format only: "
            '{"backend": "<name>", "confidence": <0.0-1.0>, "reasoning": "<explanation>"}'
        )
        n_qubits = fingerprint.get("qubits", "unknown")
        depth = fingerprint.get("depth", "unknown")
        cx_count = fingerprint.get("gate_counts", {}).get("cx", "unknown")
        user_prompt = (
            f"Circuit: {n_qubits} qubits, depth {depth}, {cx_count} CNOT gates. "
            f"Available backends: {', '.join(available_backends)}. "
            f"QASM snippet (first 500 chars): {qasm[:500]}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        raw = self._call_api(messages, max_tokens=200)
        latency = time.perf_counter() - t0

        try:
            json_str = raw
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            parsed = json.loads(json_str.strip())
            backend = parsed.get("backend", available_backends[0])
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", raw)
        except (json.JSONDecodeError, KeyError):
            backend = available_backends[0]
            confidence = 0.3
            reasoning = f"Failed to parse Groq response. Raw: {raw[:200]}"

        return AdvisorRecommendation(
            recommended_backend=backend,
            confidence=min(max(confidence, 0.0), 1.0),
            reasoning=reasoning,
            model_used=self.model,
            latency_seconds=latency,
            raw_response=raw,
        )

    def batch_analyze(
        self,
        circuits: list,
        available_backends: Optional[list] = None,
    ) -> list:
        """Analyze multiple circuits and return recommendations."""
        recommendations = []
        for circuit in circuits:
            try:
                rec = self.recommend_backend(
                    qasm=circuit["qasm"],
                    fingerprint=circuit["fingerprint"],
                    available_backends=available_backends,
                )
            except Exception as e:
                rec = AdvisorRecommendation(
                    recommended_backend=(available_backends or ["aer_statevector"])[0],
                    confidence=0.0,
                    reasoning=f"Error: {e}",
                    model_used=self.model,
                    latency_seconds=0.0,
                    raw_response=str(e),
                )
            recommendations.append(rec)
        return recommendations
