"""NVIDIA NIM LLM Advisor for quantum circuit backend selection.

Uses NVIDIA's free NIM inference APIs to analyze QASM circuits
and recommend the most efficient simulation backend.
"""

import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import requests

from backend.abstract import SimulationResult
from backend.model_catalog import NVIDIA_MODELS, NVIDIA_DEFAULT

# Convenience flat map: alias → model_id (for backwards-compat callers)
AVAILABLE_MODELS = {alias: e.model_id for alias, e in NVIDIA_MODELS.items()}
DEFAULT_MODEL = NVIDIA_DEFAULT.model_id


class RateLimiter:
    """Simple token‑bucket rate limiter for API calls."""

    def __init__(self, max_calls_per_minute: int = 60):
        self.min_interval = 60.0 / max_calls_per_minute
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


@dataclass
class AdvisorRecommendation:
    """Structured recommendation from the LLM advisor."""
    recommended_backend: str
    confidence: float
    reasoning: str
    model_used: str
    latency_seconds: float
    raw_response: str


class NIMBackendAdvisor:
    """Client for NVIDIA NIM LLM APIs to advise on quantum simulation backends.

    Usage:
        advisor = NIMBackendAdvisor(api_key="nvapi-...")
        rec = advisor.recommend_backend(qasm_string, circuit_fingerprint)
        print(rec.recommended_backend)  # "aer_mps"
        print(rec.reasoning)            # "This GHZ circuit has low entanglement..."
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: str = "https://integrate.api.nvidia.com/v1/chat/completions",
        timeout: int = 30,
    ):
        """Initialize the NVIDIA NIM advisor.

        Args:
            api_key: NVIDIA API key (starts with 'nvapi-').
            model: Model identifier (e.g., 'meta/llama-3.1-8b-instruct').
            endpoint: NVIDIA NIM chat completions endpoint.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        # Rate limiter for 60 RPM free tier
        self._rate_limiter = RateLimiter(max_calls_per_minute=60)

    def _call_api(self, messages: list[Dict[str, str]], max_tokens: int = 200) -> str:
        """Make a request to the NVIDIA NIM API.

        Args:
            messages: List of chat messages in OpenAI format.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If the API call fails after retries.
        """
        # Respect the 60 RPM limit
        self._rate_limiter.wait()

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for deterministic recommendations
        }

        for attempt in range(3):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"NVIDIA NIM API call failed: {e}") from e
                time.sleep(2 ** attempt)

        raise RuntimeError("NVIDIA NIM API call failed after retries")

    def recommend_backend(
        self,
        qasm: str,
        fingerprint: Dict[str, Any],
        available_backends: Optional[list[str]] = None,
    ) -> AdvisorRecommendation:
        """Recommend the best simulation backend for a given circuit.

        Args:
            qasm: QASM string of the circuit.
            fingerprint: Circuit fingerprint dict from circuit_fingerprint.py.
            available_backends: List of backend names to choose from.
                Defaults to ['aer_statevector', 'aer_mps'].

        Returns:
            AdvisorRecommendation with backend name, confidence, and reasoning.
        """
        if available_backends is None:
            available_backends = ["aer_statevector", "aer_mps"]

        # Build a concise prompt with circuit facts
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
        graph_diameter = fingerprint.get("interaction_graph", {}).get("diameter", "unknown")

        user_prompt = (
            f"Circuit: {n_qubits} qubits, depth {depth}, {cx_count} CNOT gates, "
            f"interaction graph diameter {graph_diameter}. "
            f"Available backends: {', '.join(available_backends)}. "
            f"QASM snippet (first 500 chars): {qasm[:500]}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        raw_response = self._call_api(messages, max_tokens=200)
        latency = time.perf_counter() - t0

        # Save original response before any manipulation
        original_response = raw_response

        # Extract JSON from response (may be wrapped in ```json blocks)
        try:
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            parsed = json.loads(json_str.strip())
            backend = parsed.get("backend", available_backends[0])
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", original_response)
        except (json.JSONDecodeError, KeyError):
            # Fallback: use first available backend with low confidence
            backend = available_backends[0] if available_backends else "aer_statevector"
            confidence = 0.3
            reasoning = f"Failed to parse LLM response. Raw: {original_response[:200]}"

        return AdvisorRecommendation(
            recommended_backend=backend,
            confidence=min(max(confidence, 0.0), 1.0),
            reasoning=reasoning,
            model_used=self.model,
            latency_seconds=latency,
            raw_response=original_response,  # Use original unmodified response
        )

    def explain_result(self, result: SimulationResult) -> str:
        """Generate a natural language explanation of a benchmark result.

        Args:
            result: A SimulationResult from a completed benchmark.

        Returns:
            Human-readable explanation string.
        """
        prompt = (
            f"Explain this quantum simulation benchmark result in 2-3 sentences: "
            f"Circuit: {result.circuit_name}, {result.n_qubits} qubits, depth {result.depth}. "
            f"Backend: {result.backend_name}. "
            f"Total time: {result.total_time_seconds:.3f}s. "
            f"Peak GPU memory: {result.peak_gpu_memory_mb:.0f}MB. "
            f"Success: {result.success}. "
            f"{'Fidelity: ' + str(result.fidelity) if result.fidelity else ''}"
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages, max_tokens=150)

    def batch_analyze(
        self,
        circuits: list[Dict[str, Any]],
        available_backends: Optional[list[str]] = None,
    ) -> list[AdvisorRecommendation]:
        """Analyze multiple circuits and return recommendations.

        Args:
            circuits: List of dicts with 'qasm' and 'fingerprint' keys.
            available_backends: Backend names to choose from.

        Returns:
            List of AdvisorRecommendation objects, one per circuit.
        """
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
                    recommended_backend=available_backends[0] if available_backends else "aer_statevector",
                    confidence=0.0,
                    reasoning=f"Error: {e}",
                    model_used=self.model,
                    latency_seconds=0.0,
                    raw_response=str(e),
                )
            recommendations.append(rec)
        return recommendations


# ---------------------------------------------------------------------------
# Groq advisor (same interface, different endpoint)
# ---------------------------------------------------------------------------

class GroqBackendAdvisor:
    """LLM advisor backed by Groq inference (same interface as NIMBackendAdvisor).

    Uses the OpenAI-compatible Groq endpoint. Intended as a rate-limit fallback
    when NVIDIA NIM returns 429 errors.

    Usage:
        advisor = GroqBackendAdvisor(api_key="gsk_...")
        rec = advisor.recommend_backend(qasm, fingerprint)
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        from backend.model_catalog import GROQ_DEFAULT, GROQ_ENDPOINT
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

    def _call_api(self, messages: list[Dict[str, str]], max_tokens: int = 200) -> str:
        self._rate_limiter.wait()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        for attempt in range(3):
            try:
                response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
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
        available_backends: Optional[list[str]] = None,
    ) -> AdvisorRecommendation:
        """Same signature as NIMBackendAdvisor.recommend_backend."""
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


# ---------------------------------------------------------------------------
# Cascading advisor: NVIDIA → Groq → rule baseline
# ---------------------------------------------------------------------------

class CascadingAdvisor:
    """Try NVIDIA NIM, fall back to Groq on 429, then to the rule baseline.

    Usage:
        advisor = CascadingAdvisor.from_env()
        rec = advisor.recommend_backend(qasm, fingerprint)

    The ``recommended_backend`` field always has a value; the ``reasoning``
    field identifies which tier answered.
    """

    def __init__(
        self,
        nvidia_advisor: Optional[NIMBackendAdvisor],
        groq_advisor: Optional["GroqBackendAdvisor"],
    ):
        self._nvidia = nvidia_advisor
        self._groq = groq_advisor

    @classmethod
    def from_env(cls) -> "CascadingAdvisor":
        """Build from environment variables (NVIDIA_API_KEY, GROQ_API_KEY)."""
        import os
        nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()

        nvidia = NIMBackendAdvisor(api_key=nvidia_key) if nvidia_key else None
        groq = GroqBackendAdvisor(api_key=groq_key) if groq_key else None
        return cls(nvidia_advisor=nvidia, groq_advisor=groq)

    def recommend_backend(
        self,
        qasm: str,
        fingerprint: Dict[str, Any],
        available_backends: Optional[list[str]] = None,
    ) -> AdvisorRecommendation:
        """Try advisors in priority order; fall back to rule baseline."""
        from research.backend_selector.rule_baseline import rule_baseline_predict

        if available_backends is None:
            available_backends = ["aer_statevector", "aer_mps"]

        # Tier 1: NVIDIA
        if self._nvidia:
            try:
                return self._nvidia.recommend_backend(qasm, fingerprint, available_backends)
            except RuntimeError as e:
                if "429" not in str(e) and "rate" not in str(e).lower():
                    raise

        # Tier 2: Groq (reached only on NVIDIA rate limit)
        if self._groq:
            try:
                rec = self._groq.recommend_backend(qasm, fingerprint, available_backends)
                rec = AdvisorRecommendation(
                    recommended_backend=rec.recommended_backend,
                    confidence=rec.confidence,
                    reasoning=f"[Groq fallback] {rec.reasoning}",
                    model_used=rec.model_used,
                    latency_seconds=rec.latency_seconds,
                    raw_response=rec.raw_response,
                )
                return rec
            except Exception:
                pass

        # Tier 3: Local rule baseline (always available)
        n_qubits = fingerprint.get("qubits", 0)
        backend = rule_baseline_predict(int(n_qubits))
        return AdvisorRecommendation(
            recommended_backend=backend,
            confidence=0.6,
            reasoning=f"[Rule baseline] qubit_count={n_qubits} → {backend}",
            model_used="rule_baseline",
            latency_seconds=0.0,
            raw_response="",
        )

    def batch_analyze(
        self,
        circuits: list[Dict[str, Any]],
        available_backends: Optional[list[str]] = None,
    ) -> list[AdvisorRecommendation]:
        """Analyze multiple circuits, cascading per call."""
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
                    reasoning=f"CascadingAdvisor error: {e}",
                    model_used="none",
                    latency_seconds=0.0,
                    raw_response=str(e),
                )
            recommendations.append(rec)
        return recommendations
        return recommendations