"""Central model catalog for NVIDIA NIM and Groq inference APIs.

Single source of truth for model IDs, endpoints, and rate limits used across
the LLM advisor (backend/llm_advisor.py) and question generators.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelEntry:
    """Metadata for one hosted inference model."""
    model_id: str
    provider: str          # "nvidia" | "groq"
    alias: str             # short human name used as dict key
    use_case: str          # "fast" | "accurate" | "reasoning" | "question_gen"
    context_window: int    # tokens
    rpm: int               # free-tier requests-per-minute (0 = unknown)
    endpoint: str
    notes: str = ""


# ---------------------------------------------------------------------------
# NVIDIA NIM
# ---------------------------------------------------------------------------
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

NVIDIA_MODELS: dict[str, ModelEntry] = {
    # ── LLM backend advisor ─────────────────────────────────────────────
    "fast": ModelEntry(
        model_id="meta/llama-3.1-8b-instruct",
        provider="nvidia",
        alias="fast",
        use_case="fast",
        context_window=128_000,
        rpm=60,
        endpoint=NVIDIA_ENDPOINT,
        notes="Best latency for interactive advisor calls.",
    ),
    "accurate": ModelEntry(
        model_id="meta/llama-3.3-70b-instruct",
        provider="nvidia",
        alias="accurate",
        use_case="accurate",
        context_window=128_000,
        rpm=30,
        endpoint=NVIDIA_ENDPOINT,
        notes="Higher accuracy; slower; use for production recommendations.",
    ),
    "reasoning": ModelEntry(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        provider="nvidia",
        alias="reasoning",
        use_case="reasoning",
        context_window=32_768,
        rpm=10,
        endpoint=NVIDIA_ENDPOINT,
        notes="Heavy chain-of-thought; overkill for simple routing decisions.",
    ),
    "quantum": ModelEntry(
        model_id="nvidia/ising-calibration-1-35b-a3b",
        provider="nvidia",
        alias="quantum",
        use_case="reasoning",
        context_window=32_768,
        rpm=10,
        endpoint=NVIDIA_ENDPOINT,
        notes="Quantum calibration VLM; experimental.",
    ),
    # ── Question generators ──────────────────────────────────────────────
    "question_gen": ModelEntry(
        model_id="nvidia/llama-3.3-nemotron-super-49b-v1",
        provider="nvidia",
        alias="question_gen",
        use_case="question_gen",
        context_window=32_768,
        rpm=10,
        endpoint=NVIDIA_ENDPOINT,
        notes="Used by nvidia_question_generator.py for GATE exam questions.",
    ),
}

# Default model for the NIM backend advisor
NVIDIA_DEFAULT = NVIDIA_MODELS["fast"]


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODELS: dict[str, ModelEntry] = {
    "fast": ModelEntry(
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        provider="groq",
        alias="fast",
        use_case="question_gen",
        context_window=131_072,
        rpm=30,
        endpoint=GROQ_ENDPOINT,
        notes="Used by groq_question_generator.py; very fast inference.",
    ),
    "accurate": ModelEntry(
        model_id="llama-3.3-70b-versatile",
        provider="groq",
        alias="accurate",
        use_case="accurate",
        context_window=128_000,
        rpm=30,
        endpoint=GROQ_ENDPOINT,
        notes="High quality; good fallback when NVIDIA keys are rate-limited.",
    ),
    "fast_mini": ModelEntry(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        alias="fast_mini",
        use_case="fast",
        context_window=131_072,
        rpm=60,
        endpoint=GROQ_ENDPOINT,
        notes="Lowest latency Groq option; use for bulk advisor calls.",
    ),
    "mixtral": ModelEntry(
        model_id="mixtral-8x7b-32768",
        provider="groq",
        alias="mixtral",
        use_case="accurate",
        context_window=32_768,
        rpm=30,
        endpoint=GROQ_ENDPOINT,
        notes="Strong reasoning/analysis; good alternative to Nemotron.",
    ),
}

# Default model for Groq-based advisor
GROQ_DEFAULT = GROQ_MODELS["accurate"]


# ---------------------------------------------------------------------------
# Unified lookup helpers
# ---------------------------------------------------------------------------

ALL_MODELS: dict[str, ModelEntry] = {
    **{f"nvidia/{k}": v for k, v in NVIDIA_MODELS.items()},
    **{f"groq/{k}": v for k, v in GROQ_MODELS.items()},
}


def get_model(provider: str, alias: str) -> ModelEntry:
    """Return a ModelEntry by provider + alias, or raise KeyError."""
    key = f"{provider}/{alias}"
    if key not in ALL_MODELS:
        available = ", ".join(ALL_MODELS)
        raise KeyError(f"Model '{key}' not in catalog. Available: {available}")
    return ALL_MODELS[key]


def list_models(provider: Optional[str] = None) -> list[ModelEntry]:
    """Return all catalog entries, optionally filtered by provider."""
    entries = list(ALL_MODELS.values())
    if provider:
        entries = [e for e in entries if e.provider == provider]
    return entries
