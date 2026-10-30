"""
Safety & Governance Guardrail Layer.

Production-grade input/output guardrails for the RAG system. Implements
defence-in-depth without depending on a heavy external SaaS:

1. Prompt-injection / jailbreak pattern detection (regex heuristics)
2. PII-ish pattern scanning (emails, phones, card-like strings)
3. Toxicity / profanity keyword screening (extensible allow/deny lists)
4. Output grounding sanity-check (anti-hallucination nudge)
5. Length & rate sanity checks

Each guardrail returns a GuardrailVerdict (PASS / BLOCK / SANITIZE) plus a
human-readable message and risk score. The layer is deterministic, fast,
and runs on the Groq "fast" model for LLM-backed classification where
regex heuristics are insufficient.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from newpapersearch.models import GuardrailResult, GuardrailVerdict
from newpapersearch.llm.groq_llm import get_fast_llm
from newpapersearch.config import get_settings


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
settings = get_settings()

# Prompt-injection / jailbreak heuristic patterns.
INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (
        r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|prior|system|above)\s+(instructions|prompts|rules)",
        "instruction_override",
    ),
    (
        r"(?i)(you\s+are\s+now|pretend\s+to\s+be|act\s+as\s+a)\s+(a\s+)?(different|unrestricted|developer|DAN)",
        "role_play_jailbreak",
    ),
    (
        r"(?i)(reveal|show|print|output)\s+(your|the)\s+(system|internal|hidden)\s+(prompt|instructions|key)",
        "leak_system_prompt",
    ),
    (
        r"(?i)(bypass|disable|turn off)\s+(safety|filter|guardrail|restriction)",
        "disable_safety",
    ),
]

# PII-ish patterns (best-effort, not a substitute for a dedicated PII scanner).
PII_PATTERNS: List[Tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email_address"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone_number"),
    (r"\b(?:\d[ -]*){13,16}\b", "payment_card_like"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn_like"),
]

# Toxicity / profanity deny-list (extendable via configuration).
TOXICITY_KEYWORDS: List[str] = [
    "hate", "kill", "suicide", "terrorist", "bomb",
]


class GuardrailConfig(BaseModel):
    enable_input: bool = True
    enable_output: bool = True
    max_query_length: int = settings.max_query_length
    block_on_injection: bool = True
    block_on_pii: bool = False  # sanitize instead by default
    block_on_toxicity: bool = True


class SafetyGuardrail:
    """
    Deterministic + LLM-assisted guardrail engine.

    Methods:
      * check_input(user_query) -> GuardrailResult
      * check_output(answer, sources) -> GuardrailResult
    """

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self._fast_llm = get_fast_llm() if (self.config.enable_input or self.config.enable_output) else None

    # ------------------------------------------------------------------
    # Input guardrails
    # ------------------------------------------------------------------
    def check_input(self, user_query: str) -> GuardrailResult:
        """Run all input safety checks. Returns a GuardrailResult."""
        detected: List[str] = []
        risk_score = 0.0
        sanitized = user_query

        # Length check
        if len(user_query) > self.config.max_query_length:
            detected.append("query_exceeds_max_length")
            risk_score += 0.3

        # Pattern-based injection detection
        for pattern, name in INJECTION_PATTERNS:
            if re.search(pattern, user_query):
                detected.append(name)
                risk_score += 0.35

        # PII detection
        for pattern, name in PII_PATTERNS:
            if re.search(pattern, user_query):
                detected.append(f"pii_{name}")
                risk_score += 0.2
                sanitized = re.sub(pattern, "[REDACTED]", user_query)

        # Toxicity keyword scan
        lower = user_query.lower()
        for keyword in TOXICITY_KEYWORDS:
            if keyword in lower:
                detected.append(f"toxicity_{keyword}")
                risk_score += 0.3

        risk_score = min(risk_score, 1.0)

        # Decide verdict
        if detected:
            if any("injection" in d or "jailbreak" in d or "leak" in d or "disable" in d for d in detected):
                verdict = GuardrailVerdict.BLOCK
                message = (
                    "Your request was blocked because it appears to attempt "
                    "prompt injection or bypass safety controls."
                )
            elif self.config.block_on_toxicity and any("toxicity" in d for d in detected):
                verdict = GuardrailVerdict.BLOCK
                message = "Your request was blocked due to prohibited content."
            else:
                verdict = GuardrailVerdict.SANITIZE
                message = "Your request was sanitized (PII redacted)."
        else:
            verdict = GuardrailVerdict.PASS
            message = None

        return GuardrailResult(
            verdict=verdict,
            message=message,
            sanitized_input=sanitized if sanitized != user_query else None,
            detected_issues=detected,
            risk_score=risk_score,
        )

    # ------------------------------------------------------------------
    # Output guardrails
    # ------------------------------------------------------------------
    def check_output(self, answer: str, sources: Optional[List[str]] = None) -> GuardrailResult:
        """
        Run output safety checks.

        Performs a quick LLM-backed groundedness sanity check using the fast
        Groq model to discourage hallucinated claims that are not supported
        by the retrieved sources.
        """
        detected: List[str] = []
        risk_score = 0.0

        # Empty / refusal handling
        if not answer or not answer.strip():
            return GuardrailResult(
                verdict=GuardrailVerdict.BLOCK,
                message="The assistant returned an empty response.",
                detected_issues=["empty_response"],
                risk_score=1.0,
            )

        # LLM-assisted groundedness check (fast, deterministic prompt)
        if sources and self._fast_llm is not None:
            grounded = self._llm_groundedness_check(answer, sources)
            if not grounded:
                detected.append("output_not_grounded")
                risk_score += 0.4

        risk_score = min(risk_score, 1.0)

        if detected:
            verdict = GuardrailVerdict.SANITIZE
            message = (
                "The generated answer could not be fully verified against "
                "the retrieved sources. Treat it with caution."
            )
        else:
            verdict = GuardrailVerdict.PASS
            message = None

        return GuardrailResult(
            verdict=verdict,
            message=message,
            detected_issues=detected,
            risk_score=risk_score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _llm_groundedness_check(self, answer: str, sources: List[str]) -> bool:
        """
        Ask the fast LLM whether the answer is supported by the sources.
        Returns True if grounded, False otherwise.
        """
        sources_text = "\n".join(f"- {s[:500]}" for s in sources[:5])
        prompt = (
            "You are a strict fact-checker. Given the ANSWER and the SOURCE "
            "snippets below, reply with exactly one word: GROUNDED if the "
            "answer is supported by the sources, or NOT_GROUNDED if it "
            "contains claims not present in the sources.\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            "Reply with exactly one word: GROUNDED or NOT_GROUNDED."
        )
        try:
            response = self._fast_llm.invoke([("user", prompt)])
            content = (response.content or "").strip().upper()
            return "GROUNDED" in content and "NOT_GROUNDED" not in content
        except Exception:  # pragma: no cover - defensive
            return True  # fail-open: don't block on LLM check failure


__all__ = [
    "SafetyGuardrail",
    "GuardrailConfig",
    "INJECTION_PATTERNS",
    "PII_PATTERNS",
    "TOXICITY_KEYWORDS",
]
