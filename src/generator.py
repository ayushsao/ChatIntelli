"""
src/generator.py

Reply generation and escalation decision module.
Uses Gemini 1.5 Flash with JSON-mode structured output.
Falls back to rule-based escalation policy when no API key is set.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class ReplyResponse(BaseModel):
    action: str              # "AUTO_HANDLE" or "ESCALATE"
    reason: str
    confidence: float
    reply: str
    unsupported_claims: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule-based fallback (no API key)
# ---------------------------------------------------------------------------

# Intents that ALWAYS require escalation regardless of retrieval
_ALWAYS_ESCALATE_INTENTS = {"account_login", "payment_issue"}

# Minimum retrieval similarity required to attempt an AUTO_HANDLE response.
# Set deliberately high (0.35) so the fallback is conservative —
# it is safer to over-escalate than to give a wrong automated reply.
_AUTO_HANDLE_SIM_THRESHOLD = 0.35

# Intents where we auto-handle only with a very strong retrieval match
_HIGH_THRESHOLD_INTENTS = {"app_bug", "content_missing"}

# Intents that can auto-handle with the standard threshold
_STANDARD_THRESHOLD_INTENTS = {"feature_request", "other"}


def _rule_based_escalate(intent: str, retrieval_context: list[dict]) -> ReplyResponse:
    """
    Conservative rule-based escalation used when GEMINI_API_KEY is not set.
    Prefers ESCALATE over AUTO_HANDLE when uncertain.
    Mirrors the LLM system prompt's escalation spirit.
    """
    # Hard-coded escalation intents
    if intent in _ALWAYS_ESCALATE_INTENTS:
        return ReplyResponse(
            action="ESCALATE",
            reason=f"Intent '{intent}' always requires account-level investigation.",
            confidence=0.95,
            reply="",
        )

    top_sim = max((r.get("similarity", 0.0) for r in retrieval_context), default=0.0)

    # Determine threshold based on intent
    if intent in _HIGH_THRESHOLD_INTENTS:
        # App bugs and missing content: require a strong retrieval match
        threshold = _AUTO_HANDLE_SIM_THRESHOLD + 0.20   # 0.55
    else:
        threshold = _AUTO_HANDLE_SIM_THRESHOLD           # 0.35

    if top_sim < threshold:
        return ReplyResponse(
            action="ESCALATE",
            reason=(
                f"Retrieval similarity ({top_sim:.2f}) below threshold ({threshold:.2f}) "
                f"for intent '{intent}'. Cannot generate a grounded reply."
            ),
            confidence=0.7,
            reply="",
        )

    top = retrieval_context[0]
    return ReplyResponse(
        action="AUTO_HANDLE",
        reason=f"Strong historical match found (sim={top_sim:.2f}) for intent '{intent}'.",
        confidence=top_sim,
        reply=top["brand_reply"],
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an AI customer support agent for SpotifyCares.

Given a customer message, a classified intent, and historical support examples, you must:
1. Draft a concise, professional reply (≤ 3 sentences).
2. Decide whether to AUTO_HANDLE or ESCALATE.

ESCALATE rules (non-negotiable):
- Account-specific actions required (login, hacked account, refund, billing dispute)
- Historical context lacks a clear resolution for this issue
- Request involves sensitive personal data
- Customer message is unclear or requires human judgement

AUTO_HANDLE rules:
- Known generic bug with a documented fix in the historical context
- Feature question with a standard documented answer
- Response can be fully grounded in the retrieved examples

CRITICAL — NEVER invent:
- Specific policy details not in the retrieved examples
- Timelines ("we'll fix this in 24 hours")
- Guarantees ("your refund will be processed")
- Account-specific information
- Links not present in retrieval

If historical context is insufficient, ESCALATE rather than guess.
Tone: professional, empathetic, concise. Match SpotifyCares Twitter support voice.

Return JSON with: action (AUTO_HANDLE or ESCALATE), reason (string), confidence (float 0-1),
reply (string, empty if escalating), unsupported_claims (list of strings).
"""


# ---------------------------------------------------------------------------
# Gemini-based generator
# ---------------------------------------------------------------------------

class ReplyGenerator:
    """
    Generates a grounded reply and escalation decision for a customer message.
    Uses Gemini 1.5 Flash with JSON-mode output.
    Falls back to rule-based policy when no API key is set.
    """

    MODEL = "gemini-flash-lite-latest"

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        self._model = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(
                    model_name=self.MODEL,
                    system_instruction=_SYSTEM_PROMPT,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ReplyResponse,
                        temperature=0.0,
                    ),
                )
                logger.info("ReplyGenerator: using Gemini %s", self.MODEL)
            except ImportError:
                logger.warning("google-generativeai not installed — using rule-based fallback.")
        else:
            logger.warning("GEMINI_API_KEY not set. Using rule-based escalation fallback.")

    def generate(
        self,
        customer_message: str,
        intent: str,
        retrieval_context: list[dict],
    ) -> ReplyResponse:
        if not self._model:
            return _rule_based_escalate(intent, retrieval_context)

        user_lines = [
            f"Customer message: {customer_message}",
            f"Intent: {intent}",
            "",
            "Historical support examples (ordered by relevance):",
        ]
        for i, ctx in enumerate(retrieval_context, 1):
            user_lines.append(
                f"[{i}] Customer: {ctx['customer_message']}\n"
                f"    SpotifyCares: {ctx['brand_reply']}\n"
                f"    Similarity: {ctx['similarity']:.2f}"
            )

        retries = 5
        for attempt in range(retries):
            try:
                response = self._model.generate_content("\n".join(user_lines))
                data = json.loads(response.text)
                return ReplyResponse(**data)
            except Exception as exc:
                if "429" in str(exc):
                    if attempt < retries - 1:
                        logger.warning("Rate limit hit (429). Retrying in 16s...")
                        time.sleep(16)
                        continue
                logger.error("Gemini generation error: %s — escalating by default", exc)
                break
                
        return ReplyResponse(
            action="ESCALATE",
            reason="Generation error / Rate limit",
            confidence=0.0,
            reply="",
        )
