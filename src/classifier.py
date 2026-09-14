"""
src/classifier.py

Intent classification module for SpotifyCares support agent.
Provides two classifiers:
  - KeywordHeuristicClassifier: Deterministic fallback (no API key required)
  - ProposedClassifier: Gemini-based structured output (requires GEMINI_API_KEY)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from pydantic import BaseModel
from src.intents import Intent, INTENT_DEFINITIONS, VALID_INTENTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class IntentClassificationResult(BaseModel):
    intent: Intent
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Deterministic keyword heuristic (no API required)
# Used as fallback AND as the baseline labeler for the golden set
# ---------------------------------------------------------------------------

class KeywordHeuristicClassifier:
    """
    Rule-based intent classifier using ordered keyword matching.

    Order matters: more specific patterns are checked first.
    app_bug is checked before feature_request to avoid classifying
    "my app crashes" as a feature request just because it mentions "update".
    """

    def predict_single(self, text: str) -> str:
        text_lower = text.lower()

        # 1. Account / Login
        if any(k in text_lower for k in [
            'hacked', 'login', 'password', 'reset', 'email changed',
            'log in', 'access my account', 'locked out', 'verify',
            'verification', 'sign in', 'sign up', 'credentials',
            'forgot my', 'approve my spotify', 'spotify for artists',
        ]):
            return Intent.ACCOUNT_LOGIN.value

        # 2. Payment
        if any(k in text_lower for k in [
            'charge', 'charged', 'payment', 'premium', 'billing',
            'credit card', 'debit card', 'refund', 'cancel', 'deducted',
            'bill', 'invoice', 'subscription', 'renew', 'money',
        ]):
            return Intent.PAYMENT_ISSUE.value

        # 3. Content missing
        if any(k in text_lower for k in [
            'greyed', 'grayed', 'removed', 'remove', 'podcast',
            'album', 'artist', 'song removed', "can't find", 'not available',
            'no longer', 'anymore', 'metadata', 'wrong title', 'wrong name',
        ]):
            return Intent.CONTENT_MISSING.value

        # 4. App bug — checked BEFORE feature_request (more specific signals)
        if any(k in text_lower for k in [
            'crash', 'crashes', 'crashing', 'skip', 'skipping',
            'pause', 'pauses', 'randomly pauses', 'offline', 'downloaded',
            'freez', 'stop', 'stops playing', 'bug', 'glitch',
            "won't play", "won't load", "won't open",
            'not working', "doesn't work", 'broken',
            'error', 'slow', 'lag', 'blank screen', 'black screen',
            'keeps stopping', 'keeps crashing',
        ]):
            return Intent.APP_BUG.value

        # 5. Feature request
        if any(k in text_lower for k in [
            'bring back', 'lyrics', 'feature', 'shuffle',
            'design', 'wish', 'would be nice',
            'should have', 'when will', 'come to', 'available in',
            'please add', 'can you add', 'request',
        ]):
            return Intent.FEATURE_REQUEST.value

        return Intent.OTHER.value

    def predict(self, messages: list[str]) -> list[str]:
        return [self.predict_single(m) for m in messages]


# ---------------------------------------------------------------------------
# Gemini-based classifier (requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------

class ProposedClassifier:
    """
    Intent classifier using Gemini 1.5 Flash with JSON-mode structured output.
    Falls back to KeywordHeuristicClassifier if no API key is set.
    """

    MODEL = "gemini-flash-lite-latest"

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        self._model = None
        self._heuristic = KeywordHeuristicClassifier()
        self._system_prompt = self._build_system_prompt()

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(
                    model_name=self.MODEL,
                    system_instruction=self._system_prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=IntentClassificationResult,
                        temperature=0.0,
                    ),
                )
                logger.info("ProposedClassifier: using Gemini %s", self.MODEL)
            except ImportError:
                logger.warning("google-generativeai not installed — using keyword heuristic.")
        else:
            logger.warning(
                "GEMINI_API_KEY not set. ProposedClassifier will use keyword heuristic fallback."
            )

    def _build_system_prompt(self) -> str:
        lines = [
            "You are a senior support routing system for SpotifyCares.",
            "Classify the customer message into EXACTLY ONE of the following intents.",
            "Return JSON with: intent (string), confidence (float 0-1), reason (short string).\n",
        ]
        for intent_enum, details in INTENT_DEFINITIONS.items():
            lines.append(f"INTENT: {intent_enum.value}")
            lines.append(f"  Description: {details['description']}")
            lines.append(f"  Include: {details['inclusion_criteria']}")
            lines.append(f"  Exclude: {details['exclusion_criteria']}")
            lines.append(f"  Examples: {details['examples']}\n")
        return "\n".join(lines)

    def predict_single(self, text: str) -> str:
        return self.predict_full(text).intent.value

    def predict_full(self, text: str) -> IntentClassificationResult:
        if not self._model:
            intent_val = self._heuristic.predict_single(text)
            return IntentClassificationResult(
                intent=Intent(intent_val),
                confidence=0.7,
                reason="Keyword heuristic fallback (no API key).",
            )
        retries = 5
        for attempt in range(retries):
            try:
                response = self._model.generate_content(text)
                data = json.loads(response.text)
                return IntentClassificationResult(**data)
            except Exception as exc:
                if "429" in str(exc):
                    if attempt < retries - 1:
                        logger.warning("Rate limit hit (429). Retrying in 16s...")
                        time.sleep(16)
                        continue
                logger.error("Gemini classification error: %s — falling back to heuristic", exc)
                break
                
        intent_val = self._heuristic.predict_single(text)
        return IntentClassificationResult(
            intent=Intent(intent_val),
            confidence=0.5,
            reason="Heuristic fallback after API error.",
        )

    def predict(self, messages: list[str]) -> list[str]:
        return [self.predict_single(m) for m in messages]
