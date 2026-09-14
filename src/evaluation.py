"""
src/evaluation.py

LLM-as-judge evaluation module using Gemini 1.5 Flash.

Scores generated replies on 5 dimensions (each 1–5):
  - relevance:         Is the reply on-topic for the customer message?
  - groundedness:      Is it strictly based on retrieved historical context?
  - helpfulness:       Would this reply actually help the customer?
  - tone:              Professional, empathetic, and concise?
  - no_hallucination:  Higher = fewer unsupported claims (5 = none detected)

Also provides:
  - pass_fail:                    "pass" or "fail"
  - reason:                       Short explanation of the overall verdict
  - detected_unsupported_claims:  List of specific unsupported statements found
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
# Pydantic rubric schema
# ---------------------------------------------------------------------------

class JudgeRubric(BaseModel):
    relevance: int            # 1–5
    groundedness: int         # 1–5
    helpfulness: int          # 1–5
    tone: int                 # 1–5
    no_hallucination: int     # 1–5 (5 = no unsupported claims)
    pass_fail: str            # "pass" or "fail"
    reason: str
    detected_unsupported_claims: list[str] = Field(default_factory=list)

    @property
    def overall(self) -> float:
        return (
            self.relevance + self.groundedness + self.helpfulness
            + self.tone + self.no_hallucination
        ) / 5.0


# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert customer service quality auditor evaluating AI-generated support replies.

Score the generated reply on each dimension from 1 to 5 (integer only):

1. relevance (1–5):
   5 = Directly addresses the customer's specific issue
   3 = Partially addresses the issue
   1 = Completely off-topic

2. groundedness (1–5):
   5 = Every claim is supported by the provided historical examples
   3 = Mostly grounded with minor assumptions
   1 = Mostly invented, not supported by context

3. helpfulness (1–5):
   5 = Customer would know exactly what to do next
   3 = Helpful but vague
   1 = Useless or harmful

4. tone (1–5):
   5 = Professional, empathetic, concise — matches SpotifyCares Twitter voice
   3 = Acceptable but off-tone
   1 = Rude, overpromising, or inappropriately casual

5. no_hallucination (1–5):
   5 = Zero unsupported claims (no invented policies, links, timelines, or account facts)
   3 = One or two minor unsupported claims
   1 = Multiple fabricated policies or guarantees

Then produce:
- pass_fail: "pass" if average score >= 3.0 AND no_hallucination >= 3, otherwise "fail"
- reason: 1–2 sentence explanation
- detected_unsupported_claims: list each specific unsupported statement (empty list if none)

If the reply is empty (message was escalated), return: relevance=3, groundedness=5,
helpfulness=3, tone=3, no_hallucination=5, pass_fail="pass",
reason="Message was correctly escalated; no reply drafted."

Return JSON only.
"""


# ---------------------------------------------------------------------------
# Dummy scorer (no API key)
# ---------------------------------------------------------------------------

def _dummy_score(generated_reply: str) -> JudgeRubric:
    if not generated_reply:
        return JudgeRubric(
            relevance=3, groundedness=5, helpfulness=3, tone=3, no_hallucination=5,
            pass_fail="pass",
            reason="No API key — dummy. Message escalated (no reply drafted).",
        )
    return JudgeRubric(
        relevance=3, groundedness=3, helpfulness=3, tone=3, no_hallucination=3,
        pass_fail="pass",
        reason="No GEMINI_API_KEY provided. Neutral dummy scores — do not report.",
    )


# ---------------------------------------------------------------------------
# Gemini-based LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Evaluates a generated support reply using Gemini 1.5 Flash as a judge.
    Falls back to neutral dummy scores when no API key is available.
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
                    system_instruction=_JUDGE_SYSTEM_PROMPT,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=JudgeRubric,
                        temperature=0.0,
                    ),
                )
                logger.info("LLMJudge: using Gemini %s", self.MODEL)
            except ImportError:
                logger.warning("google-generativeai not installed — returning dummy scores.")
        else:
            logger.warning("GEMINI_API_KEY not set. LLMJudge will return dummy scores.")

    def evaluate_reply(
        self,
        customer_message: str,
        generated_reply: str,
        reference_reply: str,
        retrieval_context: list[dict],
    ) -> JudgeRubric:
        if not self._model:
            return _dummy_score(generated_reply)

        if not generated_reply or generated_reply.strip() == "":
            return JudgeRubric(
                relevance=3, groundedness=5, helpfulness=3, tone=3, no_hallucination=5,
                pass_fail="pass",
                reason="Message was correctly escalated; no reply was drafted.",
            )

        context_text = "\n".join(
            f"- SpotifyCares said: {ctx['brand_reply']}" for ctx in retrieval_context
        )

        user_prompt = (
            f"Customer message:\n{customer_message}\n\n"
            f"Generated reply (to evaluate):\n{generated_reply}\n\n"
            f"Reference reply (historical SpotifyCares response):\n{reference_reply}\n\n"
            f"Retrieved historical context used by the system:\n{context_text}"
        )

        retries = 5
        for attempt in range(retries):
            try:
                response = self._model.generate_content(user_prompt)
                data = json.loads(response.text)
                return JudgeRubric(**data)
            except Exception as exc:
                if "429" in str(exc):
                    if attempt < retries - 1:
                        logger.warning("Rate limit hit (429). Retrying in 16s...")
                        time.sleep(16)
                        continue
                logger.error("Gemini judge error: %s", exc)
                break
                
        return JudgeRubric(
            relevance=1, groundedness=1, helpfulness=1, tone=1, no_hallucination=1,
            pass_fail="fail",
            reason="API failure during evaluation.",
        )
