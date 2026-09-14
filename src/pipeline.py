"""
src/pipeline.py

Main orchestration pipeline for the SpotifyCares AI support agent.

Flow:
    Customer Message
        ↓
    Intent Classifier (LLM or keyword heuristic fallback)
        ↓
    Historical Retrieval (TF-IDF cosine similarity)
        ↓
    Reply Generator + Escalation Decision (LLM or rule-based fallback)
        ↓
    Structured Final Response

All components are independently testable and swappable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.classifier import ProposedClassifier
from src.retrieval import RetrievalSystem
from src.generator import ReplyGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured output from the support pipeline."""
    message: str
    intent: str
    intent_confidence: float
    intent_reason: str
    action: str                          # "AUTO_HANDLE" | "ESCALATE"
    escalation_reason: str
    escalation_confidence: float
    reply: str
    retrieval_context: list[dict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


class SupportPipeline:
    """
    End-to-end support pipeline combining classification, retrieval, and generation.

    Args:
        data_path: Path to the training/retrieval CSV (default: data/train_set.csv)
    """

    def __init__(self, data_path: str = "data/train_set.csv") -> None:
        logger.info("Initializing SupportPipeline...")
        self.classifier = ProposedClassifier()
        self.retrieval = RetrievalSystem(data_path=data_path)
        self.generator = ReplyGenerator()
        logger.info("SupportPipeline ready.")

    def process_message(self, message: str, top_k: int = 3) -> PipelineResult:
        """
        Process a single customer message through the full pipeline.

        Args:
            message: Raw customer message text
            top_k: Number of historical examples to retrieve

        Returns:
            PipelineResult with all intermediate and final outputs
        """
        # Step 1: Classify intent
        classification = self.classifier.predict_full(message)
        intent = classification.intent.value
        intent_confidence = classification.confidence
        intent_reason = classification.reason

        logger.debug("Classified '%s...' as %s (conf=%.2f)", message[:50], intent, intent_confidence)

        # Step 2: Retrieve historical examples
        context = self.retrieval.retrieve_similar(message, top_k=top_k)

        logger.debug("Retrieved %d examples (top sim=%.2f)", len(context),
                     context[0]["similarity"] if context else 0.0)

        # Step 3: Generate reply and escalation decision
        response = self.generator.generate(message, intent, context)

        logger.debug("Action=%s (conf=%.2f)", response.action, response.confidence)

        return PipelineResult(
            message=message,
            intent=intent,
            intent_confidence=intent_confidence,
            intent_reason=intent_reason,
            action=response.action,
            escalation_reason=response.reason,
            escalation_confidence=response.confidence,
            reply=response.reply,
            retrieval_context=context,
            unsupported_claims=response.unsupported_claims,
        )
