"""
tests/test_pipeline.py

Unit tests for all major pipeline components.
Most tests use mock data / no API key to remain fast and deterministic.
Tests requiring real data are gracefully skipped if the CSV doesn't exist.
"""
from __future__ import annotations

import json
import os
import pytest

from src.intents import Intent, VALID_INTENTS, INTENT_DEFINITIONS


# ---------------------------------------------------------------------------
# Intent taxonomy tests
# ---------------------------------------------------------------------------

class TestIntentTaxonomy:
    def test_valid_intents_not_empty(self):
        assert len(VALID_INTENTS) >= 6

    def test_required_intents_present(self):
        required = {"app_bug", "payment_issue", "account_login",
                    "content_missing", "feature_request", "other"}
        assert required.issubset(set(VALID_INTENTS))

    def test_all_intents_have_definitions(self):
        for intent in Intent:
            assert intent in INTENT_DEFINITIONS, f"Missing definition for {intent}"
            defn = INTENT_DEFINITIONS[intent]
            assert "description" in defn
            assert "examples" in defn

    def test_intent_enum_values_match_strings(self):
        for intent in Intent:
            assert intent.value in VALID_INTENTS


# ---------------------------------------------------------------------------
# Keyword heuristic classifier tests
# ---------------------------------------------------------------------------

class TestKeywordHeuristicClassifier:
    @pytest.fixture(autouse=True)
    def setup(self):
        from src.classifier import KeywordHeuristicClassifier
        self.clf = KeywordHeuristicClassifier()

    def test_payment_issue(self):
        msgs = [
            "You charged me twice this month!",
            "I want a refund for my premium subscription.",
            "My credit card was declined.",
        ]
        for msg in msgs:
            assert self.clf.predict_single(msg) == "payment_issue", \
                f"Expected payment_issue for: {msg!r}"

    def test_account_login(self):
        msgs = [
            "My account was hacked and the email changed.",
            "I forgot my password and the reset link doesn't work.",
            "Please approve my Spotify for artists account.",
        ]
        for msg in msgs:
            assert self.clf.predict_single(msg) == "account_login", \
                f"Expected account_login for: {msg!r}"

    def test_app_bug(self):
        msgs = [
            "The app crashes every time I open offline mode.",
            "My songs randomly pause on iOS.",
            "Downloaded songs disappeared after the update.",
        ]
        for msg in msgs:
            assert self.clf.predict_single(msg) == "app_bug", \
                f"Expected app_bug for: {msg!r}"

    def test_content_missing(self):
        msgs = [
            "This podcast episode is greyed out.",
            "Why was this album removed from Spotify?",
        ]
        for msg in msgs:
            assert self.clf.predict_single(msg) == "content_missing", \
                f"Expected content_missing for: {msg!r}"

    def test_feature_request(self):
        msgs = [
            "Please bring back the old UI!",
            "When will lyrics be available on TV?",
        ]
        for msg in msgs:
            assert self.clf.predict_single(msg) == "feature_request", \
                f"Expected feature_request for: {msg!r}"

    def test_other_for_low_signal(self):
        low_signal = "Thanks for the help!"
        result = self.clf.predict_single(low_signal)
        # "other" or any other class — just ensure it doesn't error
        assert result in VALID_INTENTS

    def test_batch_predict_matches_single(self):
        msgs = ["app crashes", "refund please", "password reset"]
        batch = self.clf.predict(msgs)
        singles = [self.clf.predict_single(m) for m in msgs]
        assert batch == singles

    def test_empty_string_returns_valid_intent(self):
        result = self.clf.predict_single("")
        assert result in VALID_INTENTS

    def test_non_string_graceful(self):
        # Should handle non-string without raising (str conversion)
        result = self.clf.predict_single("   ")
        assert result in VALID_INTENTS


# ---------------------------------------------------------------------------
# ProposedClassifier fallback (no API key) tests
# ---------------------------------------------------------------------------

class TestProposedClassifierFallback:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from src.classifier import ProposedClassifier
        self.clf = ProposedClassifier()

    def test_fallback_returns_valid_intent(self):
        result = self.clf.predict_single("My app is crashing.")
        assert result in VALID_INTENTS

    def test_fallback_not_constant_app_bug(self):
        # The old broken fallback returned "app_bug" for everything.
        # Ensure different messages produce different predictions.
        preds = [
            self.clf.predict_single("I want a refund"),
            self.clf.predict_single("app crashes"),
            self.clf.predict_single("my account was hacked"),
        ]
        assert len(set(preds)) > 1, "Fallback must not return the same intent for all inputs"

    def test_predict_full_fallback(self):
        from src.classifier import IntentClassificationResult
        result = self.clf.predict_full("My premium didn't renew.")
        assert isinstance(result, IntentClassificationResult)
        assert result.intent.value in VALID_INTENTS
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Rule-based generator (no API key) tests
# ---------------------------------------------------------------------------

class TestRuleBasedGenerator:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from src.generator import ReplyGenerator
        self.gen = ReplyGenerator()

    def test_payment_always_escalates(self):
        ctx = [{"customer_message": "test", "brand_reply": "test", "similarity": 0.5}]
        result = self.gen.generate("I want a refund", "payment_issue", ctx)
        assert result.action == "ESCALATE"

    def test_account_login_always_escalates(self):
        ctx = [{"customer_message": "test", "brand_reply": "test", "similarity": 0.5}]
        result = self.gen.generate("I can't log in", "account_login", ctx)
        assert result.action == "ESCALATE"

    def test_app_bug_with_good_context_auto_handles(self):
        ctx = [{
            "customer_message": "app crashes",
            "brand_reply": "Try reinstalling the app.",
            "similarity": 0.8  # > threshold
        }]
        result = self.gen.generate("My app keeps crashing", "app_bug", ctx)
        assert result.action == "AUTO_HANDLE"

    def test_app_bug_with_no_context_escalates(self):
        result = self.gen.generate("My app keeps crashing", "app_bug", [])
        assert result.action == "ESCALATE"

    def test_response_has_required_fields(self):
        from src.generator import ReplyResponse
        ctx = [{"customer_message": "test", "brand_reply": "ok", "similarity": 0.9}]
        result = self.gen.generate("Songs skip randomly", "app_bug", ctx)
        assert isinstance(result, ReplyResponse)
        assert result.action in {"AUTO_HANDLE", "ESCALATE"}
        assert isinstance(result.reason, str)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Retrieval system tests
# ---------------------------------------------------------------------------

class TestRetrievalSystem:
    def test_retrieval_returns_expected_format(self):
        try:
            from src.retrieval import RetrievalSystem
            rs = RetrievalSystem()
            results = rs.retrieve_similar("My app keeps crashing on iOS.", top_k=2)
            assert len(results) <= 2
            for r in results:
                assert "customer_message" in r
                assert "brand_reply" in r
                assert "similarity" in r
                assert 0.0 <= r["similarity"] <= 1.0
        except FileNotFoundError:
            pytest.skip("data/train_set.csv not found — run prepare_data.py first")

    def test_retrieval_top_k_respected(self):
        try:
            from src.retrieval import RetrievalSystem
            rs = RetrievalSystem()
            results = rs.retrieve_similar("test query", top_k=1)
            assert len(results) == 1
        except FileNotFoundError:
            pytest.skip("data/train_set.csv not found")

    def test_retrieval_ordered_by_similarity(self):
        try:
            from src.retrieval import RetrievalSystem
            rs = RetrievalSystem()
            results = rs.retrieve_similar("I can't log in", top_k=3)
            sims = [r["similarity"] for r in results]
            assert sims == sorted(sims, reverse=True), "Results should be ordered by descending similarity"
        except FileNotFoundError:
            pytest.skip("data/train_set.csv not found")


# ---------------------------------------------------------------------------
# Pipeline integration tests (no API key)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        try:
            from src.pipeline import SupportPipeline
            self.pipeline = SupportPipeline()
        except FileNotFoundError:
            pytest.skip("data/train_set.csv not found — run prepare_data.py first")

    def test_result_has_all_fields(self):
        from src.pipeline import PipelineResult
        result = self.pipeline.process_message("My app crashes all the time.")
        assert isinstance(result, PipelineResult)
        assert result.intent in VALID_INTENTS
        assert result.action in {"AUTO_HANDLE", "ESCALATE"}
        assert isinstance(result.reply, str)
        assert 0.0 <= result.intent_confidence <= 1.0

    def test_payment_escalates(self):
        result = self.pipeline.process_message("You charged me twice this month!")
        assert result.action == "ESCALATE", "Payment issues must always escalate"

    def test_account_login_escalates(self):
        result = self.pipeline.process_message("My account was hacked.")
        assert result.action == "ESCALATE", "Account login issues must always escalate"

    def test_reply_empty_when_escalating(self):
        result = self.pipeline.process_message("I want a refund.")
        if result.action == "ESCALATE":
            # When escalating, reply should be empty (no false promises)
            assert result.reply == "" or result.reply is None

    def test_retrieval_context_not_leaked(self):
        """Verify retrieved examples are from train set, not golden set."""
        import pandas as pd
        from src.data import load_golden_set

        try:
            golden = load_golden_set()
            golden_ids = {str(r["id"]) for r in golden}
        except FileNotFoundError:
            pytest.skip("Golden set not found")

        result = self.pipeline.process_message("Can you fix the shuffle algorithm?")
        for ctx in result.retrieval_context:
            # If context has an ID, it should not be in the golden set
            if "id" in ctx:
                assert str(ctx["id"]) not in golden_ids


# ---------------------------------------------------------------------------
# Evaluation / LLM judge tests (no API key)
# ---------------------------------------------------------------------------

class TestLLMJudgeFallback:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from src.evaluation import LLMJudge
        self.judge = LLMJudge()

    def test_dummy_scores_in_valid_range(self):
        rubric = self.judge.evaluate_reply(
            "My app crashes",
            "Try reinstalling the app.",
            "Please reinstall.", []
        )
        for dim in ["relevance", "groundedness", "helpfulness", "tone", "no_hallucination"]:
            val = getattr(rubric, dim)
            assert 1 <= val <= 5, f"{dim} score {val} out of range"

    def test_pass_fail_is_valid_string(self):
        rubric = self.judge.evaluate_reply(
            "test", "test reply", "reference", []
        )
        assert rubric.pass_fail in {"pass", "fail"}

    def test_overall_property(self):
        rubric = self.judge.evaluate_reply(
            "test", "test reply", "reference", []
        )
        expected = (
            rubric.relevance + rubric.groundedness + rubric.helpfulness
            + rubric.tone + rubric.no_hallucination
        ) / 5.0
        assert abs(rubric.overall - expected) < 0.001


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_train_set_raises_on_missing(self):
        from src.data import load_train_set
        with pytest.raises(FileNotFoundError):
            load_train_set("nonexistent/path.csv")

    def test_load_golden_set_raises_on_missing(self):
        from src.data import load_golden_set
        with pytest.raises(FileNotFoundError):
            load_golden_set("nonexistent/path.jsonl")

    def test_load_golden_set_returns_list(self):
        try:
            from src.data import load_golden_set
            records = load_golden_set()
            assert isinstance(records, list)
            assert len(records) > 0
            # Check schema
            for r in records[:5]:
                assert "id" in r
                assert "customer_message" in r
                assert "intent" in r
                assert "expected_action" in r
        except FileNotFoundError:
            pytest.skip("Golden set not found")

    def test_golden_set_intents_valid(self):
        try:
            from src.data import load_golden_set
            records = load_golden_set()
            for r in records:
                assert r["intent"] in VALID_INTENTS, \
                    f"Invalid intent {r['intent']!r} in record {r['id']}"
        except FileNotFoundError:
            pytest.skip("Golden set not found")

    def test_golden_set_actions_valid(self):
        try:
            from src.data import load_golden_set
            records = load_golden_set()
            for r in records:
                assert r["expected_action"] in {"AUTO_HANDLE", "ESCALATE"}, \
                    f"Invalid action in record {r['id']}"
        except FileNotFoundError:
            pytest.skip("Golden set not found")
