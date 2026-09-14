"""
scripts/run_baselines.py

Evaluates two baselines on the golden set and prints a comparison table:

Baseline 1 — Majority classifier (trivial): always predicts the most common class.
Baseline 2 — TF-IDF + Logistic Regression (simple): trained on pseudo-labeled data.

Both baselines use the same keyword heuristic labels as the golden set.
See docs/decision_log.md Decision #9 for why this inflates TF-IDF metrics.

Usage (from project root):
    python scripts/run_baselines.py

Results are printed to stdout and appended to results/baseline_metrics.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from baselines.majority import MajorityClassifier
from baselines.tfidf_classifier import TfIdfBaseline
from src.data import load_golden_set, load_train_set
from scripts.build_golden_set import assign_heuristic_intent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def escalation_from_intent(intent: str) -> str:
    """Baseline rule: escalate payment/login/other, auto-handle the rest."""
    return "ESCALATE" if intent in {"account_login", "payment_issue", "other"} else "AUTO_HANDLE"


def compute_escalation_metrics(y_true: list[str], y_pred_intent: list[str]) -> dict:
    y_pred_action = [escalation_from_intent(i) for i in y_pred_intent]
    bin_true = [1 if a == "ESCALATE" else 0 for a in y_true]
    bin_pred = [1 if a == "ESCALATE" else 0 for a in y_pred_action]

    false_auto = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 0)
    total_escalate = sum(bin_true)

    return {
        "accuracy": accuracy_score(bin_true, bin_pred),
        "f1": f1_score(bin_true, bin_pred, zero_division=0),
        "false_auto_handle_rate": false_auto / max(total_escalate, 1),
    }


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main() -> None:
    logger.info("Loading data...")
    train_df = load_train_set()
    train_df = train_df.copy()
    train_df["intent"] = train_df["customer_message_cleaned"].fillna("").apply(assign_heuristic_intent)

    golden = load_golden_set()
    golden_X = [r["customer_message"] for r in golden]
    golden_y_intent = [r["intent"] for r in golden]
    golden_y_action = [r["expected_action"] for r in golden]

    train_X = train_df["customer_message_cleaned"].fillna("")
    train_y = train_df["intent"]

    results: dict = {}

    # -----------------------------------------------------------------------
    # Baseline 1: Majority classifier
    # -----------------------------------------------------------------------
    print_section("BASELINE 1: Majority (Trivial)")
    maj = MajorityClassifier()
    maj.fit(train_X.tolist(), train_y.tolist())
    maj_preds = maj.predict(golden_X)

    maj_acc = accuracy_score(golden_y_intent, maj_preds)
    maj_f1 = f1_score(golden_y_intent, maj_preds, average="macro", zero_division=0)
    maj_esc = compute_escalation_metrics(golden_y_action, maj_preds)

    print(f"Majority class: '{maj.majority_class}'")
    print(f"Intent Accuracy:     {maj_acc:.3f}")
    print(f"Intent Macro F1:     {maj_f1:.3f}")
    print(f"Esc F1:              {maj_esc['f1']:.3f}")
    print(f"False Auto-Handle Rate: {maj_esc['false_auto_handle_rate']:.1%}")
    print(classification_report(golden_y_intent, maj_preds, zero_division=0))

    results["majority_baseline"] = {
        "intent_accuracy": maj_acc,
        "intent_macro_f1": maj_f1,
        "escalation_f1": maj_esc["f1"],
        "false_auto_handle_rate": maj_esc["false_auto_handle_rate"],
    }

    # -----------------------------------------------------------------------
    # Baseline 2: TF-IDF + Logistic Regression
    # -----------------------------------------------------------------------
    print_section("BASELINE 2: TF-IDF + Logistic Regression")
    tfidf = TfIdfBaseline()
    tfidf.fit(train_X.tolist(), train_y.tolist())
    tfidf_preds = tfidf.predict(golden_X)

    tfidf_acc = accuracy_score(golden_y_intent, tfidf_preds)
    tfidf_f1 = f1_score(golden_y_intent, tfidf_preds, average="macro", zero_division=0)
    tfidf_esc = compute_escalation_metrics(golden_y_action, tfidf_preds)

    print(f"Intent Accuracy:     {tfidf_acc:.3f}")
    print(f"Intent Macro F1:     {tfidf_f1:.3f}")
    print(f"Esc F1:              {tfidf_esc['f1']:.3f}")
    print(f"False Auto-Handle Rate: {tfidf_esc['false_auto_handle_rate']:.1%}")
    print(classification_report(golden_y_intent, tfidf_preds, zero_division=0))

    results["tfidf_baseline"] = {
        "intent_accuracy": tfidf_acc,
        "intent_macro_f1": tfidf_f1,
        "escalation_f1": tfidf_esc["f1"],
        "false_auto_handle_rate": tfidf_esc["false_auto_handle_rate"],
    }

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    Path("results").mkdir(exist_ok=True)
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Baseline metrics saved to results/baseline_metrics.json")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print_section("COMPARISON SUMMARY")
    print(f"{'System':<35} {'Intent Macro F1':>16} {'Esc F1':>8} {'FAR':>8}")
    print("-" * 70)
    print(f"{'Majority baseline':<35} {maj_f1:>16.3f} {maj_esc['f1']:>8.3f} {maj_esc['false_auto_handle_rate']:>8.1%}")
    print(f"{'TF-IDF + LR':<35} {tfidf_f1:>16.3f} {tfidf_esc['f1']:>8.3f} {tfidf_esc['false_auto_handle_rate']:>8.1%}")
    print(f"{'AI Agent (see run_evaluation.py)':<35} {'see results/metrics.json':>16}")


if __name__ == "__main__":
    main()
