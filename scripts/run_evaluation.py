"""
scripts/run_evaluation.py

Full evaluation harness — runs all three systems on the golden set and reports:
  - Intent classification: accuracy, macro F1, per-intent precision/recall/F1
  - Escalation: accuracy, precision, recall, F1, false-auto-handle rate
  - Reply quality: LLM judge scores (relevance, groundedness, helpfulness, tone, no_hallucination)
  - Confusion matrix (saved to results/confusion_matrix.png)
  - Full results (saved to results/evaluation_results.json)

Run from project root:
    python scripts/run_evaluation.py

To limit to N examples (faster):
    python scripts/run_evaluation.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.data import load_golden_set
from src.evaluation import LLMJudge
from src.pipeline import SupportPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_escalation_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict:
    """
    Convert action strings to binary (1=ESCALATE, 0=AUTO_HANDLE) and compute metrics.
    The 'false auto-handle rate' (FAR) is the fraction of true ESCALATEs that were
    incorrectly predicted as AUTO_HANDLE — the most dangerous error type.
    """
    bin_true = [1 if a == "ESCALATE" else 0 for a in y_true]
    bin_pred = [1 if a == "ESCALATE" else 0 for a in y_pred]

    false_auto = sum(1 for t, p in zip(bin_true, bin_pred) if t == 1 and p == 0)
    total_true_escalate = sum(bin_true)
    far = false_auto / max(total_true_escalate, 1)

    return {
        "accuracy": accuracy_score(bin_true, bin_pred),
        "precision": precision_score(bin_true, bin_pred, zero_division=0),
        "recall": recall_score(bin_true, bin_pred, zero_division=0),
        "f1": f1_score(bin_true, bin_pred, zero_division=0),
        "false_auto_handle_rate": far,
        "false_auto_handle_count": false_auto,
        "total_true_escalate": total_true_escalate,
    }


def save_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Intent Classification Confusion Matrix\n(AI Agent — GPT-4o-mini)")
        plt.tight_layout()
        Path("results").mkdir(exist_ok=True)
        fig.savefig("results/confusion_matrix.png", dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to results/confusion_matrix.png")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping confusion matrix.")


def main(limit: int = 200) -> None:
    logger.info("Loading golden set...")
    golden = load_golden_set()
    golden = golden[:limit]
    logger.info("Evaluating %d examples...", len(golden))

    pipeline = SupportPipeline()
    judge = LLMJudge()

    results = []
    y_true_intent: list[str] = []
    y_pred_intent: list[str] = []
    y_true_action: list[str] = []
    y_pred_action: list[str] = []
    judge_scores: list[float] = []

    for record in tqdm(golden, desc="Evaluating"):
        msg = record["customer_message"]

        # Run pipeline
        pipe_out = pipeline.process_message(msg)

        # Run LLM judge (only for AUTO_HANDLE with a non-empty reply)
        judge_result = None
        if pipe_out.action == "AUTO_HANDLE" and pipe_out.reply:
            judge_result = judge.evaluate_reply(
                customer_message=msg,
                generated_reply=pipe_out.reply,
                reference_reply=record.get("brand_reply_reference", ""),
                retrieval_context=pipe_out.retrieval_context,
            )
            judge_scores.append(judge_result.overall)

        y_true_intent.append(record["intent"])
        y_pred_intent.append(pipe_out.intent)
        y_true_action.append(record["expected_action"])
        y_pred_action.append(pipe_out.action)

        results.append({
            "id": record["id"],
            "customer_message": msg,
            "true_intent": record["intent"],
            "pred_intent": pipe_out.intent,
            "intent_confidence": pipe_out.intent_confidence,
            "true_action": record["expected_action"],
            "pred_action": pipe_out.action,
            "escalation_reason": pipe_out.escalation_reason,
            "generated_reply": pipe_out.reply,
            "retrieval_top_similarity": (
                pipe_out.retrieval_context[0]["similarity"]
                if pipe_out.retrieval_context else 0.0
            ),
            "judge": judge_result.model_dump() if judge_result else None,
        })

    # -----------------------------------------------------------------------
    # Print & save results
    # -----------------------------------------------------------------------
    all_labels = sorted(set(y_true_intent + y_pred_intent))

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_true_intent, y_pred_intent):.3f}")
    print(f"Macro F1: {f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0):.3f}")
    print("\nPer-class breakdown:")
    print(classification_report(y_true_intent, y_pred_intent, zero_division=0))

    esc_metrics = compute_escalation_metrics(y_true_action, y_pred_action)
    print("\n" + "=" * 60)
    print("ESCALATION DECISION")
    print("=" * 60)
    print(f"Accuracy:               {esc_metrics['accuracy']:.3f}")
    print(f"Escalation Precision:   {esc_metrics['precision']:.3f}")
    print(f"Escalation Recall:      {esc_metrics['recall']:.3f}")
    print(f"Escalation F1:          {esc_metrics['f1']:.3f}")
    print(f"False Auto-Handle Rate: {esc_metrics['false_auto_handle_rate']:.1%}")
    print(f"  (Wrongly AUTO_HANDLE when should ESCALATE: "
          f"{esc_metrics['false_auto_handle_count']} / {esc_metrics['total_true_escalate']})")

    print("\n" + "=" * 60)
    print("REPLY QUALITY (LLM Judge)")
    print("=" * 60)
    if judge_scores:
        avg = float(np.mean(judge_scores))
        print(f"Avg Overall Score: {avg:.2f} / 5.00 (n={len(judge_scores)} AUTO_HANDLE replies)")
        # per-dimension
        for dim in ["relevance", "groundedness", "helpfulness", "tone", "no_hallucination"]:
            vals = [r["judge"][dim] for r in results if r["judge"] is not None]
            if vals:
                print(f"  {dim:<22}: {np.mean(vals):.2f}")
    else:
        print("No AUTO_HANDLE replies to judge (all escalated or no API key).")

    # Save results
    Path("results").mkdir(exist_ok=True)

    summary = {
        "n_evaluated": len(golden),
        "intent": {
            "accuracy": accuracy_score(y_true_intent, y_pred_intent),
            "macro_f1": f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0),
        },
        "escalation": esc_metrics,
        "reply_judge": {
            "n_scored": len(judge_scores),
            "avg_overall": float(np.mean(judge_scores)) if judge_scores else None,
        },
    }

    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary metrics saved to results/metrics.json")

    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results saved to results/evaluation_results.json")

    save_confusion_matrix(y_true_intent, y_pred_intent, all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full agent evaluation.")
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Max number of golden examples to evaluate (default: 200)"
    )
    args = parser.parse_args()
    main(limit=args.limit)
