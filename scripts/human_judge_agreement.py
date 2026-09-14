"""
scripts/human_judge_agreement.py

Generates a human annotation worksheet from a 40-example subset of the golden set,
then (after human scores are filled in) computes agreement statistics:

  - Pearson correlation (LLM judge vs human)
  - Cohen's kappa on pass/fail labels
  - Mean absolute error per dimension
  - Identified disagreement examples

Step 1 — Generate worksheet (requires API key for LLM judge scores):
    python scripts/human_judge_agreement.py --generate

This creates: results/human_judge_worksheet.json
  Each record has: customer_message, generated_reply, llm_scores, human_scores (empty)

Step 2 — Fill in human_scores in the worksheet JSON manually:
  For each record, populate "human_scores" with:
    {"relevance": 1-5, "groundedness": 1-5, "helpfulness": 1-5, "tone": 1-5, "no_hallucination": 1-5}

Step 3 — Compute agreement:
    python scripts/human_judge_agreement.py --analyze

This prints correlation/kappa and saves results/judge_agreement.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WORKSHEET_PATH = Path("results/human_judge_worksheet.json")
AGREEMENT_PATH = Path("results/judge_agreement.json")
DIMENSIONS = ["relevance", "groundedness", "helpfulness", "tone", "no_hallucination"]


def generate_worksheet(n: int = 40) -> None:
    """
    Run the pipeline + LLM judge on n examples and write a worksheet
    with empty human_scores slots for filling in.
    """
    from src.data import load_golden_set
    from src.pipeline import SupportPipeline
    from src.evaluation import LLMJudge

    logger.info("Loading golden set and pipeline...")
    golden = load_golden_set()
    # Pick examples that are likely to get an AUTO_HANDLE reply (better for judging)
    non_escalate = [r for r in golden if r["expected_action"] == "AUTO_HANDLE"]
    subset = non_escalate[:n]
    logger.info("Using %d AUTO_HANDLE examples for agreement study.", len(subset))

    pipeline = SupportPipeline()
    judge = LLMJudge()

    worksheet = []
    from tqdm import tqdm
    for record in tqdm(subset, desc="Generating worksheet"):
        msg = record["customer_message"]
        pipe_out = pipeline.process_message(msg)

        llm_scores = None
        if pipe_out.action == "AUTO_HANDLE" and pipe_out.reply:
            rubric = judge.evaluate_reply(
                customer_message=msg,
                generated_reply=pipe_out.reply,
                reference_reply=record.get("brand_reply_reference", ""),
                retrieval_context=pipe_out.retrieval_context,
            )
            llm_scores = {dim: getattr(rubric, dim) for dim in DIMENSIONS}
            llm_scores["pass_fail"] = rubric.pass_fail
            llm_scores["reason"] = rubric.reason

        worksheet.append({
            "id": record["id"],
            "customer_message": msg,
            "generated_reply": pipe_out.reply,
            "action": pipe_out.action,
            "reference_reply": record.get("brand_reply_reference", ""),
            "llm_scores": llm_scores,
            "human_scores": None,  # ← Fill this in manually
        })

    Path("results").mkdir(exist_ok=True)
    with WORKSHEET_PATH.open("w", encoding="utf-8") as f:
        json.dump(worksheet, f, indent=2)

    scored = sum(1 for w in worksheet if w["llm_scores"] is not None)
    logger.info("Worksheet saved to %s", WORKSHEET_PATH)
    logger.info("%d / %d examples have LLM scores (rest were escalated).", scored, len(worksheet))
    print(f"\nNext step: open {WORKSHEET_PATH} and fill in 'human_scores' for each")
    print("example that has non-null 'llm_scores'. Use the same 1–5 scale:")
    print("  relevance, groundedness, helpfulness, tone, no_hallucination")
    print(f"Then run: python scripts/human_judge_agreement.py --analyze")


def analyze_agreement() -> None:
    """
    Compute LLM judge vs human agreement from a filled-in worksheet.
    """
    if not WORKSHEET_PATH.exists():
        logger.error("Worksheet not found at %s. Run --generate first.", WORKSHEET_PATH)
        sys.exit(1)

    with WORKSHEET_PATH.open("r", encoding="utf-8") as f:
        worksheet = json.load(f)

    # Filter to examples with both scores
    paired = [
        w for w in worksheet
        if w["llm_scores"] is not None and w["human_scores"] is not None
    ]

    if len(paired) < 5:
        logger.error(
            "Only %d examples have both LLM and human scores. "
            "Please fill in more human_scores in the worksheet.", len(paired)
        )
        sys.exit(1)

    logger.info("Analyzing %d paired examples...", len(paired))

    try:
        import numpy as np
        from scipy.stats import pearsonr
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        logger.error("scipy and sklearn required. Run: pip install scipy scikit-learn")
        sys.exit(1)

    results: dict = {"n": len(paired), "dimensions": {}}

    print(f"\n{'=' * 60}")
    print(f"LLM Judge vs Human Agreement  (n={len(paired)})")
    print(f"{'=' * 60}")
    print(f"{'Dimension':<22} {'Pearson r':>10} {'MAE':>6} {'Human µ':>8} {'LLM µ':>8}")
    print("-" * 60)

    all_llm = []
    all_human = []

    for dim in DIMENSIONS:
        llm_vals = np.array([w["llm_scores"][dim] for w in paired])
        human_vals = np.array([w["human_scores"][dim] for w in paired])

        r, p_val = pearsonr(llm_vals, human_vals)
        mae = float(np.mean(np.abs(llm_vals - human_vals)))

        results["dimensions"][dim] = {
            "pearson_r": float(r),
            "p_value": float(p_val),
            "mae": mae,
            "llm_mean": float(np.mean(llm_vals)),
            "human_mean": float(np.mean(human_vals)),
        }

        print(f"{dim:<22} {r:>10.3f} {mae:>6.2f} {np.mean(human_vals):>8.2f} {np.mean(llm_vals):>8.2f}")

        all_llm.extend(llm_vals.tolist())
        all_human.extend(human_vals.tolist())

    # Overall correlation across all dimensions
    r_all, _ = pearsonr(np.array(all_llm), np.array(all_human))
    print(f"\n{'Overall correlation (all dims)':<22} {r_all:>10.3f}")
    results["overall_pearson_r"] = float(r_all)

    # Pass/fail kappa (only if human_scores has pass_fail)
    has_pf = [w for w in paired if "pass_fail" in (w.get("human_scores") or {})]
    if has_pf:
        llm_pf = [1 if w["llm_scores"]["pass_fail"] == "pass" else 0 for w in has_pf]
        human_pf = [1 if w["human_scores"]["pass_fail"] == "pass" else 0 for w in has_pf]
        kappa = cohen_kappa_score(human_pf, llm_pf)
        print(f"\nCohen's kappa (pass/fail): {kappa:.3f}")
        results["cohen_kappa_pass_fail"] = float(kappa)

    # Agreement rate (within 1 point) per dimension
    print(f"\n{'Dimension':<22} {'Agreement ±1':>12}")
    print("-" * 35)
    for dim in DIMENSIONS:
        llm_vals = np.array([w["llm_scores"][dim] for w in paired])
        human_vals = np.array([w["human_scores"][dim] for w in paired])
        within_1 = float(np.mean(np.abs(llm_vals - human_vals) <= 1))
        print(f"{dim:<22} {within_1:>12.1%}")
        results["dimensions"][dim]["agreement_within_1"] = within_1

    # Show worst disagreements
    print(f"\nTop 3 largest disagreements (overall score):")
    for w in sorted(paired, key=lambda x: abs(
        sum(x["llm_scores"][d] for d in DIMENSIONS) -
        sum(x["human_scores"][d] for d in DIMENSIONS)
    ), reverse=True)[:3]:
        llm_avg = np.mean([w["llm_scores"][d] for d in DIMENSIONS])
        hum_avg = np.mean([w["human_scores"][d] for d in DIMENSIONS])
        print(f"  LLM={llm_avg:.1f} | Human={hum_avg:.1f}")
        print(f"  Message: {w['customer_message'][:80]}...")
        print(f"  Reply:   {w['generated_reply'][:80]}...")

    with AGREEMENT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Agreement results saved to %s", AGREEMENT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true",
                       help="Generate LLM judge scores and empty worksheet")
    group.add_argument("--analyze", action="store_true",
                       help="Analyze human vs LLM agreement from filled worksheet")
    parser.add_argument("--n", type=int, default=40,
                        help="Number of examples for worksheet (default: 40)")
    args = parser.parse_args()

    if args.generate:
        generate_worksheet(n=args.n)
    else:
        analyze_agreement()


if __name__ == "__main__":
    main()
