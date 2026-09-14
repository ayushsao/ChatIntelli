"""
scripts/run_all.py

One-shot script to reproduce all headline results in under 15 minutes.

Steps:
  1. Check prerequisites (data files exist)
  2. Run baseline evaluation
  3. Run AI agent evaluation (limited to --limit examples)
  4. Run human-judge agreement analysis
  5. Print the final comparison table

Usage:
    python scripts/run_all.py              # Full evaluation (200 examples)
    python scripts/run_all.py --limit 50   # Quick evaluation (~3 minutes)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    required = [
        Path("data/train_set.csv"),
        Path("data/golden/golden_set.jsonl"),
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.error("Missing required data files:")
        for p in missing:
            logger.error("  %s", p)
        logger.error("Please run:")
        logger.error("  python scripts/prepare_data.py")
        logger.error("  python scripts/build_golden_set.py")
        return False
    return True


def run_step(name: str, cmd: list[str]) -> bool:
    logger.info("Running: %s", " ".join(cmd))
    t0 = time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time() - t0
    if result.returncode != 0:
        logger.error("❌ %s failed (%.1fs)", name, elapsed)
        return False
    logger.info("✅ %s done (%.1fs)", name, elapsed)
    return True


def print_comparison_table() -> None:
    print("\n" + "=" * 72)
    print("HEADLINE RESULTS")
    print("=" * 72)
    header = f"{'System':<35} {'Intent MacroF1':>14} {'Esc F1':>8} {'Reply Judge':>12}"
    print(header)
    print("-" * 72)

    # Load baseline metrics
    baseline_path = Path("results/baseline_metrics.json")
    agent_path = Path("results/metrics.json")

    def load_json(p: Path) -> dict:
        if p.exists():
            with p.open() as f:
                return json.load(f)
        return {}

    bl = load_json(baseline_path)
    ag = load_json(agent_path)

    def fmt(d: dict, key: str, fmt_str: str = ".3f") -> str:
        v = d.get(key)
        return format(v, fmt_str) if v is not None else "N/A"

    # Majority
    maj = bl.get("majority_baseline", {})
    print(f"{'Majority baseline':<35} "
          f"{fmt(maj, 'intent_macro_f1'):>14} "
          f"{fmt(maj, 'escalation_f1'):>8} "
          f"{'N/A':>12}")

    # TF-IDF
    tfidf = bl.get("tfidf_baseline", {})
    print(f"{'TF-IDF + LR':<35} "
          f"{fmt(tfidf, 'intent_macro_f1'):>14} "
          f"{fmt(tfidf, 'escalation_f1'):>8} "
          f"{'N/A':>12}")

    # AI Agent
    intent_f1 = ag.get("intent", {}).get("macro_f1")
    esc_f1 = ag.get("escalation", {}).get("f1")
    judge_avg = ag.get("reply_judge", {}).get("avg_overall")

    def safe_fmt(v, fmt_str=".3f") -> str:
        return format(v, fmt_str) if v is not None else "N/A"

    print(f"{'AI Agent (GPT-4o-mini)':<35} "
          f"{safe_fmt(intent_f1):>14} "
          f"{safe_fmt(esc_f1):>8} "
          f"{safe_fmt(judge_avg, '.2f') + '/5.0' if judge_avg else 'N/A':>12}")

    print("=" * 72)
    print()
    print("FAR = False Auto-Handle Rate (lower is better):")
    print(f"  Majority:   {fmt(maj, 'false_auto_handle_rate', '.1%')}")
    print(f"  TF-IDF+LR:  {fmt(tfidf, 'false_auto_handle_rate', '.1%')}")
    esc = ag.get("escalation", {})
    far = esc.get("false_auto_handle_rate")
    print(f"  AI Agent:   {safe_fmt(far, '.1%') if far is not None else 'N/A'}")


def main(limit: int = 200) -> None:
    t_start = time()

    print("=" * 60)
    print("SpotifyCares AI Support Agent — Full Evaluation")
    print("=" * 60)

    if not check_prerequisites():
        sys.exit(1)

    Path("results").mkdir(exist_ok=True)

    ok = run_step("Baseline evaluation", [sys.executable, "scripts/run_baselines.py"])
    if not ok:
        sys.exit(1)

    ok = run_step(
        "Agent evaluation",
        [sys.executable, "scripts/run_evaluation.py", "--limit", str(limit)],
    )
    if not ok:
        logger.warning("Agent evaluation failed — showing baselines only.")

    print_comparison_table()

    elapsed = time() - t_start
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all headline results.")
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Max golden examples to use for agent evaluation (default: 200)"
    )
    args = parser.parse_args()
    main(limit=args.limit)
