"""
scripts/run_agent.py

Interactive / single-message demonstration of the AI support pipeline.

Usage:
    # Process one message from command line:
    python scripts/run_agent.py --message "My app keeps crashing on iOS 17"

    # Interactive mode:
    python scripts/run_agent.py --interactive
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.pipeline import SupportPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def display_result(result) -> None:
    print("\n" + "─" * 60)
    print(f"Customer: {result.message}")
    print("─" * 60)
    print(f"Intent:        {result.intent}  (confidence={result.intent_confidence:.0%})")
    print(f"Reason:        {result.intent_reason}")
    print()
    print(f"Action:        {result.action}  (confidence={result.escalation_confidence:.0%})")
    print(f"Reason:        {result.escalation_reason}")

    if result.retrieval_context:
        top = result.retrieval_context[0]
        print(f"\nTop retrieved example (sim={top['similarity']:.2f}):")
        print(f"  Customer: {top['customer_message'][:80]}...")
        print(f"  Reply:    {top['brand_reply'][:80]}...")

    if result.action == "AUTO_HANDLE" and result.reply:
        print(f"\n💬 Drafted Reply:\n{result.reply}")
        if result.unsupported_claims:
            print(f"\n⚠ Unsupported claims flagged: {result.unsupported_claims}")
    else:
        print("\n🔴 ESCALATE — forwarding to human agent.")
    print("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SpotifyCares AI support agent.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", type=str, help="Single customer message to process")
    group.add_argument("--interactive", action="store_true", help="Interactive multi-turn mode")
    args = parser.parse_args()

    print("🎵 SpotifyCares AI Support Agent")
    print("Loading pipeline (this may take a few seconds)...")
    pipeline = SupportPipeline()
    print("Pipeline ready.\n")

    if args.message:
        result = pipeline.process_message(args.message)
        display_result(result)
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                msg = input("Customer > ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if msg.lower() in {"quit", "exit", "q"}:
                break
            if not msg:
                continue
            result = pipeline.process_message(msg)
            display_result(result)


if __name__ == "__main__":
    main()
