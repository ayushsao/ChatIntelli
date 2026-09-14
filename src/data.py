"""
src/data.py

Data loading utilities for SpotifyCares support agent.
All paths are relative to the project root.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = "data/train_set.csv"
DEFAULT_GOLDEN_PATH = "data/golden/golden_set.jsonl"


def load_train_set(path: str = DEFAULT_TRAIN_PATH) -> pd.DataFrame:
    """
    Load the retrieval / training split CSV.
    Raises FileNotFoundError with a clear message if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Training set not found at '{path}'. "
            "Run: python scripts/prepare_data.py && python scripts/build_golden_set.py"
        )
    df = pd.read_csv(p)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def load_golden_set(path: str = DEFAULT_GOLDEN_PATH) -> list[dict[str, Any]]:
    """
    Load the golden evaluation set from JSONL format.
    Each line must be a valid JSON object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Golden set not found at '{path}'. "
            "Run: python scripts/build_golden_set.py"
        )
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)

    logger.info("Loaded %d golden examples from %s", len(records), path)
    return records
