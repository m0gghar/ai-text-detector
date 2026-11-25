"""Unit test for zero benchmark contamination against training data."""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.load_external_benchmarks import ContaminationFilter, FORMAL_HUMAN_CORPUS


def test_contamination_filter_on_formal_human():
    """Verify that formal human samples have no exact or near-duplicate matches
    in the training set at the 0.80 Jaccard threshold.
    """
    train_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    if not os.path.exists(train_path):
        pytest.skip("train.csv not available for contamination check.")

    c_filter = ContaminationFilter(train_path, max_train_samples=5000, random_seed=42)

    for item in FORMAL_HUMAN_CORPUS:
        text = item["text"]
        is_contam, reason = c_filter.is_contaminated(text)
        assert is_contam is False, f"Formal text unexpectedly matched training data via {reason}."
