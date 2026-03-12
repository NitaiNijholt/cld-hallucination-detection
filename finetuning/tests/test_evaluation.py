"""Tests for evaluation utilities."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.src.evaluation.eval_utils import INCORRECT, VERDICT_LABELS, parse_verdict


def test_parse_verdict_correct():
    assert parse_verdict("REASON: x\nVERDICT: CORRECT") == "CORRECT"
    assert parse_verdict("VERDICT: CORRECT") == "CORRECT"
    assert parse_verdict("verdict=correct") == "CORRECT"


def test_parse_verdict_partially_correct():
    assert parse_verdict("REASON: y\nVERDICT: PARTIALLY_CORRECT") == "PARTIALLY_CORRECT"
    assert parse_verdict("VERDICT: PARTIALLY_CORRECT") == "PARTIALLY_CORRECT"


def test_parse_verdict_incorrect():
    assert parse_verdict("REASON: z\nVERDICT: INCORRECT") == "INCORRECT"
    assert parse_verdict("verdict: incorrect") == "INCORRECT"


def test_parse_verdict_unknown():
    assert parse_verdict("garbage output") == "UNKNOWN"
    assert parse_verdict("VERDICT: INVALID") == "UNKNOWN"


def test_binary_label():
    """INCORRECT = 1 (hallucination positive), others = 0."""
    assert (1 if INCORRECT == "INCORRECT" else 0) == 1
    # Sanity: INCORRECT is the positive class for AUC
    assert INCORRECT in VERDICT_LABELS


def test_parse_verdict_case_insensitive():
    assert parse_verdict("Verdict: correct") == "CORRECT"
    assert parse_verdict("VERDICT = partially_correct") == "PARTIALLY_CORRECT"


def test_parse_verdict_whitespace_variants():
    assert parse_verdict("VERDICT:  CORRECT") == "CORRECT"
    assert parse_verdict("VERDICT=INCORRECT") == "INCORRECT"
