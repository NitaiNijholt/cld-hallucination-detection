"""Unit tests for data preparation helpers."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.src.data.prepare_judge_data import (
    _build_rows,
    _extract_domain,
    _extract_reason,
    _split_by_edge_identity,
)


def test_extract_domain():
    assert _extract_domain("/path/to/depressive/edge.xlsx") == "depressive"
    assert _extract_domain("/path/emergency_department/file.xlsx") == "emergency_department"
    assert _extract_domain("/path/social_norms/edge.xlsx") == "social_norms"
    assert _extract_domain("/path/other/edge.xlsx") == "unknown"
    assert _extract_domain("depressive/edge.xlsx") == "depressive"


def test_extract_reason_from_json():
    msg = json.dumps({"judge_results": [{"reason": "  Plausible causal link  "}]})
    assert _extract_reason(msg) == "Plausible causal link"

    msg = json.dumps({"judge_results": []})
    assert _extract_reason(msg) == ""

    assert _extract_reason(pd.NA) == ""
    assert _extract_reason(None) == ""


def test_extract_reason_fallback_regex():
    msg = '{"reason": "Fallback reason", "other": "x"}'
    assert _extract_reason(msg) == "Fallback reason"


def test_build_rows_produces_expected_columns():
    df = pd.DataFrame([
        {
            "Source": "A", "Target": "B", "Relationship Type": "causes",
            "Motivation": "A causes B", "Judge Verdict": "CORRECT",
            "Judge Message": json.dumps({"judge_results": [{"reason": "Ok"}]}),
            "domain": "depressive",
        },
    ])
    result = _build_rows(df)
    assert "prompt" in result.columns
    assert "completion" in result.columns
    assert "source" in result.columns
    assert "target" in result.columns
    assert "domain" in result.columns
    assert "judge_verdict" in result.columns
    assert "REASON:" in result["completion"].iloc[0]
    assert "VERDICT:" in result["completion"].iloc[0]


def test_build_rows_drops_error_verdicts():
    df = pd.DataFrame([
        {
            "Source": "A", "Target": "B", "Relationship Type": "causes",
            "Motivation": "A causes B", "Judge Verdict": "ERROR",
            "Judge Message": json.dumps({"judge_results": [{"reason": "Err"}]}),
            "domain": "depressive",
        },
    ])
    result = _build_rows(df)
    assert len(result) == 0


def test_split_by_edge_identity_no_leakage():
    df = pd.DataFrame([
        {"source": "A", "target": "B", "domain": "d1"},
        {"source": "A", "target": "B", "domain": "d1"},
        {"source": "X", "target": "Y", "domain": "d1"},
    ])
    for col in ["prompt", "completion", "judge_verdict"]:
        df[col] = "x"
    train_df, val_df = _split_by_edge_identity(df, val_fraction=0.5, seed=42)
    train_edges = set(zip(train_df["source"], train_df["target"], train_df["domain"]))
    val_edges = set(zip(val_df["source"], val_df["target"], val_df["domain"]))
    assert train_edges.isdisjoint(val_edges)
    assert len(train_df) + len(val_df) == len(df)


def test_split_by_edge_identity_stratifies_domain_when_possible():
    df = pd.DataFrame([
        {"source": "A", "target": "B", "domain": "d1"},
        {"source": "C", "target": "D", "domain": "d1"},
        {"source": "E", "target": "F", "domain": "d2"},
        {"source": "G", "target": "H", "domain": "d2"},
    ])
    for col in ["prompt", "completion", "judge_verdict"]:
        df[col] = "x"

    train_df, val_df = _split_by_edge_identity(
        df, val_fraction=0.5, seed=42, stratify_col="domain"
    )
    assert set(val_df["domain"]) == {"d1", "d2"}
