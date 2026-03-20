"""Unit tests for data preparation helpers."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from zipfile import BadZipFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.src.data.prepare_judge_data import (
    _balance_rows_exact,
    _build_rows,
    _exclude_overlapping_edges,
    _extract_domain,
    _extract_reason,
    _infer_prompt_variant,
    _load_xlsx_files,
    _sanitize_reason,
    _split_preserving_group_coverage,
    _should_include_xlsx_file,
    _split_by_edge_identity,
)
from finetuning.src.metadata_utils import atomic_to_excel, read_excel_with_retry


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


def test_sanitize_reason_strips_embedded_score_and_verdict():
    reason = "REASON: Plausible mechanism\\nSCORE: 1.0\\nVERDICT: CORRECT"
    assert _sanitize_reason(reason) == "Plausible mechanism"


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


def test_build_rows_verdict_only_omits_reason():
    df = pd.DataFrame([
        {
            "Source": "A", "Target": "B", "Relationship Type": "causes",
            "Motivation": "A causes B", "Judge Verdict": "CORRECT",
            "domain": "depressive",
        },
    ])
    result = _build_rows(df, objective_mode="verdict_only")
    assert result["completion"].iloc[0] == "VERDICT: CORRECT"
    assert "Respond with VERDICT only." in result["prompt"].iloc[0]


def test_infer_prompt_variant_from_filepath():
    assert _infer_prompt_variant("/tmp/judged_demo_mechanistic_1.xlsx") == "mechanistic"
    assert _infer_prompt_variant("/tmp/judged_demo_cot_1.xlsx") == "cot"
    assert _infer_prompt_variant("/tmp/judged_demo_baseline_1.xlsx") == "baseline"
    assert _infer_prompt_variant("/tmp/other.xlsx") is None


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


def test_split_by_edge_identity_stratifies_domain_and_verdict_when_possible():
    df = pd.DataFrame([
        {"source": "A", "target": "B", "domain": "d1", "judge_verdict": "CORRECT"},
        {"source": "C", "target": "D", "domain": "d1", "judge_verdict": "CORRECT"},
        {"source": "E", "target": "F", "domain": "d2", "judge_verdict": "INCORRECT"},
        {"source": "G", "target": "H", "domain": "d2", "judge_verdict": "INCORRECT"},
    ])
    for col in ["prompt", "completion"]:
        df[col] = "x"

    train_df, val_df = _split_by_edge_identity(
        df,
        val_fraction=0.5,
        seed=42,
        stratify_cols=["domain", "judge_verdict"],
    )
    held_out_pairs = set(zip(val_df["domain"], val_df["judge_verdict"]))
    assert held_out_pairs == {("d1", "CORRECT"), ("d2", "INCORRECT")}


def test_split_by_edge_identity_can_fail_closed_when_stratification_required():
    df = pd.DataFrame([
        {"source": "A", "target": "B", "domain": "d1", "judge_verdict": "CORRECT"},
        {"source": "C", "target": "D", "domain": "d1", "judge_verdict": "INCORRECT"},
        {"source": "E", "target": "F", "domain": "d2", "judge_verdict": "CORRECT"},
    ])
    for col in ["prompt", "completion"]:
        df[col] = "x"

    with pytest.raises(ValueError, match="Unable to preserve requested stratification"):
        _split_by_edge_identity(
            df,
            val_fraction=0.5,
            seed=42,
            stratify_cols=["domain", "judge_verdict"],
            require_stratification=True,
        )


def test_balance_rows_exact_equalizes_domain_and_verdict():
    df = pd.DataFrame([
        {"prompt": "p1", "completion": "c1", "source": "a", "target": "b", "domain": "d1", "judge_verdict": "CORRECT"},
        {"prompt": "p2", "completion": "c2", "source": "c", "target": "d", "domain": "d1", "judge_verdict": "CORRECT"},
        {"prompt": "p3", "completion": "c3", "source": "e", "target": "f", "domain": "d1", "judge_verdict": "INCORRECT"},
        {"prompt": "p4", "completion": "c4", "source": "g", "target": "h", "domain": "d2", "judge_verdict": "CORRECT"},
        {"prompt": "p5", "completion": "c5", "source": "i", "target": "j", "domain": "d2", "judge_verdict": "INCORRECT"},
        {"prompt": "p6", "completion": "c6", "source": "k", "target": "l", "domain": "d2", "judge_verdict": "INCORRECT"},
    ])

    balanced = _balance_rows_exact(df, ["domain", "judge_verdict"], seed=42)
    counts = balanced.groupby(["domain", "judge_verdict"]).size()
    assert counts.to_dict() == {
        ("d1", "CORRECT"): 1,
        ("d1", "INCORRECT"): 1,
        ("d2", "CORRECT"): 1,
        ("d2", "INCORRECT"): 1,
    }


def test_load_xlsx_files_only_keeps_judged_non_backup_files(tmp_path):
    valid_path = tmp_path / "depressive" / "run_1" / "judged_demo_mechanistic.xlsx"
    backup_path = tmp_path / "depressive" / "run_1" / "judged_demo_mechanistic.backup.xlsx"
    analysis_path = tmp_path / "depressive" / "run_1" / "prompt_variants_analysis.xlsx"
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    sample = pd.DataFrame(
        [
            {
                "Source": "A",
                "Target": "B",
                "Relationship Type": "causes",
                "Motivation": "A causes B",
                "Judge Verdict": "CORRECT",
                "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
            }
        ]
    )
    sample.to_excel(valid_path, index=False)
    sample.to_excel(backup_path, index=False)
    sample.to_excel(analysis_path, index=False)

    assert _should_include_xlsx_file(str(valid_path), 0) is True
    assert _should_include_xlsx_file(str(backup_path), 0) is False
    assert _should_include_xlsx_file(str(analysis_path), 0) is False

    loaded = _load_xlsx_files(str(tmp_path / "**" / "*.xlsx"), 0)
    assert len(loaded) == 1
    assert loaded["domain"].iloc[0] == "depressive"
    assert loaded["prompt_variant"].iloc[0] == "mechanistic"


def test_split_preserving_group_coverage_can_adjust_seed():
    lit_df = pd.DataFrame([
        {"source": "x1", "target": "y1", "domain": "social_norms"},
        {"source": "x2", "target": "y2", "domain": "social_norms"},
    ])
    for col, value in [("prompt", "p"), ("completion", "c"), ("judge_verdict", "CORRECT")]:
        lit_df[col] = value

    synth_df = pd.DataFrame([
        {"source": "x1", "target": "y1", "domain": "social_norms", "judge_verdict": "CORRECT", "prompt": "p", "completion": "c"},
        {"source": "x3", "target": "y3", "domain": "social_norms", "judge_verdict": "CORRECT", "prompt": "p", "completion": "c"},
        {"source": "x4", "target": "y4", "domain": "social_norms", "judge_verdict": "INCORRECT", "prompt": "p", "completion": "c"},
        {"source": "x5", "target": "y5", "domain": "social_norms", "judge_verdict": "INCORRECT", "prompt": "p", "completion": "c"},
        {"source": "x6", "target": "y6", "domain": "social_norms", "judge_verdict": "PARTIALLY_CORRECT", "prompt": "p", "completion": "c"},
        {"source": "x7", "target": "y7", "domain": "social_norms", "judge_verdict": "PARTIALLY_CORRECT", "prompt": "p", "completion": "c"},
        {"source": "a1", "target": "b1", "domain": "depressive", "judge_verdict": "CORRECT", "prompt": "p", "completion": "c"},
        {"source": "a2", "target": "b2", "domain": "depressive", "judge_verdict": "CORRECT", "prompt": "p", "completion": "c"},
        {"source": "a3", "target": "b3", "domain": "depressive", "judge_verdict": "INCORRECT", "prompt": "p", "completion": "c"},
        {"source": "a4", "target": "b4", "domain": "depressive", "judge_verdict": "INCORRECT", "prompt": "p", "completion": "c"},
        {"source": "a5", "target": "b5", "domain": "depressive", "judge_verdict": "PARTIALLY_CORRECT", "prompt": "p", "completion": "c"},
        {"source": "a6", "target": "b6", "domain": "depressive", "judge_verdict": "PARTIALLY_CORRECT", "prompt": "p", "completion": "c"},
    ])
    filtered_synth, _ = _exclude_overlapping_edges(synth_df, lit_df)
    train_df, val_df = _split_preserving_group_coverage(
        filtered_synth,
        val_fraction=0.5,
        seed=42,
        stratify_cols=["domain"],
        group_cols=["domain", "judge_verdict"],
        require_stratification=True,
        max_seed_tries=200,
    )
    assert set(map(tuple, val_df[["domain", "judge_verdict"]].drop_duplicates().itertuples(index=False, name=None))) == {
        ("social_norms", "CORRECT"),
        ("social_norms", "INCORRECT"),
        ("social_norms", "PARTIALLY_CORRECT"),
        ("depressive", "CORRECT"),
        ("depressive", "INCORRECT"),
        ("depressive", "PARTIALLY_CORRECT"),
    }


def test_read_excel_with_retry_recovers_from_transient_badzip(tmp_path, monkeypatch):
    path = tmp_path / "sample.xlsx"
    atomic_to_excel(pd.DataFrame([{"prompt": "p", "completion": "c"}]), path)

    original = pd.read_excel
    state = {"calls": 0}

    def flaky_read_excel(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise BadZipFile("Bad CRC-32 for file 'docProps/core.xml'")
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", flaky_read_excel)

    df = read_excel_with_retry(path, attempts=3, delay_seconds=0.01)
    assert len(df) == 1
    assert state["calls"] == 2
