"""Tests for data preparation."""

import tempfile
from pathlib import Path

import pytest

# Run from repo root so finetuning package is importable
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_prepare_judge_data_produces_expected_files():
    """Run prepare_judge_data with fixture; assert output files exist and have expected columns."""
    from finetuning.src.data.prepare_judge_data import main
    import argparse

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,  # 50% val for small fixture
            seed=42,
            min_file_bytes=0,  # include small fixture files
        )
        main(args=args)

        train_path = Path(out_dir) / "judge_train.xlsx"
        val_path = Path(out_dir) / "judge_val_synth.xlsx"
        lit_path = Path(out_dir) / "judge_eval_gtlit.xlsx"

        assert train_path.exists(), "judge_train.xlsx should exist"
        assert val_path.exists(), "judge_val_synth.xlsx should exist"
        assert lit_path.exists(), "judge_eval_gtlit.xlsx should exist"

        import pandas as pd
        for p, name in [(train_path, "train"), (val_path, "val"), (lit_path, "lit")]:
            df = pd.read_excel(p)
            required = ["prompt", "completion", "source", "target", "domain", "judge_verdict"]
            for col in required:
                assert col in df.columns, f"{name} should have column {col}"
