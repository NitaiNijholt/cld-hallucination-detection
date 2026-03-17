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


def test_prepare_gt_lit_train_val_produces_holdout_split():
    """GT Lit split creates train/val/test files with no edge leakage."""
    import argparse
    import pandas as pd

    from finetuning.src.data.prepare_gt_lit_train_val import main as split_main
    from finetuning.src.data.prepare_judge_data import main as prepare_main

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        prep_args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
        )
        prepare_main(args=prep_args)

        split_args = argparse.Namespace(
            input_path=str(Path(out_dir) / "judge_eval_gtlit.xlsx"),
            output_dir=out_dir,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
        split_main(args=split_args)

        train_path = Path(out_dir) / "judge_train_gtlit.xlsx"
        val_path = Path(out_dir) / "judge_val_gtlit.xlsx"
        test_path = Path(out_dir) / "judge_test_gtlit.xlsx"
        for path in [train_path, val_path, test_path]:
            assert path.exists(), f"{path.name} should exist"

        train_df = pd.read_excel(train_path)
        val_df = pd.read_excel(val_path)
        test_df = pd.read_excel(test_path)

        train_edges = set(zip(train_df["source"], train_df["target"], train_df["domain"]))
        val_edges = set(zip(val_df["source"], val_df["target"], val_df["domain"]))
        test_edges = set(zip(test_df["source"], test_df["target"], test_df["domain"]))

        assert train_edges.isdisjoint(val_edges)
        assert train_edges.isdisjoint(test_edges)
        assert val_edges.isdisjoint(test_edges)


def test_create_gt_lit_test_subsample_from_holdout():
    """Held-out GT Lit test subsample is created from the test split only."""
    import argparse
    import pandas as pd

    from finetuning.src.data.prepare_gt_lit_train_val import main as split_main
    from finetuning.src.data.prepare_judge_data import main as prepare_main
    from finetuning.src.evaluation.create_gt_lit_test_subsample import main as subsample_main

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        prep_args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
        )
        prepare_main(args=prep_args)

        split_args = argparse.Namespace(
            input_path=str(Path(out_dir) / "judge_eval_gtlit.xlsx"),
            output_dir=out_dir,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
        split_main(args=split_args)

        import sys
        orig_argv = sys.argv
        try:
            sys.argv = [
                "create_gt_lit_test_subsample",
                "--input_path", str(Path(out_dir) / "judge_test_gtlit.xlsx"),
                "--output_path", str(Path(out_dir) / "judge_test_gtlit_1k.xlsx"),
                "--target_n", "5",
                "--seed", "42",
            ]
            subsample_main()
        finally:
            sys.argv = orig_argv

        test_df = pd.read_excel(Path(out_dir) / "judge_test_gtlit.xlsx")
        sub_df = pd.read_excel(Path(out_dir) / "judge_test_gtlit_1k.xlsx")
        assert len(sub_df) <= min(5, len(test_df))
