"""Integration tests: data prep -> training data format compatibility."""

import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_data_prep_output_compatible_with_training():
    """Data prep outputs can be loaded by training's load_split (format check)."""
    from finetuning.src.data.prepare_judge_data import main
    import argparse
    import pandas as pd

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
        )
        main(args=args)

        train_path = Path(out_dir) / "judge_train.xlsx"
        val_path = Path(out_dir) / "judge_val_synth.xlsx"

        for path in [train_path, val_path]:
            df = pd.read_excel(path).dropna(subset=["prompt", "completion"])
            assert len(df) > 0, f"{path.name} should have rows"
            assert "prompt" in df.columns and "completion" in df.columns
            for _, row in df.iterrows():
                assert isinstance(row["prompt"], str) and len(row["prompt"]) > 0
                assert isinstance(row["completion"], str) and len(row["completion"]) > 0
                assert "VERDICT:" in row["completion"] or "REASON:" in row["completion"]


def test_full_config_flow():
    """Config loads, paths resolve, and training args can be built (when deps available)."""
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")

    from hydra.core.global_hydra import GlobalHydra
    from hydra import compose, initialize_config_dir
    from finetuning.src.training.train_config import get_config_dir, get_repo_root, resolve_paths
    from finetuning.src.training.train_judge import build_training_args

    config_dir = get_config_dir()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="smoke", overrides=[])

    repo = get_repo_root()
    cfg = resolve_paths(cfg, repo)
    args = build_training_args(cfg)

    assert args.num_train_epochs == 1
    assert cfg.smoke_test is True
