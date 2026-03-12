"""Tests for training config and utilities."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_config_composes_correctly():
    """Hydra config loads and smoke override applies."""
    from hydra.core.global_hydra import GlobalHydra
    from hydra import compose, initialize_config_dir
    from finetuning.src.training.train_config import get_config_dir

    config_dir = get_config_dir()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_default = compose(config_name="default", overrides=[])
        cfg_smoke = compose(config_name="smoke", overrides=[])

    assert cfg_default.smoke_test is False
    assert cfg_smoke.smoke_test is True
    assert cfg_smoke.epochs == 1
    assert cfg_default.epochs == 3


def test_path_resolution():
    """Paths resolve correctly with REPO_ROOT."""
    from finetuning.src.training.train_config import resolve_paths
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "train_file": "finetuning/src/judge_train.xlsx",
        "val_file": "finetuning/src/judge_val_synth.xlsx",
        "output_dir": "finetuning/runs/test",
    })
    repo = Path(__file__).resolve().parent.parent.parent
    resolved = resolve_paths(cfg, repo)
    assert str(resolved.train_file).endswith("judge_train.xlsx")
    assert str(resolved.output_dir).endswith("runs/test")


def test_build_training_args():
    """TrainingArguments built from config (requires transformers)."""
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from finetuning.src.training.train_judge import build_training_args
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "output_dir": "/tmp/out",
        "epochs": 2,
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 1e-4,
        "warmup_ratio": 0.1,
        "seed": 42,
        "save_strategy": "no",
        "load_best_model_at_end": False,
    })
    args = build_training_args(cfg)
    assert args.num_train_epochs == 2
    assert args.seed == 42
