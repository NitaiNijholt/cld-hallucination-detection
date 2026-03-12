"""Unit tests for training config module."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.src.training.train_config import get_config_dir, get_repo_root, resolve_paths
from omegaconf import OmegaConf


def test_get_config_dir_exists():
    config_dir = get_config_dir()
    assert config_dir.exists()
    assert (config_dir / "default.yaml").exists()
    assert (config_dir / "smoke.yaml").exists()


def test_get_repo_root_respects_env(monkeypatch):
    monkeypatch.setenv("REPO_ROOT", "/custom/repo")
    assert str(get_repo_root()) == "/custom/repo"
    monkeypatch.delenv("REPO_ROOT", raising=False)
    monkeypatch.setenv("REPO_DIR", "/custom/dir")
    assert str(get_repo_root()) == "/custom/dir"


def test_resolve_paths_absolute_unchanged():
    cfg = OmegaConf.create({"train_file": "/abs/path/train.xlsx"})
    resolved = resolve_paths(cfg, Path("/repo"))
    assert resolved.train_file == "/abs/path/train.xlsx"


def test_resolve_paths_empty_key_unchanged():
    cfg = OmegaConf.create({"train_file": "rel/path", "output_dir": ""})
    resolved = resolve_paths(cfg, Path("/repo"))
    assert "rel" in str(resolved.train_file)
    assert resolved.output_dir == ""
