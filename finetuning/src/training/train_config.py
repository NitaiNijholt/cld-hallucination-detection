"""Training config utilities (lightweight, no torch/transformers)."""

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_config_dir() -> Path:
    """Return path to configs directory."""
    return _CONFIG_DIR


def get_repo_root() -> Path:
    """Resolve repo root (parent of finetuning/)."""
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    if env_root:
        return Path(env_root)
    return _REPO_ROOT


def resolve_paths(cfg: DictConfig, repo_root: Path) -> DictConfig:
    """Resolve relative paths to absolute under repo_root."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in ("train_file", "val_file", "output_dir"):
        if key in cfg and cfg[key]:
            val = str(cfg[key])
            if not Path(val).is_absolute():
                cfg[key] = str(repo_root / val)
    return cfg
