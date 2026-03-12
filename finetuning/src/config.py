"""Hydra config schema and utilities."""

from omegaconf import DictConfig, OmegaConf


def resolve_repo_paths(cfg: DictConfig, repo_root: str) -> DictConfig:
    """Resolve relative paths in config to absolute paths under repo_root."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in ("train_file", "val_file", "output_dir"):
        if key in cfg and cfg[key]:
            val = str(cfg[key])
            if not val.startswith("/"):
                cfg[key] = f"{repo_root}/{val}"
    return cfg
