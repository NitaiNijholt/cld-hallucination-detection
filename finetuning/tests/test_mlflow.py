"""Tests for MLflow integration."""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_resolve_tracking_uri_env():
    """MLFLOW_TRACKING_URI env takes precedence."""
    from finetuning.src import mlflow_utils

    orig = os.environ.pop("MLFLOW_TRACKING_URI", None)
    try:
        os.environ["MLFLOW_TRACKING_URI"] = "file:///custom/mlruns"
        uri = mlflow_utils.resolve_tracking_uri("mlruns", repo_root=REPO_ROOT)
        assert uri == "file:///custom/mlruns"
    finally:
        if orig is not None:
            os.environ["MLFLOW_TRACKING_URI"] = orig
        elif "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]


def test_resolve_tracking_uri_default():
    """Default uses repo_root/mlruns when config is 'mlruns'."""
    from finetuning.src import mlflow_utils

    orig = os.environ.pop("MLFLOW_TRACKING_URI", None)
    try:
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
        uri = mlflow_utils.resolve_tracking_uri("mlruns", repo_root=REPO_ROOT)
        assert uri == str(REPO_ROOT / "mlruns")
    finally:
        if orig is not None:
            os.environ["MLFLOW_TRACKING_URI"] = orig


def test_resolve_tracking_uri_config_remote():
    """Config URI used when not 'mlruns' and env not set."""
    from finetuning.src import mlflow_utils

    orig = os.environ.pop("MLFLOW_TRACKING_URI", None)
    try:
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
        uri = mlflow_utils.resolve_tracking_uri(
            "https://mlflow.example.com", repo_root=REPO_ROOT
        )
        assert uri == "https://mlflow.example.com"
    finally:
        if orig is not None:
            os.environ["MLFLOW_TRACKING_URI"] = orig


def test_log_training_run_missing_dir_returns_none():
    """log_training_run returns None when merged_dir does not exist (no crash)."""
    pytest.importorskip("mlflow")
    from finetuning.src import mlflow_utils

    run_id = mlflow_utils.log_training_run(
        tracking_uri=str(REPO_ROOT / "mlruns"),
        experiment_name="test",
        registry_name="test_reg",
        merged_dir="/nonexistent/path",
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        cfg_dict={"lora_rank": 32, "epochs": 1},
    )
    # Should fail gracefully (model load fails) and return None
    assert run_id is None
