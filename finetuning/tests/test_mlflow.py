"""Tests for MLflow integration."""

import os
import sys
import types
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


def test_log_training_run_sets_text_generation_task(monkeypatch, tmp_path):
    """Local model logging should pass task='text-generation' explicitly."""
    from finetuning.src import mlflow_utils

    calls = {}

    class DummyRun:
        class Info:
            run_id = "run-123"
        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMlflow:
        def set_tracking_uri(self, uri):
            calls["tracking_uri"] = uri

        def set_experiment(self, name):
            calls["experiment_name"] = name

        def start_run(self, run_name=None):
            calls["run_name"] = run_name
            return DummyRun()

        def log_params(self, params):
            calls["params"] = params

        def log_metric(self, name, value):
            calls.setdefault("metrics", {})[name] = value

    class DummyTransformersFlavor:
        @staticmethod
        def log_model(**kwargs):
            calls["log_model_kwargs"] = kwargs
            return types.SimpleNamespace(model_uri="models:/dummy")

    dummy_mlflow = DummyMlflow()
    dummy_mlflow.transformers = DummyTransformersFlavor()

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class DummyAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.models",
        types.SimpleNamespace(infer_signature=lambda **kwargs: "sig"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=DummyAutoModel,
            AutoTokenizer=DummyAutoTokenizer,
        ),
    )

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()

    run_id = mlflow_utils.log_training_run(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="test-exp",
        registry_name="test-registry",
        merged_dir=str(merged_dir),
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        cfg_dict={"lora_rank": 32, "epochs": 1, "lr": 2e-4, "max_length": 2048},
        train_loss=1.23,
        eval_loss=0.45,
    )

    assert run_id == "run-123"
    assert calls["log_model_kwargs"]["task"] == "text-generation"
