"""MLflow utilities for model versioning and registry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_tracking_uri(
    config_uri: str | None,
    repo_root: Path | None = None,
) -> str:
    """Resolve MLflow tracking URI: env MLFLOW_TRACKING_URI > config > default local."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    if config_uri and config_uri != "mlruns":
        return config_uri
    # Default: mlruns relative to repo root or cwd
    base = repo_root or Path.cwd()
    return str(base / "mlruns")


def log_training_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    registry_name: str,
    merged_dir: str,
    base_model: str,
    cfg_dict: dict[str, Any],
    train_loss: float | None = None,
    eval_loss: float | None = None,
    best_epoch: float | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
) -> str | None:
    """
    Log training run and model to MLflow. Returns run_id or None if logging fails.
    """
    try:
        import mlflow
        from mlflow.models import infer_signature
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.warning("MLflow/transformers not available: %s", e)
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        params = {
            "base_model": base_model,
            "lora_rank": cfg_dict.get("lora_rank"),
            "lora_alpha": cfg_dict.get("lora_alpha"),
            "epochs": cfg_dict.get("epochs"),
            "lr": cfg_dict.get("lr"),
            "max_length": cfg_dict.get("max_length"),
        }
        params = {k: v for k, v in params.items() if v is not None}
        mlflow.log_params(params)

        # Metrics
        if train_loss is not None:
            mlflow.log_metric("train_loss_final", train_loss)
        if eval_loss is not None:
            mlflow.log_metric("eval_loss_best", eval_loss)
        if best_epoch is not None:
            mlflow.log_metric("best_epoch", best_epoch)

        # Model
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype="auto", low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        signature = infer_signature(
            model_input="[INST] Judge this claim. [/INST]",
            model_output="REASON: ... VERDICT: CORRECT",
        )
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            signature=signature,
            registered_model_name=registry_name,
        )
        logger.info("MLflow model logged: %s", model_info.model_uri)
        return run.info.run_id
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        return None
