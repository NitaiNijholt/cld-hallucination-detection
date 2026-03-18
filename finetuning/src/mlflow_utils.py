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
    run_metadata: dict[str, Any] | None = None,
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

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Params
            params = {
                "base_model": base_model,
                "dataset_label": cfg_dict.get("dataset_label"),
                "objective_mode": cfg_dict.get("objective_mode"),
                "train_file": cfg_dict.get("train_file"),
                "val_file": cfg_dict.get("val_file"),
                "lora_rank": cfg_dict.get("lora_rank"),
                "lora_alpha": cfg_dict.get("lora_alpha"),
                "batch_size": cfg_dict.get("batch_size"),
                "grad_accum": cfg_dict.get("grad_accum"),
                "epochs": cfg_dict.get("epochs"),
                "lr": cfg_dict.get("lr"),
                "max_length": cfg_dict.get("max_length"),
                "seed": cfg_dict.get("seed"),
            }
            if run_metadata is not None:
                params["git_sha"] = run_metadata.get("git_sha")
                params["train_dataset_sha256"] = (
                    run_metadata.get("train_dataset", {}).get("sha256")
                )
                params["val_dataset_sha256"] = (
                    run_metadata.get("val_dataset", {}).get("sha256")
                )
            params = {k: v for k, v in params.items() if v is not None}
            mlflow.log_params(params)
            mlflow.log_dict(cfg_dict, "resolved_config.json")
            if run_metadata is not None:
                mlflow.log_dict(run_metadata, "run_metadata.json")
                for split_name in ("train_dataset", "val_dataset"):
                    sidecar_path = run_metadata.get(split_name, {}).get("sidecar_path")
                    if sidecar_path and os.path.exists(sidecar_path):
                        mlflow.log_artifact(sidecar_path, artifact_path="dataset_metadata")

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
            objective_mode = cfg_dict.get("objective_mode", "reason_verdict")
            signature = infer_signature(
                model_input="[INST] Judge this claim. [/INST]",
                model_output=(
                    "REASON: ... VERDICT: CORRECT"
                    if objective_mode == "reason_verdict"
                    else "VERDICT: CORRECT"
                ),
            )
            model_info = mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="model",
                task="text-generation",
                signature=signature,
                registered_model_name=registry_name,
            )
            logger.info("MLflow model logged: %s", model_info.model_uri)
            return run.info.run_id
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        return None
