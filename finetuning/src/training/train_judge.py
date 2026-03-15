#!/usr/bin/env python3
"""
QLoRA supervised finetuning of Mistral-7B-Instruct-v0.2 on the CLD
correctness judge task (GT Synth data only).

Run locally:
    python -m finetuning.src.training.train_judge
    python -m finetuning.src.training.train_judge --config-name smoke

Run on Snellius (via SLURM):
    sbatch finetuning/jobs/train_judge.job
"""

import logging
import os
import random

import re
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    load_dotenv(os.path.join(_root, ".env"))
    load_dotenv(os.path.join(_root, "finetuning", ".env"))
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def _setup_logging(output_dir: str | None) -> None:
    """Configure logging to stdout and optionally to a file. Unbuffer stdout."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except (OSError, AttributeError):
        pass
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if output_dir:
        log_path = os.path.join(output_dir, "train.log")
        os.makedirs(output_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    for name in ("transformers", "transformers.trainer"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        for h in handlers:
            if h not in log.handlers:
                log.addHandler(h)


logger = logging.getLogger(__name__)


class StepProgressCallback(TrainerCallback):
    """Print step/epoch progress to stdout with immediate flush."""

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file

    def _emit(self, msg: str) -> None:
        print(msg, flush=True)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except OSError:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None:
            step = state.global_step
            epoch = state.epoch or 0
            self._emit(f"  step {step}  loss={loss:.4f}  epoch={epoch:.2f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            step = state.global_step
            self._emit(f"  step {step}  eval_loss={eval_loss:.4f}")


class WandbConfigCallback(TrainerCallback):
    """Log Hydra config, base model, and output paths to W&B on train begin."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is not None:
                cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
                wandb.config.update({"hydra_config": cfg_dict})
                wandb.config.update({
                    "base_model": str(self.cfg.get("base_model", "")),
                    "output_dir": str(self.cfg.get("output_dir", "")),
                })
        except Exception:
            pass


from .train_config import (
    get_config_dir as _get_config_dir,
    get_repo_root as _get_repo_root,
    resolve_paths as _resolve_paths,
)

_CONFIG_DIR = _get_config_dir()


def _set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_git_sha() -> str:
    """Return current git commit SHA for run metadata."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=_get_repo_root()
        ).strip()[:8]
    except Exception:
        return "unknown"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(path: str, smoke_test: bool) -> Dataset:
    df = pd.read_excel(path).dropna(subset=["prompt", "completion"])
    if smoke_test:
        df = df.head(64)
    return Dataset.from_pandas(df[["prompt", "completion"]].reset_index(drop=True))


def make_chat_text(prompt: str, completion: str, tokenizer) -> str:
    """Use tokenizer chat template for consistent special tokens."""
    clean = re.sub(r"<s>\[INST\]|\[/INST\]|</s>", "", prompt).strip()
    messages = [
        {"role": "user", "content": clean},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_tokenised_dataset(
    ds: Dataset, tokenizer, max_length: int, pad_id: int
) -> Dataset:
    """Tokenise and mask prompt tokens so loss is computed on completion only."""

    def process(batch):
        full_texts = []
        prompt_texts = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            full_texts.append(make_chat_text(prompt, completion, tokenizer))
            clean = re.sub(r"<s>\[INST\]|\[/INST\]|</s>", "", prompt).strip()
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": clean}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(prompt_only)

        full_enc = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=False,
        )
        prompt_enc = tokenizer(
            prompt_texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        masked_labels = []
        for i, p_ids in enumerate(prompt_enc["input_ids"]):
            p_len = len(p_ids)
            ids = full_enc["input_ids"][i]
            labels = [-100] * p_len + ids[p_len:]
            labels = (labels + [-100] * max_length)[:max_length]
            masked_labels.append(labels)

        full_enc["labels"] = masked_labels
        full_enc["attention_mask"] = [
            [0 if t == pad_id else 1 for t in seq]
            for seq in full_enc["input_ids"]
        ]
        return full_enc

    num_proc = min(4, max(1, len(ds) // 4))  # avoid num_proc > shards for tiny datasets
    return ds.map(
        process,
        batched=True,
        remove_columns=["prompt", "completion"],
        num_proc=num_proc,
        desc="Tokenising",
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def load_quantised_model(cfg: DictConfig):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def build_training_args(cfg: DictConfig) -> TrainingArguments:
    wandb_project = getattr(cfg, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
    use_wandb = bool(wandb_project) and not os.environ.get("WANDB_DISABLED")
    report_to = "wandb" if use_wandb else "none"
    if use_wandb and not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = str(wandb_project)
    if use_wandb:
        group = getattr(cfg, "wandb_group", None)
        if group:
            os.environ["WANDB_GROUP"] = str(group)

    save_strategy = getattr(cfg, "save_strategy", "epoch")
    load_best = getattr(cfg, "load_best_model_at_end", True)
    metric_for_best = getattr(cfg, "metric_for_best_model", "eval_loss")
    save_total_limit = getattr(cfg, "save_total_limit", 2)
    run_name = getattr(cfg, "wandb_run_name", None) or None

    return TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=run_name,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit if save_strategy != "no" else None,
        load_best_model_at_end=load_best and save_strategy != "no",
        metric_for_best_model=metric_for_best,
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=getattr(cfg, "logging_steps", 10),
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        report_to=report_to,
        optim="adamw_torch",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )


def save_artifacts(model, tokenizer, cfg: DictConfig) -> None:
    adapter_dir = os.path.join(cfg.output_dir, "lora_adapter")
    merged_dir = os.path.join(cfg.output_dir, "merged_fp16")

    logger.info("Saving LoRA adapter → %s", adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    logger.info("Merging weights → %s", merged_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


def save_loss_curve(trainer, output_dir: str) -> None:
    history = trainer.state.log_history
    steps = [e["step"] for e in history if "step" in e]
    train_loss = [e["loss"] for e in history if "loss" in e]
    eval_loss = [e["eval_loss"] for e in history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_loss:
        ax.plot(steps[:len(train_loss)], train_loss, label="train")
    if eval_loss:
        ax.plot(steps[1:1 + len(eval_loss)], eval_loss, label="val (GT Synth)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("CLD Judge — GT Synth QLoRA training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(cfg: DictConfig) -> None:
    repo_root = _get_repo_root()
    cfg = _resolve_paths(cfg, repo_root)

    os.makedirs(cfg.output_dir, exist_ok=True)
    _setup_logging(cfg.output_dir)

    _set_seeds(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    git_sha = _get_git_sha()
    logger.info("Config: %s", OmegaConf.to_yaml(cfg))
    logger.info("Git SHA: %s", git_sha)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB)", props.name, props.total_memory / 1e9)

    logger.info("Loading data…")
    train_ds = load_split(cfg.train_file, cfg.smoke_test)
    val_ds = load_split(cfg.val_file, cfg.smoke_test)
    logger.info("train=%s val=%s", f"{len(train_ds):,}", f"{len(val_ds):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tok = build_tokenised_dataset(
        train_ds, tokenizer, cfg.max_length, tokenizer.pad_token_id
    )
    val_tok = build_tokenised_dataset(
        val_ds, tokenizer, cfg.max_length, tokenizer.pad_token_id
    )

    logger.info("Loading %s (4-bit QLoRA)…", cfg.base_model)
    model = load_quantised_model(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train.log")
    step_callback = StepProgressCallback(log_file=log_path)

    callbacks = [step_callback, WandbConfigCallback(cfg)]
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Training… (step logs every 10 steps)")
    trainer.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "run_metadata.txt"), "w") as f:
        f.write(f"git_sha={git_sha}\n")
        f.write(f"config={OmegaConf.to_yaml(cfg)}\n")
    save_loss_curve(trainer, cfg.output_dir)
    save_artifacts(model, tokenizer, cfg)

    # MLflow model versioning and registry
    if getattr(cfg, "mlflow_experiment_name", None):
        from .. import mlflow_utils as mlflow_mod
        tracking_uri = mlflow_mod.resolve_tracking_uri(
            getattr(cfg, "mlflow_tracking_uri", None),
            repo_root=repo_root,
        )
        history = trainer.state.log_history
        train_loss_final = next((h["loss"] for h in reversed(history) if "loss" in h), None)
        eval_loss_best = trainer.state.best_metric
        best_epoch = None
        for h in reversed(history):
            if "eval_loss" in h and h.get("eval_loss") == eval_loss_best:
                best_epoch = h.get("epoch")
                break
        merged_dir = os.path.join(cfg.output_dir, "merged_fp16")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            mlflow_mod.log_training_run(
                tracking_uri=tracking_uri,
                experiment_name=cfg.mlflow_experiment_name,
                registry_name=getattr(cfg, "mlflow_registry_name", "cld_judge"),
                merged_dir=merged_dir,
                base_model=str(cfg.base_model),
                cfg_dict=cfg_dict,
                train_loss=train_loss_final,
                eval_loss=float(eval_loss_best) if eval_loss_best is not None else None,
                best_epoch=float(best_epoch) if best_epoch is not None else None,
            )

    logger.info("Finished.")


if __name__ == "__main__":
    import sys
    from hydra import compose, initialize_config_dir

    config_name = "default"
    overrides = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config-name" and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
            continue
        if "=" in args[i]:
            overrides.append(args[i])
        i += 1

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    main(cfg)
