#!/usr/bin/env python3
"""Multi-GPU QLoRA finetuning for large models (e.g. Qwen3-235B-A22B on 4× H100).

Thin wrapper around train_judge.py that overrides model loading to shard
the 4-bit quantised base model across all visible GPUs via device_map="auto".
Everything else (data prep, tokenisation, Trainer loop, artifact saving)
is inherited unchanged.

Run on Snellius (via SLURM):
    sbatch finetuning/jobs/train_qwen3_235b_shared_synth_reason_verdict.job
"""

import logging
import sys

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from . import train_judge
from .train_config import get_config_dir

logger = logging.getLogger(__name__)


def load_quantised_model(cfg: DictConfig):
    """Load 4-bit quantised model sharded across all visible GPUs."""
    n_gpus = torch.cuda.device_count()
    logger.info("Multi-GPU mode: %d GPU(s) detected", n_gpus)
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info("  GPU %d: %s (%.1f GB)", i, props.name, props.total_memory / 1e9)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    per_gpu_gib = 70  # leave ~10 GiB headroom per H100
    max_memory = {i: f"{per_gpu_gib}GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "64GiB"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# Patch the module-level function so train_judge.main() uses our multi-GPU loader
train_judge.load_quantised_model = load_quantised_model


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

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
    with initialize_config_dir(config_dir=str(get_config_dir()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    train_judge.main(cfg)
