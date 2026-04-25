#!/usr/bin/env python3
"""Multi-GPU evaluation for large finetuned models (e.g. Qwen3-235B-A22B on 4× H100).

Thin wrapper around evaluate_judge.py that overrides model loading to shard
the 4-bit quantised model across all visible GPUs via device_map="auto"
with explicit per-GPU memory limits.  All evaluation logic (inference,
metrics, plotting, JSON export) is inherited unchanged.

Usage:
    python -m finetuning.src.evaluation.evaluate_judge_multigpu \
        --finetuned_model path/to/lora_adapter \
        --base_model Qwen/Qwen3-235B-A22B
"""

import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import evaluate_judge

logger = logging.getLogger(__name__)


def _load_model_tokenizer(model_path: str, base_model: str, is_peft: bool):
    """Load model sharded across all visible GPUs for inference."""
    n_gpus = torch.cuda.device_count()
    logger.info("Multi-GPU eval: %d GPU(s) detected", n_gpus)
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info("  GPU %d: %s (%.1f GB)", i, props.name, props.total_memory / 1e9)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    per_gpu_gib = 70
    max_memory = {i: f"{per_gpu_gib}GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "64GiB"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        quantization_config=bnb_cfg,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if is_peft:
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        base = PeftModel.from_pretrained(base, model_path)
    else:
        base = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    base.eval()
    return base, tokenizer


# Patch the module-level loader so evaluate_judge.main() uses our multi-GPU version
evaluate_judge._load_model_tokenizer = _load_model_tokenizer


if __name__ == "__main__":
    evaluate_judge.main()
