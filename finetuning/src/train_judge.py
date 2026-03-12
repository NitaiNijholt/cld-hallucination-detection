#!/usr/bin/env python3
"""
QLoRA supervised finetuning of Mistral-7B-Instruct-v0.2 on the CLD
correctness judge task (GT Synth data only).

Run locally:
    python finetuning/src/train_judge.py --smoke_test

Run on Snellius (via SLURM):
    sbatch finetuning/jobs/train_judge.job
"""

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class FinetuneConfig:
    base_model:   str   = "mistralai/Mistral-7B-Instruct-v0.2"
    train_file:   str   = "finetuning/src/judge_train.xlsx"
    val_file:     str   = "finetuning/src/judge_val_synth.xlsx"
    output_dir:   str   = "finetuning/runs/mistral7b_judge_gtsynth"
    max_length:   int   = 2048
    lora_rank:    int   = 32
    lora_alpha:   int   = 64
    lora_dropout: float = 0.05
    batch_size:   int   = 1
    grad_accum:   int   = 32
    epochs:       int   = 3
    lr:           float = 2e-4
    warmup_ratio: float = 0.05
    smoke_test:   bool  = False   # train on 64 examples for quick sanity check


def parse_args() -> FinetuneConfig:
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(FinetuneConfig):
        kwargs = {"default": f.default, "type": type(f.default)}
        if f.type is bool:
            kwargs = {"action": "store_true", "default": f.default}
        p.add_argument(f"--{f.name}", **kwargs)
    ns = p.parse_args()
    return FinetuneConfig(**vars(ns))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(path: str, smoke_test: bool) -> Dataset:
    df = pd.read_excel(path).dropna(subset=["prompt", "completion"])
    if smoke_test:
        df = df.head(64)
    return Dataset.from_pandas(df[["prompt", "completion"]].reset_index(drop=True))


def make_chat_text(prompt: str, completion: str, tokenizer) -> str:
    """
    Use the tokenizer's built-in chat template so the special tokens
    (BOS, [INST], [/INST], EOS) are always consistent with the model.
    The prompt column already contains the user turn text; completion is
    the assistant turn.
    """
    # Strip the manually-added [INST] wrapper that prepare_judge_data.py added
    # so we can re-apply via apply_chat_template for correctness.
    import re
    clean = re.sub(r"<s>\[INST\]|\[/INST\]|</s>", "", prompt).strip()

    messages = [
        {"role": "user",      "content": clean},
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
        full_texts    = []
        prompt_texts  = []

        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            full_texts.append(make_chat_text(prompt, completion, tokenizer))
            import re
            clean = re.sub(r"<s>\[INST\]|\[/INST\]|</s>", "", prompt).strip()
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": clean}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(prompt_only)

        full_enc   = tokenizer(full_texts,   truncation=True, max_length=max_length,
                               padding="max_length", add_special_tokens=False)
        prompt_enc = tokenizer(prompt_texts, truncation=True, max_length=max_length,
                               add_special_tokens=False)

        masked_labels = []
        for i, p_ids in enumerate(prompt_enc["input_ids"]):
            p_len  = len(p_ids)
            ids    = full_enc["input_ids"][i]
            labels = [-100] * p_len + ids[p_len:]
            labels = (labels + [-100] * max_length)[:max_length]
            masked_labels.append(labels)

        full_enc["labels"] = masked_labels
        full_enc["attention_mask"] = [
            [0 if t == pad_id else 1 for t in seq]
            for seq in full_enc["input_ids"]
        ]
        return full_enc

    return ds.map(
        process,
        batched=True,
        remove_columns=["prompt", "completion"],
        num_proc=4,
        desc="Tokenising",
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def load_quantised_model(cfg: FinetuneConfig):
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

def build_training_args(cfg: FinetuneConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
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
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        report_to="none",
        optim="adamw_torch",
    )


def save_artifacts(model, tokenizer, cfg: FinetuneConfig) -> None:
    adapter_dir = os.path.join(cfg.output_dir, "lora_adapter")
    merged_dir  = os.path.join(cfg.output_dir, "merged_fp16")

    print(f"Saving LoRA adapter → {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"Merging weights   → {merged_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


def save_loss_curve(trainer, output_dir: str) -> None:
    history    = trainer.state.log_history
    steps      = [e["step"] for e in history if "step" in e]
    train_loss = [e["loss"] for e in history if "loss" in e]
    eval_loss  = [e["eval_loss"] for e in history if "eval_loss" in e]

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

def main() -> None:
    cfg = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    print(f"\nLoading data …")
    train_ds = load_split(cfg.train_file, cfg.smoke_test)
    val_ds   = load_split(cfg.val_file,   cfg.smoke_test)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tok = build_tokenised_dataset(train_ds, tokenizer, cfg.max_length, tokenizer.pad_token_id)
    val_tok   = build_tokenised_dataset(val_ds,   tokenizer, cfg.max_length, tokenizer.pad_token_id)

    print(f"\nLoading {cfg.base_model} (4-bit QLoRA) …")
    model = load_quantised_model(cfg)

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
    )

    print("\nTraining …")
    trainer.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_loss_curve(trainer, cfg.output_dir)
    save_artifacts(model, tokenizer, cfg)
    print("\nFinished.")


if __name__ == "__main__":
    main()
