#!/usr/bin/env python3
"""
Evaluate a finetuned CLD judge model on GT Synth (in-distribution) and
GT Lit (out-of-distribution) to measure the transfer gap.

Evaluation protocol:
  - Load finetuned LoRA adapter (or merged checkpoint) + base model
  - Optionally also evaluate base (zero-shot) for direct comparison
  - Run inference on judge_val_synth.xlsx and judge_eval_gtlit.xlsx
  - Parse VERDICT token from model output
  - Compute F1 macro, AUC; print comparison table vs GPT-4.1 baselines
  - Save evaluation_results.json and confusion_matrices.png

Usage:
    python -m finetuning.src.evaluation.evaluate_judge --finetuned_model finetuning/runs/mistral7b_judge_gtsynth/lora_adapter
    python -m finetuning.src.evaluation.evaluate_judge --finetuned_model path/to/adapter --eval_base
"""

import argparse
import json
import logging
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

THESIS_BASELINES = {
    "GT Synth": {"f1": 0.74, "auc": 0.86},
    "GT Lit": {"f1": 0.35, "auc": 0.60},
}

from .eval_utils import INCORRECT, VERDICT_LABELS, parse_verdict


def _get_default_paths():
    """Resolve default paths relative to repo root."""
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    if env_root:
        root = Path(env_root)
    else:
        root = Path(__file__).resolve().parent.parent.parent.parent
    return {
        "val_path": root / "finetuning" / "src" / "judge_val_synth.xlsx",
        "lit_path": root / "finetuning" / "src" / "judge_eval_gtlit.xlsx",
        "output_dir": root / "results",
    }


def parse_args() -> argparse.Namespace:
    defaults = _get_default_paths()
    p = argparse.ArgumentParser(description="Evaluate CLD judge model")
    p.add_argument(
        "--finetuned_model",
        required=True,
        help="Path to finetuned LoRA adapter or merged checkpoint",
    )
    p.add_argument(
        "--base_model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace base model id or path",
    )
    p.add_argument(
        "--eval_base",
        action="store_true",
        help="Evaluate base model zero-shot (for F1/AUC + loss baseline)",
    )
    p.add_argument(
        "--no_eval_base",
        dest="eval_base",
        action="store_false",
        help="Skip base model evaluation",
    )
    p.set_defaults(eval_base=True)
    p.add_argument(
        "--val_path",
        default=str(defaults["val_path"]),
        help="Path to GT Synth val xlsx",
    )
    p.add_argument(
        "--lit_path",
        default=str(defaults["lit_path"]),
        help="Path to GT Lit eval xlsx",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    p.add_argument(
        "--output_dir",
        default=str(defaults["output_dir"]),
    )
    return p.parse_args()


def _make_chat_text(prompt: str, completion: str, tokenizer) -> str:
    """Match train_judge: full prompt+completion for loss (completion tokens only)."""
    clean = re.sub(r"<s>\[INST\]|\[/INST\]|</s>", "", prompt).strip()
    messages = [
        {"role": "user", "content": clean},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


@torch.no_grad()
def compute_eval_loss(model, tokenizer, df: pd.DataFrame, max_length: int = 2048, batch_size: int = 4) -> float:
    """Compute mean cross-entropy loss on completion tokens (same protocol as training)."""
    df = df.dropna(subset=["prompt", "completion"])
    if len(df) == 0:
        return float("nan")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    total_loss = 0.0
    n_tokens = 0
    for i in tqdm(range(0, len(df), batch_size), desc="  Eval loss"):
        batch = df.iloc[i : i + batch_size]
        full_texts = []
        prompt_lengths = []
        for _, r in batch.iterrows():
            full_texts.append(_make_chat_text(r["prompt"], r["completion"], tokenizer))
            clean = re.sub(r"<s>\[INST\]|\[/INST\]|</s>", "", r["prompt"]).strip()
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": clean}],
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            prompt_lengths.append(len(p_ids))
        full_enc = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        labels = full_enc["input_ids"].clone()
        for j, p_len in enumerate(prompt_lengths):
            labels[j, :p_len] = -100
        labels[labels == pad_id] = -100
        out = model(**full_enc, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += out.loss.item() * n
        n_tokens += n
    return total_loss / n_tokens if n_tokens > 0 else float("nan")


def _load_model_tokenizer(model_path: str, base_model: str, is_peft: bool):
    """Load model (LoRA adapter or merged checkpoint) in 4-bit for inference."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        base = PeftModel.from_pretrained(base, model_path)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    base.eval()
    return base, tokenizer


@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="  Inference"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j, g in enumerate(gen):
            prompt_len = enc["input_ids"][j].shape[0]
            text = tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
            outputs.append(text)
    return outputs


def evaluate_on_df(
    model,
    tokenizer,
    df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    dataset_label: str,
) -> dict:
    """Run inference and compute metrics for one evaluation split."""
    prompts = df["prompt"].tolist()
    logger.info("Running inference on %s (%s examples)...", dataset_label, f"{len(prompts):,}")
    raw_outputs = run_inference(model, tokenizer, prompts, max_new_tokens, batch_size)

    df = df.copy()
    df["predicted"] = [parse_verdict(o) for o in raw_outputs]
    df["raw_output"] = raw_outputs

    valid = df[df["predicted"].isin(VERDICT_LABELS)].copy()
    skip_rate = 1 - len(valid) / len(df)
    if skip_rate > 0:
        logger.warning("%.1f%% predictions could not be parsed", skip_rate * 100)

    df.loc[~df["predicted"].isin(VERDICT_LABELS), "predicted"] = INCORRECT

    y_true_3 = df["judge_verdict"].tolist()
    y_pred_3 = df["predicted"].tolist()

    f1_macro = f1_score(
        y_true_3, y_pred_3, labels=VERDICT_LABELS, average="macro", zero_division=0
    )

    y_true_bin = [1 if v == INCORRECT else 0 for v in y_true_3]
    y_pred_bin = [1 if v == INCORRECT else 0 for v in y_pred_3]
    try:
        auc = roc_auc_score(y_true_bin, y_pred_bin)
    except ValueError:
        auc = float("nan")

    logger.info("%s  F1=%.3f  AUC=%.3f", dataset_label, f1_macro, auc)
    logger.info("\n%s", classification_report(y_true_3, y_pred_3, labels=VERDICT_LABELS, zero_division=0))

    result = {
        "dataset": dataset_label,
        "n_total": len(df),
        "n_valid": len(valid),
        "skip_rate": round(skip_rate, 4),
        "f1_macro": round(f1_macro, 4),
        "auc": round(auc, 4),
        "predictions": df[
            [c for c in ["source", "target", "domain", "judge_verdict", "predicted", "classification", "is_corrupted"] if c in df.columns]
        ].to_dict(orient="records"),
    }

    cm = confusion_matrix(y_true_3, y_pred_3, labels=VERDICT_LABELS)
    result["confusion_matrix"] = cm.tolist()

    return result


def plot_confusion(results: list[dict], output_dir: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = np.array(res["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(VERDICT_LABELS)))
        ax.set_yticks(range(len(VERDICT_LABELS)))
        ax.set_xticklabels(["COR", "PART", "INCOR"], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(["COR", "PART", "INCOR"], fontsize=8)
        ax.set_title(
            f"{res['model_label']}\n{res['dataset']}\nF1={res['f1_macro']:.3f}  AUC={res['auc']:.3f}",
            fontsize=9,
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i,j]}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()


def print_summary_table(all_results: list[dict]) -> None:
    rows = []
    rows.append({
        "model": "GPT-4.1 Mechanistic (thesis)",
        "GT Synth F1": f"{THESIS_BASELINES['GT Synth']['f1']:.2f}",
        "GT Synth AUC": f"{THESIS_BASELINES['GT Synth']['auc']:.2f}",
        "GT Synth Val loss": "-",
        "GT Lit F1": f"{THESIS_BASELINES['GT Lit']['f1']:.2f}",
        "GT Lit AUC": f"{THESIS_BASELINES['GT Lit']['auc']:.2f}",
        "Cost/1k": "~$0.30",
    })

    model_names = list(dict.fromkeys(r["model_label"] for r in all_results))
    for model_name in model_names:
        row = {"model": model_name, "Cost/1k": "~$0.001"}
        for r in all_results:
            if r["model_label"] != model_name:
                continue
            if "Synth" in r["dataset"]:
                row["GT Synth F1"] = f"{r['f1_macro']:.3f}"
                row["GT Synth AUC"] = f"{r['auc']:.3f}"
                if "eval_loss" in r and not np.isnan(r.get("eval_loss", float("nan"))):
                    row["GT Synth Val loss"] = f"{r['eval_loss']:.4f}"
            elif "Lit" in r["dataset"]:
                row["GT Lit F1"] = f"{r['f1_macro']:.3f}"
                row["GT Lit AUC"] = f"{r['auc']:.3f}"
        if "GT Synth Val loss" not in row:
            row["GT Synth Val loss"] = "-"
        rows.append(row)

    df = pd.DataFrame(rows).fillna("-")
    logger.info("\n" + "═" * 80)
    logger.info("SUMMARY TABLE — GT Synth→GT Lit Transfer Gap Experiment")
    logger.info("═" * 80)
    logger.info("\n%s", df.to_string(index=False))
    logger.info("═" * 80)
    logger.info("Key result: compare 'Mistral-7B QLoRA' GT Lit AUC vs GPT-4.1 baseline")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    val_df = pd.read_excel(args.val_path).dropna(subset=["prompt", "judge_verdict"])
    lit_df = pd.read_excel(args.lit_path).dropna(subset=["prompt", "judge_verdict"])
    logger.info("Loaded: GT Synth val = %s rows | GT Lit = %s rows", f"{len(val_df):,}", f"{len(lit_df):,}")

    val_df_loss = val_df.dropna(subset=["completion"]) if "completion" in val_df.columns else pd.DataFrame()

    all_results: list[dict] = []
    loss_finetuned = loss_base = float("nan")

    is_peft = os.path.exists(os.path.join(args.finetuned_model, "adapter_config.json"))
    model_label = "Mistral-7B QLoRA (GT Synth trained)"
    logger.info("Loading finetuned model: %s (peft=%s)", args.finetuned_model, is_peft)
    model, tokenizer = _load_model_tokenizer(args.finetuned_model, args.base_model, is_peft)

    if len(val_df_loss) > 0:
        logger.info("Computing eval loss (GT Synth Val) for finetuned model...")
        loss_finetuned = compute_eval_loss(model, tokenizer, val_df_loss, batch_size=args.batch_size)
        logger.info("Finetuned eval_loss (GT Synth Val) = %.4f", loss_finetuned)

    for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
        res = evaluate_on_df(model, tokenizer, df, args.max_new_tokens, args.batch_size, label)
        res["model_label"] = model_label
        if "Synth" in label and not np.isnan(loss_finetuned):
            res["eval_loss"] = loss_finetuned
        all_results.append(res)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.eval_base:
        base_label = "Mistral-7B base (zero-shot)"
        logger.info("Loading base model: %s", args.base_model)
        base_model, tokenizer = _load_model_tokenizer(
            args.base_model, args.base_model, is_peft=False
        )
        if len(val_df_loss) > 0:
            logger.info("Computing eval loss (GT Synth Val) for base model...")
            loss_base = compute_eval_loss(base_model, tokenizer, val_df_loss, batch_size=args.batch_size)
            logger.info("Base eval_loss (GT Synth Val) = %.4f", loss_base)

        for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
            res = evaluate_on_df(base_model, tokenizer, df, args.max_new_tokens, args.batch_size, label)
            res["model_label"] = base_label
            if "Synth" in label and not np.isnan(loss_base):
                res["eval_loss"] = loss_base
            all_results.append(res)

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not np.isnan(loss_finetuned) or not np.isnan(loss_base):
        logger.info("\n" + "═" * 80)
        logger.info("LOSS BASELINE (GT Synth Val) — cross-entropy on completion tokens")
        logger.info("═" * 80)
        logger.info("  Base (zero-shot):    %.4f", loss_base)
        logger.info("  Finetuned (QLoRA):   %.4f", loss_finetuned)
        logger.info("═" * 80)

    print_summary_table(all_results)
    plot_confusion(all_results, args.output_dir)

    save_results = [{k: v for k, v in r.items() if k != "predictions"} for r in all_results]
    out_json = os.path.join(args.output_dir, "evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info("Results saved to %s", out_json)

    for r in all_results:
        safe_label = r["model_label"].replace(" ", "_").replace("(", "").replace(")", "")
        safe_set = r["dataset"].replace(" ", "_")
        pred_df = pd.DataFrame(r["predictions"])
        pred_path = os.path.join(args.output_dir, f"predictions_{safe_label}_{safe_set}.xlsx")
        pred_df.to_excel(pred_path, index=False)
        logger.info("Predictions saved to %s", pred_path)


if __name__ == "__main__":
    main()
