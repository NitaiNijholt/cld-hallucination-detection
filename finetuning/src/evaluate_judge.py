#!/usr/bin/env python3
"""
Evaluate a finetuned CLD judge model on GT Synth (in-distribution) and
GT Lit (out-of-distribution) to measure the transfer gap.

Evaluation protocol:
  - Load finetuned LoRA adapter (or merged checkpoint) + base model
  - Optionally also evaluate base (zero-shot) for direct comparison
  - Run inference on:
      src/judge_val_synth.xlsx   — GT Synth val   (in-dist)
      src/judge_eval_gtlit.xlsx  — GT Lit          (out-of-dist)
  - Parse VERDICT token from model output
  - Compute:
      F1 macro  (3-class: CORRECT / PARTIALLY_CORRECT / INCORRECT)
      AUC       (binary hallucination: INCORRECT=1 vs. rest=0)
      For GT Lit also uses Classification (TP/TN/FP/FN) for expert-vs-model agreement
  - Print comparison table against GPT-4.1 thesis baselines
  - Save results/evaluation_results.json and results/confusion_matrices.png

Usage:
    python src/evaluate_judge.py \
        --finetuned_model Mistral-7B-Instruct-v0.2_qlora_cld_judge_gtsynth/lora_adapter \
        [--base_model mistralai/Mistral-7B-Instruct-v0.2] \
        [--val_path src/judge_val_synth.xlsx] \
        [--lit_path src/judge_eval_gtlit.xlsx] \
        [--max_new_tokens 512] \
        [--batch_size 4]
"""

import argparse
import json
import os
import re
import warnings

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

# ── GPT-4.1 thesis baselines (mechanistic prompt, from thesis) ────────────────
THESIS_BASELINES = {
    "GT Synth": {"f1": 0.74, "auc": 0.86},   # mid-range from thesis table
    "GT Lit":   {"f1": 0.35, "auc": 0.60},   # mid-range, varies by domain
}

VERDICT_LABELS  = ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
HALLUC_POSITIVE = "INCORRECT"   # binary positive class for AUC


# ── Argument parsing ──────────────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CLD judge model")
    p.add_argument("--finetuned_model", default="Mistral-7B-Instruct-v0.2_qlora_cld_judge_gtsynth/lora_adapter",
                   help="Path to finetuned LoRA adapter directory (or merged checkpoint)")
    p.add_argument("--base_model",      default="mistralai/Mistral-7B-Instruct-v0.2",
                   help="HuggingFace base model id or path")
    p.add_argument("--eval_base",       action="store_true",
                   help="Also evaluate base model zero-shot for comparison")
    p.add_argument("--val_path",        default="src/judge_val_synth.xlsx")
    p.add_argument("--lit_path",        default="src/judge_eval_gtlit.xlsx")
    p.add_argument("--max_new_tokens",  type=int, default=512)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--output_dir",      default="results")
    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────
def _parse_verdict(text: str) -> str:
    """Extract VERDICT token from model output."""
    m = re.search(
        r"VERDICT\s*[:=]?\s*(CORRECT|PARTIALLY_CORRECT|INCORRECT)",
        text.upper(),
    )
    return m.group(1) if m else "UNKNOWN"


def _binary_label(verdict: str) -> int:
    """Binary hallucination label: 1 if INCORRECT else 0."""
    return 1 if verdict == INCORRECT else 0


INCORRECT = "INCORRECT"


def _load_model_tokenizer(model_path: str, base_model: str, is_peft: bool):
    """Load model (with or without LoRA adapter) in 4-bit for inference."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if is_peft:
        base = PeftModel.from_pretrained(base, model_path)
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
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Strip prompt tokens
        for j, g in enumerate(gen):
            prompt_len = enc["input_ids"][j].shape[0]
            text       = tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
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
    print(f"\n  Running inference on {dataset_label} ({len(prompts):,} examples)...")
    raw_outputs = run_inference(model, tokenizer, prompts, max_new_tokens, batch_size)

    df = df.copy()
    df["predicted"]  = [_parse_verdict(o) for o in raw_outputs]
    df["raw_output"] = raw_outputs

    # Filter parseable predictions for metrics
    valid = df[df["predicted"].isin(VERDICT_LABELS)].copy()
    skip_rate = 1 - len(valid) / len(df)
    if skip_rate > 0:
        print(f"  Warning: {skip_rate:.1%} predictions could not be parsed")

    # Remap 3-class to UNKNOWN=INCORRECT for unparseable rows
    df.loc[~df["predicted"].isin(VERDICT_LABELS), "predicted"] = "INCORRECT"

    y_true_3 = df["judge_verdict"].tolist()
    y_pred_3 = df["predicted"].tolist()

    f1_macro = f1_score(y_true_3, y_pred_3, labels=VERDICT_LABELS,
                        average="macro", zero_division=0)

    # Binary AUC: INCORRECT = hallucination positive
    y_true_bin = [1 if v == INCORRECT else 0 for v in y_true_3]
    y_pred_bin = [1 if v == INCORRECT else 0 for v in y_pred_3]
    try:
        auc = roc_auc_score(y_true_bin, y_pred_bin)
    except ValueError:
        auc = float("nan")

    print(f"  {dataset_label}  F1={f1_macro:.3f}  AUC={auc:.3f}")
    print(classification_report(y_true_3, y_pred_3, labels=VERDICT_LABELS, zero_division=0))

    result = {
        "dataset":   dataset_label,
        "n_total":   len(df),
        "n_valid":   len(valid),
        "skip_rate": round(skip_rate, 4),
        "f1_macro":  round(f1_macro, 4),
        "auc":       round(auc, 4),
        "predictions": df[["source", "target", "domain",
                            "judge_verdict", "predicted", "classification",
                            "is_corrupted"]].to_dict(orient="records"),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true_3, y_pred_3, labels=VERDICT_LABELS)
    result["confusion_matrix"] = cm.tolist()

    return result


def plot_confusion(results: list[dict], output_dir: str) -> None:
    """Plot side-by-side confusion matrices for all evaluated models/splits."""
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
        ax.set_title(f"{res['model_label']}\n{res['dataset']}\nF1={res['f1_macro']:.3f}  AUC={res['auc']:.3f}", fontsize=9)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()


def print_summary_table(all_results: list[dict]) -> None:
    rows = []

    # GPT-4.1 thesis baselines
    rows.append({
        "model": "GPT-4.1 Mechanistic (thesis)",
        "GT Synth F1":  f"{THESIS_BASELINES['GT Synth']['f1']:.2f}",
        "GT Synth AUC": f"{THESIS_BASELINES['GT Synth']['auc']:.2f}",
        "GT Lit F1":    f"{THESIS_BASELINES['GT Lit']['f1']:.2f}",
        "GT Lit AUC":   f"{THESIS_BASELINES['GT Lit']['auc']:.2f}",
        "Cost/1k":      "~$0.30",
    })

    # Evaluated models
    model_names  = list(dict.fromkeys(r["model_label"] for r in all_results))
    for model_name in model_names:
        row = {"model": model_name, "Cost/1k": "~$0.001"}
        for r in all_results:
            if r["model_label"] != model_name:
                continue
            if "Synth" in r["dataset"]:
                row["GT Synth F1"]  = f"{r['f1_macro']:.3f}"
                row["GT Synth AUC"] = f"{r['auc']:.3f}"
            elif "Lit" in r["dataset"]:
                row["GT Lit F1"]  = f"{r['f1_macro']:.3f}"
                row["GT Lit AUC"] = f"{r['auc']:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows).fillna("-")
    print("\n" + "═" * 80)
    print("SUMMARY TABLE — GT Synth→GT Lit Transfer Gap Experiment")
    print("═" * 80)
    print(df.to_string(index=False))
    print("═" * 80)
    print("\nKey result: compare 'Mistral-7B QLoRA' GT Lit AUC vs GPT-4.1 baseline")
    print("  Improvement → finetuning partially closes the transfer gap")
    print("  No improvement → confirms gap requires GT Lit supervision\n")


def main() -> None:
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    val_df = pd.read_excel(args.val_path).dropna(subset=["prompt", "judge_verdict"])
    lit_df = pd.read_excel(args.lit_path).dropna(subset=["prompt", "judge_verdict"])
    print(f"Loaded: GT Synth val = {len(val_df):,} rows | GT Lit = {len(lit_df):,} rows")

    all_results: list[dict] = []

    # ── Evaluate finetuned model ──────────────────────────────────────────────
    is_peft = os.path.exists(os.path.join(args.finetuned_model, "adapter_config.json"))
    model_label = "Mistral-7B QLoRA (GT Synth trained)"
    print(f"\nLoading finetuned model: {args.finetuned_model}  (peft={is_peft})")
    model, tokenizer = _load_model_tokenizer(args.finetuned_model, args.base_model, is_peft)

    for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
        res = evaluate_on_df(model, tokenizer, df, args.max_new_tokens, args.batch_size, label)
        res["model_label"] = model_label
        all_results.append(res)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Optionally evaluate base model ────────────────────────────────────────
    if args.eval_base:
        base_label = "Mistral-7B base (zero-shot)"
        print(f"\nLoading base model: {args.base_model}")
        base_model, tokenizer = _load_model_tokenizer(
            args.base_model, args.base_model, is_peft=False
        )
        for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
            res = evaluate_on_df(base_model, tokenizer, df, args.max_new_tokens, args.batch_size, label)
            res["model_label"] = base_label
            all_results.append(res)

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(all_results)
    plot_confusion(all_results, args.output_dir)

    # Save full results (without per-row predictions for file size)
    save_results = [
        {k: v for k, v in r.items() if k != "predictions"}
        for r in all_results
    ]
    out_json = os.path.join(args.output_dir, "evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved to {out_json}")

    # Save per-row predictions to xlsx for error analysis
    for r in all_results:
        safe_label = r["model_label"].replace(" ", "_").replace("(", "").replace(")", "")
        safe_set   = r["dataset"].replace(" ", "_")
        pred_df    = pd.DataFrame(r["predictions"])
        pred_path  = os.path.join(args.output_dir, f"predictions_{safe_label}_{safe_set}.xlsx")
        pred_df.to_excel(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
