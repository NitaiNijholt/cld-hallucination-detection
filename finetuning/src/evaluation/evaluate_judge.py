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
    accuracy_score,
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
EVAL_MODES = ("teacher_verdict", "ground_truth")

from .eval_utils import INCORRECT, VERDICT_LABELS, parse_verdict
from ..metadata_utils import load_json, sha256_file, sidecar_metadata_path, write_json


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
        "--val_label",
        default="GT Synth Val",
        help="Display label for the first evaluation split",
    )
    p.add_argument(
        "--lit_label",
        default="GT Lit",
        help="Display label for the second evaluation split",
    )
    p.add_argument(
        "--benchmark_label",
        default="Judge Benchmark",
        help="Display label for summary reporting",
    )
    p.add_argument(
        "--include_thesis_baselines",
        action="store_true",
        help="Include hard-coded thesis GPT-4.1 baselines in the summary when matching labels exist",
    )
    p.add_argument(
        "--no_include_thesis_baselines",
        dest="include_thesis_baselines",
        action="store_false",
        help="Do not include hard-coded thesis GPT-4.1 baselines in the summary",
    )
    p.set_defaults(include_thesis_baselines=True)
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
        "--inference_backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend: transformers (HF) or vllm (faster batch)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per split (for smoke test); None = use all",
    )
    p.add_argument(
        "--output_dir",
        default=str(defaults["output_dir"]),
    )
    p.add_argument(
        "--model_label",
        default=None,
        help="Optional display label for the finetuned model in reports",
    )
    p.add_argument(
        "--eval_mode",
        choices=EVAL_MODES,
        default="teacher_verdict",
        help="Score against teacher verdict labels or actual ground-truth labels",
    )
    return p.parse_args()


def _dataset_metadata(path: str | Path) -> dict:
    file_path = Path(path)
    sidecar_path = sidecar_metadata_path(file_path)
    payload = {
        "path": str(file_path),
        "sha256": sha256_file(file_path) if file_path.exists() else None,
        "sidecar_path": str(sidecar_path) if sidecar_path.exists() else None,
    }
    sidecar = load_json(sidecar_path)
    if sidecar is not None:
        payload["sidecar"] = sidecar
    return payload


def _load_run_metadata_for_model(model_path: str | Path) -> dict | None:
    path = Path(model_path)
    candidates = []
    if path.is_dir():
        candidates.append(path / "run_metadata.json")
        candidates.append(path.parent / "run_metadata.json")
    else:
        candidates.append(path.parent / "run_metadata.json")
    for candidate in candidates:
        metadata = load_json(candidate)
        if metadata is not None:
            return metadata
    return None


def _default_model_label(model_path: str, run_metadata: dict | None) -> str:
    if run_metadata is None:
        return f"Mistral-7B QLoRA ({Path(model_path).name})"
    dataset_label = run_metadata.get("dataset_label", "unknown dataset")
    objective_mode = str(run_metadata.get("objective_mode", "reason_verdict")).replace("_", "+")
    return f"Mistral-7B QLoRA ({dataset_label} trained, {objective_mode})"


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
    # Load tokenizer from base model to avoid TokenizersBackend compat issues with adapter config
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct generation with decoder-only models

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


def _coerce_boolish(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "f", "no", "n"}:
        return 0
    return None


def _resolve_ground_truth_targets(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "is_corrupted" in df.columns:
        mapped = df["is_corrupted"].map(_coerce_boolish)
        usable = mapped.dropna().astype(int)
        if not usable.empty and usable.nunique() > 1:
            return mapped, "is_corrupted"

    if "classification" in df.columns:
        mapping = {"TP": 0, "TN": 0, "FP": 1, "FN": 1}
        mapped = df["classification"].astype(str).str.strip().str.upper().map(mapping)
        usable = mapped.dropna().astype(int)
        if not usable.empty and usable.nunique() > 1:
            return mapped, "classification"

    raise ValueError(
        "Ground-truth eval mode requires either `is_corrupted` with both classes present "
        "or `classification` containing TP/TN/FP/FN."
    )


def _predicted_hallucination_label(verdict: str) -> int:
    return int(verdict in {INCORRECT, "PARTIALLY_CORRECT"})


def evaluate_on_df(
    df: pd.DataFrame,
    max_new_tokens: int,
    batch_size: int,
    dataset_label: str,
    model=None,
    tokenizer=None,
    run_inference_fn=None,
    eval_mode: str = "teacher_verdict",
) -> dict:
    """Run inference and compute metrics for one evaluation split.
    Pass either (model, tokenizer) for HF, or run_inference_fn for vLLM.
    """
    prompts = df["prompt"].tolist()
    logger.info("Running inference on %s (%s examples)...", dataset_label, f"{len(prompts):,}")
    if run_inference_fn is not None:
        raw_outputs = run_inference_fn(prompts)
    else:
        raw_outputs = run_inference(model, tokenizer, prompts, max_new_tokens, batch_size)

    df = df.copy()
    df["predicted"] = [parse_verdict(o) for o in raw_outputs]
    df["raw_output"] = raw_outputs
    df["parse_valid"] = df["predicted"].isin(VERDICT_LABELS)

    valid = df[df["parse_valid"]].copy()
    skip_rate = 1 - len(valid) / len(df)
    if skip_rate > 0:
        logger.warning("%.1f%% predictions could not be parsed", skip_rate * 100)

    df.loc[~df["predicted"].isin(VERDICT_LABELS), "predicted"] = INCORRECT

    if eval_mode == "teacher_verdict":
        y_true = df["judge_verdict"].tolist()
        y_pred = df["predicted"].tolist()
        f1_value = f1_score(
            y_true, y_pred, labels=VERDICT_LABELS, average="macro", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        y_true_auc = [1 if v == INCORRECT else 0 for v in y_true]
        y_pred_auc = [1 if v == INCORRECT else 0 for v in y_pred]
        try:
            auc = roc_auc_score(y_true_auc, y_pred_auc)
        except ValueError:
            auc = float("nan")
        report = classification_report(y_true, y_pred, labels=VERDICT_LABELS, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=VERDICT_LABELS)
        cm_labels = VERDICT_LABELS
        truth_source = "judge_verdict"
        metric_name = "macro_f1_teacher_verdict"
    else:
        y_true_series, truth_source = _resolve_ground_truth_targets(df)
        gt_mask = y_true_series.notna()
        y_true = y_true_series[gt_mask].astype(int).tolist()
        y_pred = [
            _predicted_hallucination_label(v)
            for v in df.loc[gt_mask, "predicted"].tolist()
        ]
        f1_value = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float("nan")
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["clean", "hallucination"],
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_labels = ["clean", "hallucination"]
        metric_name = "binary_f1_ground_truth"

    logger.info("%s  F1=%.3f  AUC=%.3f", dataset_label, f1_value, auc)
    logger.info("\n%s", report)

    result = {
        "dataset": dataset_label,
        "n_total": len(df),
        "n_valid": len(valid),
        "n_scored": len(y_pred),
        "skip_rate": round(skip_rate, 4),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_value, 4),
        "auc": round(auc, 4),
        "eval_mode": eval_mode,
        "f1_metric": metric_name,
        "ground_truth_source": truth_source,
        "predictions": df[
            [
                c
                for c in [
                    "source",
                    "target",
                    "domain",
                    "judge_verdict",
                    "predicted",
                    "parse_valid",
                    "raw_output",
                    "classification",
                    "is_corrupted",
                ]
                if c in df.columns
            ]
        ].to_dict(orient="records"),
    }

    result["confusion_matrix"] = cm.tolist()
    result["confusion_matrix_labels"] = cm_labels

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
        labels = res.get("confusion_matrix_labels", VERDICT_LABELS)
        short_labels = []
        for label in labels:
            if label == "CORRECT":
                short_labels.append("COR")
            elif label == "PARTIALLY_CORRECT":
                short_labels.append("PART")
            elif label == "INCORRECT":
                short_labels.append("INCOR")
            elif label == "hallucination":
                short_labels.append("HALL")
            elif label == "clean":
                short_labels.append("CLEAN")
            else:
                short_labels.append(str(label))
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
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


def _summary_columns(val_label: str, lit_label: str) -> dict[str, str]:
    return {
        "val_f1": f"{val_label} F1",
        "val_auc": f"{val_label} AUC",
        "val_parse": f"{val_label} Parse%",
        "val_loss": f"{val_label} loss",
        "lit_f1": f"{lit_label} F1",
        "lit_auc": f"{lit_label} AUC",
        "lit_parse": f"{lit_label} Parse%",
    }


def print_summary_table(
    all_results: list[dict],
    *,
    val_label: str,
    lit_label: str,
    benchmark_label: str,
    include_thesis_baselines: bool,
) -> list[dict]:
    columns = _summary_columns(val_label, lit_label)
    rows = []
    thesis_baseline_row = None
    if include_thesis_baselines and val_label in THESIS_BASELINES and lit_label in THESIS_BASELINES:
        thesis_baseline_row = {
            "model": "GPT-4.1 Mechanistic (thesis)",
            columns["val_f1"]: f"{THESIS_BASELINES[val_label]['f1']:.2f}",
            columns["val_auc"]: f"{THESIS_BASELINES[val_label]['auc']:.2f}",
            columns["val_parse"]: "-",
            columns["val_loss"]: "-",
            columns["lit_f1"]: f"{THESIS_BASELINES[lit_label]['f1']:.2f}",
            columns["lit_auc"]: f"{THESIS_BASELINES[lit_label]['auc']:.2f}",
            columns["lit_parse"]: "-",
            "Cost/1k": "~$0.30",
        }
        rows.append(thesis_baseline_row)

    model_names = list(dict.fromkeys(r["model_label"] for r in all_results))
    for model_name in model_names:
        row = {"model": model_name, "Cost/1k": "~$0.001"}
        for r in all_results:
            if r["model_label"] != model_name:
                continue
            if r["dataset"] == val_label:
                row[columns["val_f1"]] = f"{r['f1_macro']:.3f}"
                row[columns["val_auc"]] = f"{r['auc']:.3f}"
                row[columns["val_parse"]] = f"{(1 - r['skip_rate']) * 100:.1f}"
                if "eval_loss" in r and not np.isnan(r.get("eval_loss", float("nan"))):
                    row[columns["val_loss"]] = f"{r['eval_loss']:.4f}"
            elif r["dataset"] == lit_label:
                row[columns["lit_f1"]] = f"{r['f1_macro']:.3f}"
                row[columns["lit_auc"]] = f"{r['auc']:.3f}"
                row[columns["lit_parse"]] = f"{(1 - r['skip_rate']) * 100:.1f}"
        if columns["val_loss"] not in row:
            row[columns["val_loss"]] = "-"
        if columns["val_parse"] not in row:
            row[columns["val_parse"]] = "-"
        if columns["lit_parse"] not in row:
            row[columns["lit_parse"]] = "-"
        rows.append(row)

    df = pd.DataFrame(rows).fillna("-")
    logger.info("\n" + "═" * 80)
    logger.info("SUMMARY TABLE — %s", benchmark_label)
    logger.info("═" * 80)
    logger.info("\n%s", df.to_string(index=False))
    logger.info("═" * 80)
    return rows, thesis_baseline_row


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    val_df = pd.read_excel(args.val_path).dropna(subset=["prompt"])
    lit_df = pd.read_excel(args.lit_path).dropna(subset=["prompt"])
    if args.max_samples is not None:
        val_df = val_df.head(args.max_samples)
        lit_df = lit_df.head(args.max_samples)
        logger.info("Smoke test: limited to %s samples per split", args.max_samples)
    logger.info(
        "Loaded: %s = %s rows | %s = %s rows",
        args.val_label,
        f"{len(val_df):,}",
        args.lit_label,
        f"{len(lit_df):,}",
    )

    val_df_loss = val_df.dropna(subset=["completion"]) if "completion" in val_df.columns else pd.DataFrame()

    all_results: list[dict] = []
    loss_finetuned = loss_base = float("nan")

    is_peft = os.path.exists(os.path.join(args.finetuned_model, "adapter_config.json"))
    finetuned_run_metadata = _load_run_metadata_for_model(args.finetuned_model)
    model_label = args.model_label or _default_model_label(args.finetuned_model, finetuned_run_metadata)
    use_vllm = args.inference_backend == "vllm"

    if use_vllm:
        from .vllm_inference import run_inference_vllm
        lora_path = args.finetuned_model if is_peft else None
        base_for_finetuned = args.base_model if is_peft else args.finetuned_model
        run_fn_finetuned = lambda p: run_inference_vllm(
            p, base_for_finetuned, lora_path, args.max_new_tokens, args.batch_size
        )
        for df, label in [(val_df, args.val_label), (lit_df, args.lit_label)]:
            res = evaluate_on_df(
                df, args.max_new_tokens, args.batch_size, label,
                run_inference_fn=run_fn_finetuned,
                eval_mode=args.eval_mode,
            )
            res["model_label"] = model_label
            all_results.append(res)

        if args.eval_base:
            base_label = "Mistral-7B base (zero-shot)"
            run_fn_base = lambda p: run_inference_vllm(
                p, args.base_model, None, args.max_new_tokens, args.batch_size
            )
            for df, label in [(val_df, args.val_label), (lit_df, args.lit_label)]:
                res = evaluate_on_df(
                    df, args.max_new_tokens, args.batch_size, label,
                    run_inference_fn=run_fn_base,
                    eval_mode=args.eval_mode,
                )
                res["model_label"] = base_label
                all_results.append(res)
    else:
        logger.info("Loading finetuned model: %s (peft=%s)", args.finetuned_model, is_peft)
        model, tokenizer = _load_model_tokenizer(args.finetuned_model, args.base_model, is_peft)

        if len(val_df_loss) > 0:
            logger.info("Computing eval loss (%s) for finetuned model...", args.val_label)
            loss_finetuned = compute_eval_loss(model, tokenizer, val_df_loss, batch_size=args.batch_size)
            logger.info("Finetuned eval_loss (%s) = %.4f", args.val_label, loss_finetuned)

        for df, label in [(val_df, args.val_label), (lit_df, args.lit_label)]:
            res = evaluate_on_df(
                df, args.max_new_tokens, args.batch_size, label,
                model=model, tokenizer=tokenizer,
                eval_mode=args.eval_mode,
            )
            res["model_label"] = model_label
            if label == args.val_label and not np.isnan(loss_finetuned):
                res["eval_loss"] = loss_finetuned
            all_results.append(res)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.eval_base:
            base_label = "Mistral-7B base (zero-shot)"
            logger.info("Loading base model: %s", args.base_model)
            base_model_obj, tokenizer = _load_model_tokenizer(
                args.base_model, args.base_model, is_peft=False
            )
            if len(val_df_loss) > 0:
                logger.info("Computing eval loss (%s) for base model...", args.val_label)
                loss_base = compute_eval_loss(base_model_obj, tokenizer, val_df_loss, batch_size=args.batch_size)
                logger.info("Base eval_loss (%s) = %.4f", args.val_label, loss_base)

            for df, label in [(val_df, args.val_label), (lit_df, args.lit_label)]:
                res = evaluate_on_df(
                    df, args.max_new_tokens, args.batch_size, label,
                    model=base_model_obj, tokenizer=tokenizer,
                    eval_mode=args.eval_mode,
                )
                res["model_label"] = base_label
                if label == args.val_label and not np.isnan(loss_base):
                    res["eval_loss"] = loss_base
                all_results.append(res)

            del base_model_obj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not np.isnan(loss_finetuned) or not np.isnan(loss_base):
        logger.info("\n" + "═" * 80)
        logger.info("LOSS BASELINE (%s) — cross-entropy on completion tokens", args.val_label)
        logger.info("═" * 80)
        logger.info("  Base (zero-shot):    %.4f", loss_base)
        logger.info("  Finetuned (QLoRA):   %.4f", loss_finetuned)
        logger.info("═" * 80)

    _, thesis_baseline_row = print_summary_table(
        all_results,
        val_label=args.val_label,
        lit_label=args.lit_label,
        benchmark_label=args.benchmark_label,
        include_thesis_baselines=args.include_thesis_baselines,
    )
    plot_confusion(all_results, args.output_dir)

    baseline_results = []
    if thesis_baseline_row is not None:
        for dataset_name in [args.val_label, args.lit_label]:
            metrics = THESIS_BASELINES[dataset_name]
            baseline_results.append(
                {
                    "model_label": "GPT-4.1 Mechanistic (thesis)",
                    "dataset": dataset_name,
                    "f1_macro": metrics["f1"],
                    "auc": metrics["auc"],
                    "source": "thesis",
                    "kind": "external_baseline",
                }
            )
    save_results = [{k: v for k, v in r.items() if k != "predictions"} for r in all_results]
    save_results.extend(baseline_results)
    out_json = os.path.join(args.output_dir, "evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info("Results saved to %s", out_json)
    write_json(
        os.path.join(args.output_dir, "evaluation_metadata.json"),
        {
            "finetuned_model": {
                "path": str(args.finetuned_model),
                "run_metadata": finetuned_run_metadata,
            },
            "base_model": args.base_model,
            "inference_backend": args.inference_backend,
            "datasets": {
                "first_split": {
                    "label": args.val_label,
                    **_dataset_metadata(args.val_path),
                },
                "second_split": {
                    "label": args.lit_label,
                    **_dataset_metadata(args.lit_path),
                },
            },
            "benchmark_label": args.benchmark_label,
            "eval_mode": args.eval_mode,
            "include_thesis_baselines": args.include_thesis_baselines,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
        },
    )

    for r in all_results:
        safe_label = r["model_label"].replace(" ", "_").replace("(", "").replace(")", "")
        safe_set = r["dataset"].replace(" ", "_")
        pred_df = pd.DataFrame(r["predictions"])
        pred_path = os.path.join(args.output_dir, f"predictions_{safe_label}_{safe_set}.xlsx")
        pred_df.to_excel(pred_path, index=False)
        logger.info("Predictions saved to %s", pred_path)


if __name__ == "__main__":
    main()
