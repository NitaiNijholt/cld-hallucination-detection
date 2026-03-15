#!/usr/bin/env python3
"""
Classification-only eval: finetuned vs base (zero-shot) on GT Synth Val and GT Lit.

Runs inference, parses verdicts, computes F1/AUC for both models. No loss computation.
Use as a second eval after training, or when you only need classification metrics.

Usage:
    python -m finetuning.src.evaluation.eval_classification_compare \
        --finetuned_model finetuning/runs/mistral7b_judge_gtsynth/lora_adapter \
        --val_path finetuning/src/judge_val_synth.xlsx \
        --lit_path finetuning/src/judge_eval_gtlit.xlsx \
        --output_dir results
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch

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


def _get_default_paths():
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    root = Path(env_root) if env_root else Path(__file__).resolve().parent.parent.parent.parent
    return {
        "val_path": root / "finetuning" / "src" / "judge_val_synth.xlsx",
        "lit_path": root / "finetuning" / "src" / "judge_eval_gtlit.xlsx",
        "output_dir": root / "results",
    }


def parse_args() -> argparse.Namespace:
    defaults = _get_default_paths()
    p = argparse.ArgumentParser(description="Classification comparison: finetuned vs base")
    p.add_argument("--finetuned_model", required=True, help="Path to LoRA adapter or merged checkpoint")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model id")
    p.add_argument("--val_path", default=str(defaults["val_path"]))
    p.add_argument("--lit_path", default=str(defaults["lit_path"]))
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--inference_backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend: transformers (HF) or vllm",
    )
    p.add_argument("--output_dir", default=str(defaults["output_dir"]))
    return p.parse_args()


def main() -> None:
    from .evaluate_judge import (
        _load_model_tokenizer,
        evaluate_on_df,
        plot_confusion,
    )

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    val_df = pd.read_excel(args.val_path).dropna(subset=["prompt", "judge_verdict"])
    lit_df = pd.read_excel(args.lit_path).dropna(subset=["prompt", "judge_verdict"])
    logger.info("Loaded: GT Synth val = %s | GT Lit = %s", f"{len(val_df):,}", f"{len(lit_df):,}")

    all_results: list[dict] = []
    is_peft = os.path.exists(os.path.join(args.finetuned_model, "adapter_config.json"))
    use_vllm = args.inference_backend == "vllm"

    if use_vllm:
        from .vllm_inference import run_inference_vllm
        lora_path = args.finetuned_model if is_peft else None
        base_for_finetuned = args.base_model if is_peft else args.finetuned_model
        run_fn_finetuned = lambda p: run_inference_vllm(
            p, base_for_finetuned, lora_path, args.max_new_tokens, args.batch_size
        )
        for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
            res = evaluate_on_df(
                df, args.max_new_tokens, args.batch_size, label,
                run_inference_fn=run_fn_finetuned,
            )
            res["model_label"] = "Mistral-7B QLoRA (finetuned)"
            all_results.append(res)

        run_fn_base = lambda p: run_inference_vllm(
            p, args.base_model, None, args.max_new_tokens, args.batch_size
        )
        for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
            res = evaluate_on_df(
                df, args.max_new_tokens, args.batch_size, label,
                run_inference_fn=run_fn_base,
            )
            res["model_label"] = "Mistral-7B base (zero-shot)"
            all_results.append(res)
    else:
        # Finetuned
        logger.info("Loading finetuned model: %s", args.finetuned_model)
        model, tokenizer = _load_model_tokenizer(args.finetuned_model, args.base_model, is_peft)
        for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
            res = evaluate_on_df(
                df, args.max_new_tokens, args.batch_size, label,
                model=model, tokenizer=tokenizer,
            )
            res["model_label"] = "Mistral-7B QLoRA (finetuned)"
            all_results.append(res)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Base (zero-shot)
        logger.info("Loading base model: %s", args.base_model)
        base_model_obj, tokenizer = _load_model_tokenizer(args.base_model, args.base_model, is_peft=False)
        for df, label in [(val_df, "GT Synth Val"), (lit_df, "GT Lit")]:
            res = evaluate_on_df(
                df, args.max_new_tokens, args.batch_size, label,
                model=base_model_obj, tokenizer=tokenizer,
            )
            res["model_label"] = "Mistral-7B base (zero-shot)"
            all_results.append(res)
        del base_model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    rows = [
        {
            "model": "GPT-4.1 Mechanistic (thesis)",
            "GT Synth F1": f"{THESIS_BASELINES['GT Synth']['f1']:.2f}",
            "GT Synth AUC": f"{THESIS_BASELINES['GT Synth']['auc']:.2f}",
            "GT Lit F1": f"{THESIS_BASELINES['GT Lit']['f1']:.2f}",
            "GT Lit AUC": f"{THESIS_BASELINES['GT Lit']['auc']:.2f}",
        }
    ]
    for model_name in ["Mistral-7B base (zero-shot)", "Mistral-7B QLoRA (finetuned)"]:
        row = {"model": model_name}
        for r in all_results:
            if r["model_label"] != model_name:
                continue
            if "Synth" in r["dataset"]:
                row["GT Synth F1"] = f"{r['f1_macro']:.3f}"
                row["GT Synth AUC"] = f"{r['auc']:.3f}"
            elif "Lit" in r["dataset"]:
                row["GT Lit F1"] = f"{r['f1_macro']:.3f}"
                row["GT Lit AUC"] = f"{r['auc']:.3f}"
        rows.append(row)

    df_out = pd.DataFrame(rows).fillna("-")
    logger.info("\n" + "═" * 60)
    logger.info("CLASSIFICATION — Finetuned vs Base")
    logger.info("═" * 60)
    logger.info("\n%s", df_out.to_string(index=False))
    logger.info("═" * 60)

    plot_confusion(all_results, args.output_dir)

    save_results = [{k: v for k, v in r.items() if k != "predictions"} for r in all_results]
    out_json = os.path.join(args.output_dir, "classification_compare_results.json")
    with open(out_json, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info("Results saved to %s", out_json)

    for r in all_results:
        safe = r["model_label"].replace(" ", "_").replace("(", "").replace(")", "")
        pred_path = os.path.join(args.output_dir, f"predictions_{safe}_{r['dataset'].replace(' ', '_')}.xlsx")
        pd.DataFrame(r["predictions"]).to_excel(pred_path, index=False)
        logger.info("Predictions saved to %s", pred_path)


if __name__ == "__main__":
    main()
