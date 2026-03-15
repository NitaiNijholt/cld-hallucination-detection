#!/usr/bin/env python3
"""
Create subsampled eval datasets (~1000 samples each) for faster evaluation.

Usage:
    python -m finetuning.src.evaluation.create_eval_subsamples
    python -m finetuning.src.evaluation.create_eval_subsamples --target_n 500 --seed 42

Outputs: judge_val_synth_1k.xlsx, judge_eval_gtlit_1k.xlsx (in same dir as inputs)
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def _get_default_paths():
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    root = Path(env_root) if env_root else Path(__file__).resolve().parent.parent.parent.parent
    src = root / "finetuning" / "src"
    return {
        "val_path": src / "judge_val_synth.xlsx",
        "lit_path": src / "judge_eval_gtlit.xlsx",
    }


def main() -> None:
    defaults = _get_default_paths()
    p = argparse.ArgumentParser(description="Create subsampled eval datasets")
    p.add_argument("--val_path", default=str(defaults["val_path"]))
    p.add_argument("--lit_path", default=str(defaults["lit_path"]))
    p.add_argument("--target_n", type=int, default=1000, help="Target samples per split")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for label, path in [("GT Synth Val", args.val_path), ("GT Lit", args.lit_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
        df = pd.read_excel(path).dropna(subset=["prompt", "judge_verdict"])
        n_orig = len(df)
        if n_orig <= args.target_n:
            sub = df
            out_path = path.replace(".xlsx", "_1k.xlsx")
            sub.to_excel(out_path, index=False)
            print(f"{label}: {n_orig} rows (no subsample needed) -> {out_path}")
        else:
            sub = df.sample(n=args.target_n, random_state=args.seed).reset_index(drop=True)
            out_path = path.replace(".xlsx", "_1k.xlsx")
            sub.to_excel(out_path, index=False)
            print(f"{label}: {n_orig} -> {args.target_n} (seed={args.seed}) -> {out_path}")


if __name__ == "__main__":
    main()
