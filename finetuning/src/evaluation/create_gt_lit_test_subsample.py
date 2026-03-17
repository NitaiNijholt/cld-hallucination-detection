#!/usr/bin/env python3
"""
Create a held-out GT Lit test subsample from the GT Lit test split.

This is intended for evaluating models trained on GT Lit without leaking
training/validation data into the reported GT Lit results.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def _get_default_paths():
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    root = Path(env_root) if env_root else Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root / "finetuning" / "data" / "gt_lit_trainval"
    return {
        "input_path": data_dir / "judge_test_gtlit.xlsx",
        "output_path": data_dir / "judge_test_gtlit_1k.xlsx",
    }


def main() -> None:
    defaults = _get_default_paths()
    p = argparse.ArgumentParser(description="Create held-out GT Lit test subsample")
    p.add_argument("--input_path", default=str(defaults["input_path"]))
    p.add_argument("--output_path", default=str(defaults["output_path"]))
    p.add_argument("--target_n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_excel(args.input_path).dropna(subset=["prompt", "judge_verdict"])
    n_orig = len(df)
    if n_orig <= args.target_n:
        sub = df
    else:
        sub = df.sample(n=args.target_n, random_state=args.seed).reset_index(drop=True)
    sub.to_excel(args.output_path, index=False)
    print(f"GT Lit test: {n_orig} -> {len(sub)} (seed={args.seed}) -> {args.output_path}")


if __name__ == "__main__":
    main()
