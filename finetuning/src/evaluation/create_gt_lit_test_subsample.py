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
from sklearn.model_selection import train_test_split

from ..metadata_utils import (
    atomic_to_excel,
    load_json,
    read_excel_with_retry,
    sha256_file,
    sidecar_metadata_path,
    summarize_counts,
    write_json,
)


def _get_default_paths():
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    root = Path(env_root) if env_root else Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root / "finetuning" / "data" / "gt_lit_trainval"
    return {
        "input_path": data_dir / "judge_test_gtlit.xlsx",
        "output_path": data_dir / "judge_test_gtlit_1k.xlsx",
    }


def parse_args() -> argparse.Namespace:
    defaults = _get_default_paths()
    p = argparse.ArgumentParser(description="Create held-out GT Lit test subsample")
    p.add_argument("--input_path", default=str(defaults["input_path"]))
    p.add_argument("--output_path", default=str(defaults["output_path"]))
    p.add_argument("--target_n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stratify-cols",
        default="domain,judge_verdict",
        help="Comma-separated columns to preserve in the held-out GT Lit subset",
    )
    p.add_argument(
        "--require-stratification",
        action="store_true",
        help="Fail instead of falling back to an unstratified subset when requested strata cannot be preserved",
    )
    return p.parse_args()


def _parse_stratify_cols(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_stratify_labels(df: pd.DataFrame, stratify_cols: list[str]) -> pd.Series | None:
    usable_cols = [col for col in stratify_cols if col in df.columns]
    if not usable_cols:
        return None
    return df[usable_cols].fillna("unknown").astype(str).agg("||".join, axis=1)


def _stratified_subsample(
    df: pd.DataFrame,
    target_n: int,
    seed: int,
    stratify_cols: list[str],
    require_stratification: bool = False,
) -> pd.DataFrame:
    if len(df) <= target_n:
        return df.reset_index(drop=True)
    labels = _build_stratify_labels(df, stratify_cols)
    stratify = None
    if labels is not None:
        counts = labels.value_counts()
        remaining = len(df) - target_n
        if len(counts) > 1 and counts.min() >= 2 and target_n >= len(counts) and remaining >= len(counts):
            stratify = labels
        elif require_stratification:
            raise ValueError(
                f"Unable to preserve requested stratification on {stratify_cols}; "
                "dataset does not have enough rows per class for a frozen subset"
            )
    _, sub = train_test_split(
        df,
        test_size=target_n,
        random_state=seed,
        stratify=stratify,
    )
    return sub.reset_index(drop=True)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    stratify_cols = _parse_stratify_cols(args.stratify_cols)

    df = read_excel_with_retry(args.input_path).dropna(subset=["prompt", "judge_verdict"])
    n_orig = len(df)
    sub = _stratified_subsample(
        df,
        args.target_n,
        args.seed,
        stratify_cols,
        require_stratification=getattr(args, "require_stratification", False),
    )
    atomic_to_excel(sub, args.output_path)
    input_metadata = load_json(sidecar_metadata_path(args.input_path))
    payload = {
        "split_name": "gt_lit_test_frozen_subset",
        "source_path": str(args.input_path),
        "source_path_sha256": sha256_file(args.input_path),
        "file_name": Path(args.output_path).name,
        "file_sha256": sha256_file(args.output_path),
        "seed": args.seed,
        "stratify_cols": stratify_cols,
        "row_count": int(len(sub)),
        "domain_counts": summarize_counts(sub["domain"].tolist()) if "domain" in sub.columns else {},
        "judge_verdict_counts": summarize_counts(sub["judge_verdict"].tolist()) if "judge_verdict" in sub.columns else {},
    }
    if input_metadata is not None:
        payload["source_dataset_metadata"] = input_metadata
    write_json(sidecar_metadata_path(args.output_path), payload)
    print(f"GT Lit test: {n_orig} -> {len(sub)} (seed={args.seed}) -> {args.output_path}")


if __name__ == "__main__":
    main()
