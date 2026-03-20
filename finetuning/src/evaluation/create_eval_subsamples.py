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
from sklearn.model_selection import train_test_split

from ..data.prepare_judge_data import _balance_rows_exact
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
    src = root / "finetuning" / "src"
    return {
        "val_path": src / "judge_val_synth.xlsx",
        "lit_path": src / "judge_eval_gtlit.xlsx",
    }


def parse_args() -> argparse.Namespace:
    defaults = _get_default_paths()
    p = argparse.ArgumentParser(description="Create subsampled eval datasets")
    p.add_argument("--val_path", default=str(defaults["val_path"]))
    p.add_argument("--lit_path", default=str(defaults["lit_path"]))
    p.add_argument("--target_n", type=int, default=1000, help="Target samples per split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stratify-cols",
        default="domain,judge_verdict",
        help="Comma-separated columns to preserve in frozen eval subsets",
    )
    p.add_argument(
        "--output-suffix",
        default="_1k_stratified",
        help="Suffix to append before .xlsx for the frozen subset",
    )
    p.add_argument(
        "--require-stratification",
        action="store_true",
        help="Fail instead of falling back to an unstratified subset when requested strata cannot be preserved",
    )
    p.add_argument(
        "--balance-cols",
        default="",
        help="Comma-separated row-level columns to exact-balance in the eval subset, e.g. domain,judge_verdict",
    )
    p.add_argument(
        "--balance-samples-per-group",
        type=int,
        default=None,
        help="Optional exact row count per balance group; defaults to min(group_size, target_n // groups)",
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


def _resolve_balance_samples_per_group(
    df: pd.DataFrame,
    balance_cols: list[str],
    target_n: int,
    requested_samples_per_group: int | None,
) -> int | None:
    if not balance_cols:
        return None
    counts = df.groupby(balance_cols, dropna=False).size()
    if counts.empty:
        return None
    if requested_samples_per_group is not None:
        return requested_samples_per_group
    return min(int(counts.min()), max(1, target_n // len(counts)))


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


def _save_subset(
    df: pd.DataFrame,
    output_path: Path,
    *,
    label: str,
    input_path: Path,
    input_metadata: dict | None,
    stratify_cols: list[str],
    seed: int,
) -> None:
    atomic_to_excel(df, output_path)
    payload = {
        "split_name": label,
        "source_path": str(input_path),
        "source_path_sha256": sha256_file(input_path),
        "file_name": output_path.name,
        "file_sha256": sha256_file(output_path),
        "seed": seed,
        "stratify_cols": stratify_cols,
        "row_count": int(len(df)),
        "domain_counts": summarize_counts(df["domain"].tolist()) if "domain" in df.columns else {},
        "judge_verdict_counts": summarize_counts(df["judge_verdict"].tolist()) if "judge_verdict" in df.columns else {},
    }
    if input_metadata is not None:
        payload["source_dataset_metadata"] = input_metadata
    write_json(sidecar_metadata_path(output_path), payload)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    stratify_cols = _parse_stratify_cols(args.stratify_cols)
    balance_cols = _parse_stratify_cols(getattr(args, "balance_cols", ""))

    for label, path in [("GT Synth Val", args.val_path), ("GT Lit", args.lit_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
        df = read_excel_with_retry(path).dropna(subset=["prompt", "judge_verdict"])
        input_path = Path(path)
        input_metadata = load_json(sidecar_metadata_path(input_path))
        n_orig = len(df)
        if balance_cols:
            sub = _balance_rows_exact(
                df,
                balance_cols,
                args.seed,
                samples_per_group=_resolve_balance_samples_per_group(
                    df,
                    balance_cols,
                    args.target_n,
                    getattr(args, "balance_samples_per_group", None),
                ),
            )
        else:
            sub = _stratified_subsample(
                df,
                args.target_n,
                args.seed,
                stratify_cols,
                require_stratification=getattr(args, "require_stratification", False),
            )
        out_path = input_path.with_name(f"{input_path.stem}{args.output_suffix}.xlsx")
        _save_subset(
            sub,
            out_path,
            label=label,
            input_path=input_path,
            input_metadata=input_metadata,
            stratify_cols=stratify_cols,
            seed=args.seed,
        )
        print(f"{label}: {n_orig} -> {len(sub)} (seed={args.seed}) -> {out_path}")


if __name__ == "__main__":
    main()
