#!/usr/bin/env python3
"""
Prepare a train/validation split from the existing GT Lit dataset.

This is intended for an additional experiment where we finetune on GT Lit
directly, without overwriting the default GT Synth-based training setup.

Outputs:
  judge_train_gtlit.xlsx
  judge_val_gtlit.xlsx
  judge_test_gtlit.xlsx
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from .prepare_judge_data import (
    _balance_rows_exact,
    _build_rows,
    _load_xlsx_files,
    _split_by_edge_identity,
)
from ..metadata_utils import (
    atomic_to_excel,
    load_json,
    read_excel_with_retry,
    sha256_file,
    sidecar_metadata_path,
    summarize_counts,
    write_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent.parent.parent.parent


def parse_args() -> argparse.Namespace:
    repo_root = _get_repo_root()
    p = argparse.ArgumentParser(description="Prepare GT Lit train/val split")
    p.add_argument(
        "--input-path",
        default=str(repo_root / "finetuning" / "src" / "judge_eval_gtlit.xlsx"),
        help="Path to the full GT Lit dataset xlsx",
    )
    p.add_argument(
        "--output-dir",
        default=str(repo_root / "finetuning" / "data" / "gt_lit_trainval"),
        help="Output directory for GT Lit train/val split",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Validation fraction for GT Lit split",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
        help="Held-out test fraction for GT Lit split",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting",
    )
    p.add_argument(
        "--stratify-cols",
        default="domain,judge_verdict",
        help="Comma-separated edge-level columns to stratify GT Lit splits on",
    )
    p.add_argument(
        "--require-stratification",
        action="store_true",
        help="Fail instead of falling back to unstratified splits when requested strata cannot be preserved",
    )
    p.add_argument(
        "--balance-cols",
        default="",
        help="Comma-separated row-level columns to exact-balance after splitting, e.g. domain,judge_verdict",
    )
    p.add_argument(
        "--balance-samples-per-group",
        type=int,
        default=None,
        help="Optional exact row count per balance group; defaults to each split's smallest group",
    )
    return p.parse_args()


def _parse_stratify_cols(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_split_metadata(
    df: pd.DataFrame,
    *,
    split_name: str,
    input_path: Path,
    input_metadata: dict | None,
    stratify_cols: list[str],
    seed: int,
) -> dict:
    unique_edges = df[["source", "target", "domain"]].drop_duplicates().shape[0] if len(df) else 0
    payload = {
        "split_name": split_name,
        "source_path": str(input_path),
        "source_path_sha256": sha256_file(input_path),
        "seed": seed,
        "stratify_cols": stratify_cols,
        "row_count": int(len(df)),
        "unique_edge_count": int(unique_edges),
        "domain_counts": summarize_counts(df["domain"].tolist()) if "domain" in df.columns else {},
        "judge_verdict_counts": summarize_counts(df["judge_verdict"].tolist()) if "judge_verdict" in df.columns else {},
    }
    if input_metadata is not None:
        payload["source_dataset_metadata"] = input_metadata
    return payload


def _save_split(df: pd.DataFrame, path: Path, metadata: dict) -> None:
    atomic_to_excel(df, path)
    write_json(
        sidecar_metadata_path(path),
        {
            **metadata,
            "file_name": path.name,
            "file_sha256": sha256_file(path),
        },
    )


def _rebuild_input_df_from_source_metadata(input_metadata: dict | None) -> pd.DataFrame | None:
    if input_metadata is None:
        return None
    source_glob = input_metadata.get("source_description")
    objective_mode = input_metadata.get("objective_mode")
    if not source_glob or not objective_mode:
        return None
    raw = _load_xlsx_files(source_glob, 0)
    prompt_variant_counts = input_metadata.get("prompt_variant_counts", {})
    if len(prompt_variant_counts) == 1:
        prompt_variant = next(iter(prompt_variant_counts))
        raw = raw[raw["prompt_variant"].eq(prompt_variant)].copy()
    return _build_rows(raw, objective_mode=objective_mode)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stratify_cols = _parse_stratify_cols(
        getattr(args, "stratify_cols", "domain,judge_verdict")
    )
    balance_cols = _parse_stratify_cols(getattr(args, "balance_cols", ""))

    logger.info("Input path: %s", input_path)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "Val fraction: %.2f, test fraction: %.2f, seed: %d",
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    input_metadata = load_json(sidecar_metadata_path(input_path))
    rebuilt_df = _rebuild_input_df_from_source_metadata(input_metadata) if balance_cols else None
    if rebuilt_df is not None:
        df = rebuilt_df
        logger.info("Rebuilt GT Lit split source from metadata-backed raw judged files")
    else:
        df = read_excel_with_retry(input_path).dropna(subset=["prompt", "completion"])
    logger.info("Loaded GT Lit rows: %s", f"{len(df):,}")
    logger.info(
        "Unique edges: %s",
        f"{df[['source', 'target', 'domain']].drop_duplicates().shape[0]:,}",
    )
    logger.info("Verdict dist:\n%s", df["judge_verdict"].value_counts().to_string())

    train_val_df, test_df = _split_by_edge_identity(
        df.copy(),
        args.test_fraction,
        args.seed,
        stratify_cols=stratify_cols,
        require_stratification=getattr(args, "require_stratification", False),
    )
    adjusted_val_fraction = args.val_fraction / (1.0 - args.test_fraction)
    train_df, val_df = _split_by_edge_identity(
        train_val_df.copy(),
        adjusted_val_fraction,
        args.seed,
        stratify_cols=stratify_cols,
        require_stratification=getattr(args, "require_stratification", False),
    )
    if balance_cols:
        train_df = _balance_rows_exact(
            train_df,
            balance_cols,
            args.seed,
            samples_per_group=getattr(args, "balance_samples_per_group", None),
        )
        val_df = _balance_rows_exact(
            val_df,
            balance_cols,
            args.seed,
            samples_per_group=getattr(args, "balance_samples_per_group", None),
        )
        test_df = _balance_rows_exact(
            test_df,
            balance_cols,
            args.seed,
            samples_per_group=getattr(args, "balance_samples_per_group", None),
        )

    train_path = output_dir / "judge_train_gtlit.xlsx"
    val_path = output_dir / "judge_val_gtlit.xlsx"
    test_path = output_dir / "judge_test_gtlit.xlsx"
    _save_split(
        train_df,
        train_path,
        _build_split_metadata(
            train_df,
            split_name="gt_lit_train",
            input_path=input_path,
            input_metadata=input_metadata,
            stratify_cols=stratify_cols,
            seed=args.seed,
        ),
    )
    _save_split(
        val_df,
        val_path,
        _build_split_metadata(
            val_df,
            split_name="gt_lit_val",
            input_path=input_path,
            input_metadata=input_metadata,
            stratify_cols=stratify_cols,
            seed=args.seed,
        ),
    )
    _save_split(
        test_df,
        test_path,
        _build_split_metadata(
            test_df,
            split_name="gt_lit_test",
            input_path=input_path,
            input_metadata=input_metadata,
            stratify_cols=stratify_cols,
            seed=args.seed,
        ),
    )

    logger.info("Train rows: %s", f"{len(train_df):,}")
    logger.info("Val rows: %s", f"{len(val_df):,}")
    logger.info("Test rows: %s", f"{len(test_df):,}")
    logger.info("Saved: %s", train_path)
    logger.info("Saved: %s", val_path)
    logger.info("Saved: %s", test_path)


if __name__ == "__main__":
    main()
