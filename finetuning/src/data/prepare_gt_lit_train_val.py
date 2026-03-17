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

from .prepare_judge_data import _split_by_edge_identity

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
    return p.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Input path: %s", input_path)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "Val fraction: %.2f, test fraction: %.2f, seed: %d",
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    df = pd.read_excel(input_path).dropna(subset=["prompt", "completion"])
    logger.info("Loaded GT Lit rows: %s", f"{len(df):,}")
    logger.info(
        "Unique edges: %s",
        f"{df[['source', 'target', 'domain']].drop_duplicates().shape[0]:,}",
    )
    logger.info("Verdict dist:\n%s", df["judge_verdict"].value_counts().to_string())

    train_val_df, test_df = _split_by_edge_identity(
        df.copy(), args.test_fraction, args.seed, stratify_col="domain"
    )
    adjusted_val_fraction = args.val_fraction / (1.0 - args.test_fraction)
    train_df, val_df = _split_by_edge_identity(
        train_val_df.copy(), adjusted_val_fraction, args.seed, stratify_col="domain"
    )

    train_path = output_dir / "judge_train_gtlit.xlsx"
    val_path = output_dir / "judge_val_gtlit.xlsx"
    test_path = output_dir / "judge_test_gtlit.xlsx"
    train_df.to_excel(train_path, index=False)
    val_df.to_excel(val_path, index=False)
    test_df.to_excel(test_path, index=False)

    logger.info("Train rows: %s", f"{len(train_df):,}")
    logger.info("Val rows: %s", f"{len(val_df):,}")
    logger.info("Test rows: %s", f"{len(test_df):,}")
    logger.info("Saved: %s", train_path)
    logger.info("Saved: %s", val_path)
    logger.info("Saved: %s", test_path)


if __name__ == "__main__":
    main()
