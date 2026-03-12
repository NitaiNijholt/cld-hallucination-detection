#!/usr/bin/env python3
"""
Prepare CLD judge training and evaluation datasets.

Training data  : GT Synth (RQ1a_gt_synth_correctness) — clean, unambiguous corruption labels
Evaluation data: GT Lit   (RQ1a_gt_lit_correctness)   — held-out, never seen during training

The GT Synth→GT Lit transfer gap is the research question: does finetuning on synthetic
corruptions improve detection on real expert ground truth?

Outputs:
  judge_train.xlsx        — 90% of GT Synth edges (by unique edge identity)
  judge_val_synth.xlsx    — 10% of GT Synth edges (in-distribution validation)
  judge_eval_gtlit.xlsx   — all GT Lit edges (out-of-distribution evaluation)

Each output has columns: prompt, completion, source, target, domain,
                          classification, judge_verdict, is_corrupted

Usage:
    python -m finetuning.src.data.prepare_judge_data
    python -m finetuning.src.data.prepare_judge_data --data-root /path/to/final_runs --output-dir /tmp/out
"""

import argparse
import json
import logging
import os
import re
import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a causal diagram expert evaluating the quality of a causal explanation."
)

USER_TEMPLATE = """\
SOURCE: {source}
TARGET: {target}
RELATIONSHIP: {relationship}
EXPLANATION: {motivation}

Assess whether the causal reasoning is logically sound based on:
- Plausibility: is there a clear causal mechanism?
- Temporality: does the cause precede the effect?
- Strength: is the relationship substantial?
- Coherence: is the reasoning internally consistent?

Respond with REASON then VERDICT."""

COMPLETION_TEMPLATE = "REASON: {reason}\nVERDICT: {verdict}"


def _get_default_data_root() -> Path:
    """Resolve default data root (final_runs at repo root)."""
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    if env_root:
        return Path(env_root) / "final_runs"
    # __file__ is at finetuning/src/data/prepare_judge_data.py
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "final_runs"


def _get_default_output_dir() -> Path:
    """Resolve default output dir (finetuning/src at repo root)."""
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    if env_root:
        return Path(env_root) / "finetuning" / "src"
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "finetuning" / "src"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CLD judge training/eval data")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to final_runs (default: repo_root/final_runs)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for xlsx files (default: finetuning/src)",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of GT Synth edges for validation",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    p.add_argument(
        "--min-file-bytes",
        type=int,
        default=100_000,
        help="Skip xlsx files smaller than this",
    )
    return p.parse_args()


def _extract_domain(filepath: str) -> str:
    """Infer CLD domain name from file path component."""
    parts = filepath.replace("\\", "/").split("/")
    for part in parts:
        if part in ("depressive", "emergency_department", "social_norms"):
            return part
    return "unknown"


def _extract_reason(judge_message_str) -> str:
    """Pull reason text out of the Judge Message JSON blob."""
    if pd.isna(judge_message_str):
        return ""
    try:
        msg = json.loads(judge_message_str)
        results = msg.get("judge_results", [])
        if results:
            return results[0].get("reason", "").strip()
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r'"reason"\s*:\s*"(.*?)"(?:,\s*"|\s*})', judge_message_str, re.DOTALL)
    return m.group(1).strip() if m else ""


def _load_xlsx_files(glob_pattern: str, min_file_bytes: int) -> pd.DataFrame:
    """Load all per-edge xlsx files matching glob, add domain column."""
    files = [
        f for f in glob.glob(glob_pattern, recursive=True)
        if os.path.getsize(f) > min_file_bytes
    ]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {glob_pattern}")
    base = os.path.dirname(glob_pattern.split("**")[0])
    logger.info("Loading %d files from %s", len(files), base)

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
            df["_file"] = f
            df["domain"] = _extract_domain(f)
            dfs.append(df)
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    return pd.concat(dfs, ignore_index=True)


def _build_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw xlsx rows to prompt/completion pairs, drop unusable rows."""
    required = ["Source", "Target", "Relationship Type", "Motivation",
                "Judge Verdict", "Judge Message"]
    df = df.dropna(subset=required).copy()

    df["_reason"] = df["Judge Message"].apply(_extract_reason)
    df = df[df["_reason"] != ""]
    df = df[~df["Judge Verdict"].str.upper().eq("ERROR")]

    rows = []
    for _, row in df.iterrows():
        user_msg = USER_TEMPLATE.format(
            source=str(row["Source"]).strip(),
            target=str(row["Target"]).strip(),
            relationship=str(row["Relationship Type"]).strip(),
            motivation=str(row["Motivation"]).strip(),
        )
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST]"
        completion = COMPLETION_TEMPLATE.format(
            reason=row["_reason"],
            verdict=str(row["Judge Verdict"]).strip().upper(),
        )
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "source": str(row["Source"]).strip(),
            "target": str(row["Target"]).strip(),
            "domain": row["domain"],
            "judge_verdict": str(row["Judge Verdict"]).strip().upper(),
            "classification": row.get("Classification", None),
            "is_corrupted": row.get("Is Corrupted", None),
        })
    return pd.DataFrame(rows)


def _split_by_edge_identity(df: pd.DataFrame, val_fraction: float, seed: int):
    """Split rows by unique (source, target, domain) identity."""
    edge_ids = df[["source", "target", "domain"]].drop_duplicates()
    train_ids, val_ids = train_test_split(
        edge_ids,
        test_size=val_fraction,
        random_state=seed,
    )

    train_key = set(zip(train_ids["source"], train_ids["target"], train_ids["domain"]))
    val_key = set(zip(val_ids["source"], val_ids["target"], val_ids["domain"]))

    df["_key"] = list(zip(df["source"], df["target"], df["domain"]))
    train_df = df[df["_key"].isin(train_key)].drop(columns="_key")
    val_df = df[df["_key"].isin(val_key)].drop(columns="_key")
    return train_df, val_df


def main(args=None) -> None:
    if args is None:
        args = parse_args()
    data_root = Path(args.data_root) if args.data_root else _get_default_data_root()
    output_dir = Path(args.output_dir) if args.output_dir else _get_default_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)

    synth_glob = str(data_root / "RQ1a_gt_synth_correctness" / "**" / "*.xlsx")
    lit_glob = str(data_root / "RQ1a_gt_lit_correctness" / "**" / "*.xlsx")

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", output_dir)
    logger.info("Val fraction: %.2f, seed: %d", args.val_fraction, args.seed)

    # ── GT Synth ──────────────────────────────────────────────────────────────
    logger.info("[1/2] Loading GT Synth (training data)...")
    synth_raw = _load_xlsx_files(synth_glob, args.min_file_bytes)
    logger.info("Raw rows: %s", f"{len(synth_raw):,}")

    synth_df = _build_rows(synth_raw)
    logger.info("After cleaning: %s rows", f"{len(synth_df):,}")
    logger.info("Unique edges: %s", f"{synth_df[['source','target','domain']].drop_duplicates().shape[0]:,}")
    logger.info("Verdict dist:\n%s", synth_df["judge_verdict"].value_counts().to_string())

    train_df, val_synth_df = _split_by_edge_identity(
        synth_df, args.val_fraction, args.seed
    )
    logger.info("Train rows: %s | Val rows: %s", f"{len(train_df):,}", f"{len(val_synth_df):,}")

    train_path = output_dir / "judge_train.xlsx"
    val_synth_path = output_dir / "judge_val_synth.xlsx"
    train_df.to_excel(train_path, index=False)
    val_synth_df.to_excel(val_synth_path, index=False)
    logger.info("Saved: %s", train_path)
    logger.info("Saved: %s", val_synth_path)

    # ── GT Lit ─────────────────────────────────────────────────────────────────
    logger.info("[2/2] Loading GT Lit (evaluation only)...")
    lit_raw = _load_xlsx_files(lit_glob, args.min_file_bytes)
    logger.info("Raw rows: %s", f"{len(lit_raw):,}")

    lit_df = _build_rows(lit_raw)
    logger.info("After cleaning: %s rows", f"{len(lit_df):,}")
    logger.info("Unique edges: %s", f"{lit_df[['source','target','domain']].drop_duplicates().shape[0]:,}")
    logger.info("Verdict dist:\n%s", lit_df["judge_verdict"].value_counts().to_string())
    if "classification" in lit_df.columns:
        logger.info("Classification dist:\n%s", lit_df["classification"].value_counts().to_string())

    lit_path = output_dir / "judge_eval_gtlit.xlsx"
    lit_df.to_excel(lit_path, index=False)
    logger.info("Saved: %s", lit_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("Summary:")
    logger.info("  judge_train.xlsx: %s rows (GT Synth train)", f"{len(train_df):>6,}")
    logger.info("  judge_val_synth.xlsx: %s rows (GT Synth val)", f"{len(val_synth_df):>6,}")
    logger.info("  judge_eval_gtlit.xlsx: %s rows (GT Lit eval)", f"{len(lit_df):>6,}")
    logger.info("Transfer gap experiment: Train on GT Synth → evaluate on GT Lit")


if __name__ == "__main__":
    main()
