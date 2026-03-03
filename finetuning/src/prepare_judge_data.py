#!/usr/bin/env python3
"""
Prepare CLD judge training and evaluation datasets.

Training data  : GT Synth (RQ1a_gt_synth_correctness) — clean, unambiguous corruption labels
Evaluation data: GT Lit   (RQ1a_gt_lit_correctness)   — held-out, never seen during training

The GT Synth→GT Lit transfer gap is the research question: does finetuning on synthetic
corruptions improve detection on real expert ground truth?

Outputs:
  src/judge_train.xlsx        — 90% of GT Synth edges (by unique edge identity)
  src/judge_val_synth.xlsx    — 10% of GT Synth edges (in-distribution validation)
  src/judge_eval_gtlit.xlsx   — all GT Lit edges (out-of-distribution evaluation)

Each output has columns: prompt, completion, source, target, domain,
                          classification, judge_verdict, is_corrupted
"""

import json
import os
import re
import glob

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
# __file__ is at <repo>/finetuning/src/prepare_judge_data.py
# Going up 3 levels lands at the repo root (<repo>/)
# final_runs/ lives directly in the repo root
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))   # .../finetuning/src
_FINETUNING_DIR = os.path.dirname(_SRC_DIR)                     # .../finetuning
REPO_ROOT       = os.path.dirname(_FINETUNING_DIR)              # .../cld-hallucination-detection
CLD_DATA_ROOT   = os.path.join(REPO_ROOT, "final_runs")

SYNTH_GLOB = os.path.join(CLD_DATA_ROOT, "RQ1a_gt_synth_correctness", "**", "*.xlsx")
LIT_GLOB   = os.path.join(CLD_DATA_ROOT, "RQ1a_gt_lit_correctness",   "**", "*.xlsx")

MIN_FILE_BYTES = 100_000   # skip aggregate stats files (< 100 KB)
RANDOM_SEED    = 42
VAL_FRACTION   = 0.10      # 10% of GT Synth unique edges held out as in-dist val

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
    # Fallback: try regex
    m = re.search(r'"reason"\s*:\s*"(.*?)"(?:,\s*"|\s*})', judge_message_str, re.DOTALL)
    return m.group(1).strip() if m else ""


def _load_xlsx_files(glob_pattern: str) -> pd.DataFrame:
    """Load all per-edge xlsx files matching glob, add domain column."""
    files = [
        f for f in glob.glob(glob_pattern, recursive=True)
        if os.path.getsize(f) > MIN_FILE_BYTES
    ]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {glob_pattern}")
    print(f"  Loading {len(files)} files from {os.path.dirname(glob_pattern.split('**')[0])}...")

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
            df["_file"] = f
            df["domain"] = _extract_domain(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True)


def _build_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw xlsx rows to prompt/completion pairs, drop unusable rows."""
    required = ["Source", "Target", "Relationship Type", "Motivation",
                "Judge Verdict", "Judge Message"]
    df = df.dropna(subset=required).copy()

    # Extract chain-of-thought reason from JSON blob
    df["_reason"] = df["Judge Message"].apply(_extract_reason)

    # Drop rows where reason extraction failed or verdict is ERROR
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
        prompt     = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST]"
        completion = COMPLETION_TEMPLATE.format(
            reason=row["_reason"],
            verdict=str(row["Judge Verdict"]).strip().upper(),
        )
        rows.append({
            "prompt":        prompt,
            "completion":    completion,
            "source":        str(row["Source"]).strip(),
            "target":        str(row["Target"]).strip(),
            "domain":        row["domain"],
            "judge_verdict": str(row["Judge Verdict"]).strip().upper(),
            "classification": row.get("Classification", None),
            "is_corrupted":  row.get("Is Corrupted", None),
        })
    return pd.DataFrame(rows)


def _split_by_edge_identity(df: pd.DataFrame, val_fraction: float, seed: int):
    """
    Split rows by unique (source, target, domain) identity — not by row.
    All rows for an edge go to the same split to prevent leakage across
    prompt variants / runs.
    """
    edge_ids = df[["source", "target", "domain"]].drop_duplicates()
    train_ids, val_ids = train_test_split(
        edge_ids,
        test_size=val_fraction,
        random_state=seed,
    )

    train_key = set(zip(train_ids["source"], train_ids["target"], train_ids["domain"]))
    val_key   = set(zip(val_ids["source"],   val_ids["target"],   val_ids["domain"]))

    df["_key"] = list(zip(df["source"], df["target"], df["domain"]))
    train_df = df[df["_key"].isin(train_key)].drop(columns="_key")
    val_df   = df[df["_key"].isin(val_key)].drop(columns="_key")
    return train_df, val_df


def main():
    out_dir = _SRC_DIR  # write outputs next to this script: finetuning/src/
    os.makedirs(out_dir, exist_ok=True)

    # ── GT Synth ──────────────────────────────────────────────────────────────
    print("\n[1/2] Loading GT Synth (training data)...")
    synth_raw = _load_xlsx_files(SYNTH_GLOB)
    print(f"  Raw rows: {len(synth_raw):,}")

    synth_df = _build_rows(synth_raw)
    print(f"  After cleaning: {len(synth_df):,} rows")
    print(f"  Unique edges:   {synth_df[['source','target','domain']].drop_duplicates().shape[0]:,}")
    print(f"  Verdict dist:\n{synth_df['judge_verdict'].value_counts().to_string()}")

    train_df, val_synth_df = _split_by_edge_identity(synth_df, VAL_FRACTION, RANDOM_SEED)
    print(f"  Train rows: {len(train_df):,}  |  Val rows: {len(val_synth_df):,}")

    train_path     = os.path.join(out_dir, "judge_train.xlsx")
    val_synth_path = os.path.join(out_dir, "judge_val_synth.xlsx")
    train_df.to_excel(train_path,     index=False)
    val_synth_df.to_excel(val_synth_path, index=False)
    print(f"  Saved: {train_path}")
    print(f"  Saved: {val_synth_path}")

    # ── GT Lit (held-out eval) ────────────────────────────────────────────────
    print("\n[2/2] Loading GT Lit (evaluation only — never used in training)...")
    lit_raw = _load_xlsx_files(LIT_GLOB)
    print(f"  Raw rows: {len(lit_raw):,}")

    lit_df = _build_rows(lit_raw)
    print(f"  After cleaning: {len(lit_df):,} rows")
    print(f"  Unique edges:   {lit_df[['source','target','domain']].drop_duplicates().shape[0]:,}")
    print(f"  Verdict dist:\n{lit_df['judge_verdict'].value_counts().to_string()}")
    if "classification" in lit_df.columns:
        print(f"  Classification dist:\n{lit_df['classification'].value_counts().to_string()}")

    lit_path = os.path.join(out_dir, "judge_eval_gtlit.xlsx")
    lit_df.to_excel(lit_path, index=False)
    print(f"  Saved: {lit_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  judge_train.xlsx        : {len(train_df):>6,} rows  (GT Synth train)")
    print(f"  judge_val_synth.xlsx    : {len(val_synth_df):>6,} rows  (GT Synth val — in-dist)")
    print(f"  judge_eval_gtlit.xlsx   : {len(lit_df):>6,} rows  (GT Lit eval — out-of-dist)")
    print("\nTransfer gap experiment:")
    print("  Train on GT Synth → evaluate on GT Lit to test generalisation.")


if __name__ == "__main__":
    main()
