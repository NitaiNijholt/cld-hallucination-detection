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

from ..metadata_utils import (
    atomic_to_excel,
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

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a causal diagram expert evaluating the quality of a causal explanation."
)

USER_TEMPLATE_BODY = """\
SOURCE: {source}
TARGET: {target}
RELATIONSHIP: {relationship}
EXPLANATION: {motivation}

Assess whether the causal reasoning is logically sound based on:
- Plausibility: is there a clear causal mechanism?
- Temporality: does the cause precede the effect?
- Strength: is the relationship substantial?
- Coherence: is the reasoning internally consistent?
"""

OBJECTIVE_SPECS = {
    "reason_verdict": {
        "response_instruction": "Respond with REASON then VERDICT.",
        "completion_template": "REASON: {reason}\nVERDICT: {verdict}",
    },
    "verdict_only": {
        "response_instruction": "Respond with VERDICT only.",
        "completion_template": "VERDICT: {verdict}",
    },
}

PROMPT_VARIANTS = ("baseline", "cot", "mechanistic")


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
        default=0,
        help="Skip xlsx files smaller than this",
    )
    p.add_argument(
        "--objective-mode",
        choices=sorted(OBJECTIVE_SPECS),
        default="reason_verdict",
        help="Target format to build for supervised training/evaluation",
    )
    p.add_argument(
        "--stratify-cols",
        default="domain,judge_verdict",
        help="Comma-separated edge-level columns to stratify train/val splits on",
    )
    p.add_argument(
        "--require-stratification",
        action="store_true",
        help="Fail instead of falling back to unstratified splits when requested strata cannot be preserved",
    )
    p.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=None,
        help="Only keep judged rows generated from a single prompt variant",
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
    p.add_argument(
        "--balance-lit-eval-cols",
        default="",
        help="Comma-separated row-level columns to exact-balance only the GT Lit eval set",
    )
    p.add_argument(
        "--balance-lit-eval-samples-per-group",
        type=int,
        default=None,
        help="Optional exact row count per GT Lit eval balance group; defaults to the smallest group",
    )
    p.add_argument(
        "--preserve-split-group-cols",
        default="",
        help="Comma-separated columns whose group coverage should be preserved in both GT Synth train/val without exact balancing",
    )
    p.add_argument(
        "--exclude-lit-overlap-from-synth",
        action="store_true",
        help="Remove GT Synth edges that also appear in GT Lit before splitting GT Synth train/val",
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


def _sanitize_reason(reason: str) -> str:
    text = str(reason).strip()
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    text = re.sub(r"^\s*REASON\s*:\s*", "", text, flags=re.IGNORECASE)
    trailing_metadata = re.search(r"(?:^|\n)\s*(?:VERDICT|SCORE)\s*:", text, flags=re.IGNORECASE)
    if trailing_metadata is not None:
        text = text[: trailing_metadata.start()]
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(VERDICT|SCORE)\s*:", line, flags=re.IGNORECASE):
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _parse_stratify_cols(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        cols = value
    else:
        cols = [part.strip() for part in str(value).split(",")]
    return [col for col in cols if col]


def _infer_prompt_variant(filepath: str) -> str | None:
    normalized = filepath.replace("\\", "/").lower()
    for variant in PROMPT_VARIANTS:
        if re.search(rf"(^|[^a-z]){re.escape(variant)}([^a-z]|$)", normalized):
            return variant
    return None


def _get_user_template(objective_mode: str) -> str:
    spec = OBJECTIVE_SPECS[objective_mode]
    return USER_TEMPLATE_BODY + "\n" + spec["response_instruction"]


def _build_prompt(row: pd.Series, objective_mode: str) -> str:
    user_msg = _get_user_template(objective_mode).format(
        source=str(row["Source"]).strip(),
        target=str(row["Target"]).strip(),
        relationship=str(row["Relationship Type"]).strip(),
        motivation=str(row["Motivation"]).strip(),
    )
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST]"


def _build_completion(reason: str, verdict: str, objective_mode: str) -> str:
    spec = OBJECTIVE_SPECS[objective_mode]
    if objective_mode == "reason_verdict":
        return spec["completion_template"].format(reason=reason, verdict=verdict)
    return spec["completion_template"].format(verdict=verdict)


def _collapse_edge_value(values: pd.Series) -> str:
    unique = sorted({str(v).strip() for v in values.dropna() if str(v).strip()})
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "__MULTI__"


def _build_edge_stratify_labels(
    df: pd.DataFrame,
    edge_ids: pd.DataFrame,
    stratify_cols: list[str],
) -> pd.Series | None:
    usable_cols = [col for col in stratify_cols if col in df.columns]
    if not usable_cols:
        return None
    edge_with_meta = edge_ids.copy()
    aggregated_cols = [col for col in usable_cols if col not in edge_with_meta.columns]
    if aggregated_cols:
        selection_cols = list(dict.fromkeys(["source", "target", "domain", *aggregated_cols]))
        edge_meta = (
            df[selection_cols]
            .groupby(["source", "target", "domain"], dropna=False)
            .agg({col: _collapse_edge_value for col in aggregated_cols})
            .reset_index()
        )
        edge_with_meta = edge_with_meta.merge(
            edge_meta, on=["source", "target", "domain"], how="left"
        )
    return edge_with_meta[usable_cols].fillna("unknown").astype(str).agg("||".join, axis=1)


def _build_dataset_metadata(
    df: pd.DataFrame,
    *,
    split_name: str,
    objective_mode: str,
    source_description: str,
    stratify_cols: list[str],
    seed: int,
) -> dict:
    unique_edges = df[["source", "target", "domain"]].drop_duplicates().shape[0] if len(df) else 0
    return {
        "split_name": split_name,
        "objective_mode": objective_mode,
        "source_description": source_description,
        "seed": seed,
        "stratify_cols": stratify_cols,
        "row_count": int(len(df)),
        "unique_edge_count": int(unique_edges),
        "domain_counts": summarize_counts(df["domain"].tolist()) if "domain" in df.columns else {},
        "judge_verdict_counts": summarize_counts(df["judge_verdict"].tolist()) if "judge_verdict" in df.columns else {},
        "prompt_variant_counts": summarize_counts(df["prompt_variant"].tolist()) if "prompt_variant" in df.columns else {},
        "prompt_spec": {
            "system_prompt": SYSTEM_PROMPT,
            "user_template": _get_user_template(objective_mode),
            "completion_template": OBJECTIVE_SPECS[objective_mode]["completion_template"],
        },
    }


def _save_dataset_with_metadata(df: pd.DataFrame, path: Path, metadata: dict) -> None:
    atomic_to_excel(df, path)
    payload = {
        **metadata,
        "file_name": path.name,
        "file_sha256": sha256_file(path),
    }
    write_json(sidecar_metadata_path(path), payload)


def _should_include_xlsx_file(filepath: str, min_file_bytes: int) -> bool:
    path = Path(filepath)
    name = path.name
    if path.suffix.lower() != ".xlsx":
        return False
    if name.endswith(".backup.xlsx"):
        return False
    if not name.startswith("judged_"):
        return False
    return path.stat().st_size >= min_file_bytes


def _load_xlsx_files(glob_pattern: str, min_file_bytes: int) -> pd.DataFrame:
    """Load all per-edge xlsx files matching glob, add domain column."""
    candidate_files = [
        f for f in glob.glob(glob_pattern, recursive=True)
        if Path(f).suffix.lower() == ".xlsx"
        and not Path(f).name.endswith(".backup.xlsx")
        and Path(f).stat().st_size >= min_file_bytes
    ]
    judged_files = [f for f in candidate_files if Path(f).name.startswith("judged_")]
    files = judged_files or candidate_files
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {glob_pattern}")
    base = os.path.dirname(glob_pattern.split("**")[0])
    logger.info("Loading %d files from %s", len(files), base)

    dfs = []
    for f in files:
        try:
            df = read_excel_with_retry(f)
            df["_file"] = f
            df["domain"] = _extract_domain(f)
            df["prompt_variant"] = _infer_prompt_variant(f)
            dfs.append(df)
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    return pd.concat(dfs, ignore_index=True)


def _build_rows(df: pd.DataFrame, objective_mode: str = "reason_verdict") -> pd.DataFrame:
    """Convert raw xlsx rows to prompt/completion pairs, drop unusable rows."""
    required = ["Source", "Target", "Relationship Type", "Motivation", "Judge Verdict"]
    if objective_mode == "reason_verdict":
        required.append("Judge Message")
    df = df.dropna(subset=required).copy()
    df = df[~df["Judge Verdict"].str.upper().eq("ERROR")]

    if objective_mode == "reason_verdict":
        df["_reason"] = df["Judge Message"].apply(_extract_reason).apply(_sanitize_reason)
        df = df[df["_reason"] != ""]
    else:
        df["_reason"] = ""

    rows = []
    for _, row in df.iterrows():
        prompt = _build_prompt(row, objective_mode)
        verdict = str(row["Judge Verdict"]).strip().upper()
        completion = _build_completion(row["_reason"], verdict, objective_mode)
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "source": str(row["Source"]).strip(),
            "target": str(row["Target"]).strip(),
            "domain": row["domain"],
            "judge_verdict": verdict,
            "classification": row.get("Classification", None),
            "is_corrupted": row.get("Is Corrupted", None),
            "objective_mode": objective_mode,
            "prompt_variant": row.get("prompt_variant", _infer_prompt_variant(str(row.get("_file", "")))),
        })
    return pd.DataFrame(rows)


def _balance_group_counts(df: pd.DataFrame, balance_cols: list[str]) -> pd.Series:
    usable_cols = [col for col in balance_cols if col in df.columns]
    if not usable_cols:
        raise ValueError(f"No usable balance columns found in dataframe for {balance_cols}")
    return df.groupby(usable_cols, dropna=False).size().sort_index()


def _balance_rows_exact(
    df: pd.DataFrame,
    balance_cols: list[str],
    seed: int,
    samples_per_group: int | None = None,
) -> pd.DataFrame:
    if not balance_cols:
        return df
    counts = _balance_group_counts(df, balance_cols)
    if counts.empty:
        return df
    target_n = samples_per_group if samples_per_group is not None else int(counts.min())
    if target_n <= 0:
        raise ValueError(f"Exact balance target must be positive; got {target_n}")
    if int(counts.min()) < target_n:
        raise ValueError(
            f"Cannot exact-balance on {balance_cols} with {target_n} rows per group; "
            f"smallest group only has {int(counts.min())}"
        )
    parts = [
        part.sample(n=target_n, random_state=seed)
        for _, part in df.groupby(balance_cols, dropna=False)
    ]
    return pd.concat(parts, ignore_index=True)


def _exclude_overlapping_edges(reference_df: pd.DataFrame, exclusion_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    exclusion_edges = set(
        map(tuple, exclusion_df[["source", "target", "domain"]].drop_duplicates().itertuples(index=False, name=None))
    )
    if not exclusion_edges:
        return reference_df, 0
    keys = list(map(tuple, reference_df[["source", "target", "domain"]].itertuples(index=False, name=None)))
    keep_mask = [key not in exclusion_edges for key in keys]
    filtered = reference_df.loc[keep_mask].reset_index(drop=True)
    removed_edges = (
        reference_df.loc[[not keep for keep in keep_mask], ["source", "target", "domain"]]
        .drop_duplicates()
        .shape[0]
    )
    return filtered, int(removed_edges)


def _group_value_set(df: pd.DataFrame, group_cols: list[str]) -> set[tuple[str, ...]]:
    usable_cols = [col for col in group_cols if col in df.columns]
    if not usable_cols or df.empty:
        return set()
    return set(map(tuple, df[usable_cols].drop_duplicates().astype(str).itertuples(index=False, name=None)))


def _split_preserving_group_coverage(
    df: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
    stratify_cols: list[str],
    group_cols: list[str],
    require_stratification: bool,
    max_seed_tries: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_groups = _group_value_set(df, group_cols)
    if not required_groups:
        return _split_by_edge_identity(
            df,
            val_fraction,
            seed,
            stratify_cols=stratify_cols,
            require_stratification=require_stratification,
        )

    best_split: tuple[pd.DataFrame, pd.DataFrame] | None = None
    best_score = -1
    best_full_split: tuple[pd.DataFrame, pd.DataFrame] | None = None
    best_full_seed = seed
    best_full_val_min = -1
    for candidate_seed in range(seed, seed + max_seed_tries):
        train_df, val_df = _split_by_edge_identity(
            df,
            val_fraction,
            candidate_seed,
            stratify_cols=stratify_cols,
            require_stratification=require_stratification,
        )
        train_groups = _group_value_set(train_df, group_cols)
        val_groups = _group_value_set(val_df, group_cols)
        if train_groups == required_groups and val_groups == required_groups:
            val_counts = val_df.groupby(group_cols).size()
            val_min = int(val_counts.min()) if not val_counts.empty else 0
            if val_min > best_full_val_min:
                best_full_val_min = val_min
                best_full_seed = candidate_seed
                best_full_split = (train_df, val_df)
        coverage_score = len(train_groups & required_groups) + len(val_groups & required_groups)
        if coverage_score > best_score:
            best_score = coverage_score
            best_split = (train_df, val_df)

    if best_full_split is not None:
        if best_full_seed != seed:
            logger.info(
                "Adjusted split seed from %d to %d to preserve group coverage on %s (min val group count=%d)",
                seed,
                best_full_seed,
                group_cols,
                best_full_val_min,
            )
        return best_full_split

    assert best_split is not None
    logger.warning(
        "Could not preserve full group coverage on %s after %d seed attempts; using best available split",
        group_cols,
        max_seed_tries,
    )
    return best_split


def _split_by_edge_identity(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
    stratify_col: str | None = None,
    stratify_cols: list[str] | None = None,
    require_stratification: bool = False,
):
    """Split rows by unique (source, target, domain) identity."""
    edge_ids = df[["source", "target", "domain"]].drop_duplicates().copy()
    if len(edge_ids) < 2 or val_fraction <= 0:
        logger.warning("Skipping split; insufficient unique edges for validation split")
        return df.copy(), df.iloc[0:0].copy()
    stratify = None
    effective_cols = stratify_cols or ([stratify_col] if stratify_col else [])
    if effective_cols:
        labels = _build_edge_stratify_labels(df, edge_ids, effective_cols)
        counts = labels.value_counts() if labels is not None else pd.Series(dtype=int)
        n_classes = len(counts)
        n_val = max(1, int(round(len(edge_ids) * val_fraction)))
        n_train = len(edge_ids) - n_val
        if len(counts) > 1 and counts.min() >= 2 and n_val >= n_classes and n_train >= n_classes:
            stratify = labels
        else:
            if require_stratification:
                raise ValueError(
                    f"Unable to preserve requested stratification on {effective_cols}; "
                    "dataset does not have enough edge counts per class"
                )
            logger.warning(
                "Skipping stratified split on %s; insufficient edge counts per class",
                effective_cols,
            )
    train_ids, val_ids = train_test_split(
        edge_ids,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
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
    objective_mode = getattr(args, "objective_mode", "reason_verdict")
    stratify_cols = _parse_stratify_cols(
        getattr(args, "stratify_cols", "domain,judge_verdict")
    )
    balance_cols = _parse_stratify_cols(getattr(args, "balance_cols", ""))
    balance_lit_eval_cols = _parse_stratify_cols(getattr(args, "balance_lit_eval_cols", ""))
    preserve_split_group_cols = _parse_stratify_cols(getattr(args, "preserve_split_group_cols", ""))
    prompt_variant = getattr(args, "prompt_variant", None)

    synth_glob = str(data_root / "RQ1a_gt_synth_correctness" / "**" / "*.xlsx")
    lit_glob = str(data_root / "RQ1a_gt_lit_correctness" / "**" / "*.xlsx")

    logger.info("Data root: %s", data_root)
    logger.info("Output dir: %s", output_dir)
    logger.info("Val fraction: %.2f, seed: %d", args.val_fraction, args.seed)
    if prompt_variant:
        logger.info("Prompt variant filter: %s", prompt_variant)
    if balance_cols:
        logger.info(
            "Exact row balancing on %s (samples_per_group=%s)",
            balance_cols,
            getattr(args, "balance_samples_per_group", None),
        )
    if balance_lit_eval_cols:
        logger.info(
            "Exact GT Lit eval balancing on %s (samples_per_group=%s)",
            balance_lit_eval_cols,
            getattr(args, "balance_lit_eval_samples_per_group", None),
        )
    if preserve_split_group_cols:
        logger.info("Preserving GT Synth train/val group coverage on %s", preserve_split_group_cols)

    # ── GT Lit ─────────────────────────────────────────────────────────────────
    logger.info("[1/2] Loading GT Lit (evaluation only)...")
    lit_raw = _load_xlsx_files(lit_glob, args.min_file_bytes)
    if prompt_variant:
        lit_raw = lit_raw[lit_raw["prompt_variant"].eq(prompt_variant)].copy()
        if lit_raw.empty:
            raise ValueError(f"No GT Lit rows found for prompt_variant={prompt_variant}")
    logger.info("Raw rows: %s", f"{len(lit_raw):,}")

    lit_df = _build_rows(lit_raw, objective_mode=objective_mode)
    lit_overlap_df = lit_df
    if balance_lit_eval_cols:
        lit_overlap_df = _balance_rows_exact(
            lit_df,
            balance_lit_eval_cols,
            args.seed,
            samples_per_group=getattr(args, "balance_lit_eval_samples_per_group", None),
        )
    if balance_cols:
        lit_df = _balance_rows_exact(
            lit_df,
            balance_cols,
            args.seed,
            samples_per_group=getattr(args, "balance_samples_per_group", None),
        )
    logger.info("After cleaning: %s rows", f"{len(lit_df):,}")
    logger.info("Unique edges: %s", f"{lit_df[['source','target','domain']].drop_duplicates().shape[0]:,}")
    logger.info("Verdict dist:\n%s", lit_df["judge_verdict"].value_counts().to_string())
    if "classification" in lit_df.columns:
        logger.info("Classification dist:\n%s", lit_df["classification"].value_counts().to_string())

    # ── GT Synth ──────────────────────────────────────────────────────────────
    logger.info("[2/2] Loading GT Synth (training data)...")
    synth_raw = _load_xlsx_files(synth_glob, args.min_file_bytes)
    if prompt_variant:
        synth_raw = synth_raw[synth_raw["prompt_variant"].eq(prompt_variant)].copy()
        if synth_raw.empty:
            raise ValueError(f"No GT Synth rows found for prompt_variant={prompt_variant}")
    logger.info("Raw rows: %s", f"{len(synth_raw):,}")

    synth_df = _build_rows(synth_raw, objective_mode=objective_mode)
    if getattr(args, "exclude_lit_overlap_from_synth", False):
        synth_df, removed_overlap_edges = _exclude_overlapping_edges(synth_df, lit_overlap_df)
        logger.info(
            "Removed %s GT Synth edges overlapping with GT Lit",
            f"{removed_overlap_edges:,}",
        )
    logger.info("After cleaning: %s rows", f"{len(synth_df):,}")
    logger.info("Unique edges: %s", f"{synth_df[['source','target','domain']].drop_duplicates().shape[0]:,}")
    logger.info("Verdict dist:\n%s", synth_df["judge_verdict"].value_counts().to_string())

    if balance_cols:
        train_df, val_synth_df = _split_preserving_group_coverage(
            synth_df,
            val_fraction=args.val_fraction,
            seed=args.seed,
            stratify_cols=stratify_cols,
            group_cols=balance_cols,
            require_stratification=getattr(args, "require_stratification", False),
        )
    elif preserve_split_group_cols:
        train_df, val_synth_df = _split_preserving_group_coverage(
            synth_df,
            val_fraction=args.val_fraction,
            seed=args.seed,
            stratify_cols=stratify_cols,
            group_cols=preserve_split_group_cols,
            require_stratification=getattr(args, "require_stratification", False),
        )
    else:
        train_df, val_synth_df = _split_by_edge_identity(
            synth_df,
            args.val_fraction,
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
        val_synth_df = _balance_rows_exact(
            val_synth_df,
            balance_cols,
            args.seed,
            samples_per_group=getattr(args, "balance_samples_per_group", None),
        )
    logger.info("Train rows: %s | Val rows: %s", f"{len(train_df):,}", f"{len(val_synth_df):,}")

    train_path = output_dir / "judge_train.xlsx"
    val_synth_path = output_dir / "judge_val_synth.xlsx"
    _save_dataset_with_metadata(
        train_df,
        train_path,
        _build_dataset_metadata(
            train_df,
            split_name="gt_synth_train",
            objective_mode=objective_mode,
            source_description=synth_glob,
            stratify_cols=stratify_cols,
            seed=args.seed,
        ),
    )
    _save_dataset_with_metadata(
        val_synth_df,
        val_synth_path,
        _build_dataset_metadata(
            val_synth_df,
            split_name="gt_synth_val",
            objective_mode=objective_mode,
            source_description=synth_glob,
            stratify_cols=stratify_cols,
            seed=args.seed,
        ),
    )
    logger.info("Saved: %s", train_path)
    logger.info("Saved: %s", val_synth_path)

    lit_path = output_dir / "judge_eval_gtlit.xlsx"
    _save_dataset_with_metadata(
        lit_df,
        lit_path,
        _build_dataset_metadata(
            lit_df,
            split_name="gt_lit_eval",
            objective_mode=objective_mode,
            source_description=lit_glob,
            stratify_cols=stratify_cols,
            seed=args.seed,
        ),
    )
    logger.info("Saved: %s", lit_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("Summary:")
    logger.info("  judge_train.xlsx: %s rows (GT Synth train)", f"{len(train_df):>6,}")
    logger.info("  judge_val_synth.xlsx: %s rows (GT Synth val)", f"{len(val_synth_df):>6,}")
    logger.info("  judge_eval_gtlit.xlsx: %s rows (GT Lit eval)", f"{len(lit_df):>6,}")
    logger.info("Transfer gap experiment: Train on GT Synth → evaluate on GT Lit")


if __name__ == "__main__":
    main()
