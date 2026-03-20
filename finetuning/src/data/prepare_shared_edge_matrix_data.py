#!/usr/bin/env python3
"""Prepare a shared-edge benchmark from existing canonical datasets.

This benchmark reframes GT Synth and GT Lit as two supervision sources over the
same underlying edge universe. It consumes an existing canonical bundle,
constructs one shared train/val/test partition over overlapping edges, and then
materializes source-specific label views for each split.
"""

from __future__ import annotations

import argparse
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OBJECTIVE_MODES = ["reason_verdict", "verdict_only"]
DEFAULT_STRATIFY_COLS = "domain,judge_verdict_synth,judge_verdict_lit"


def _repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT", os.environ.get("REPO_DIR"))
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent.parent.parent.parent


def _default_input_root() -> str | None:
    tag = os.environ.get("CANONICAL_DATA_TAG")
    if not tag:
        return None
    return str(_repo_root() / "finetuning" / "data" / "canonical" / tag)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare shared-edge benchmark from canonical data")
    p.add_argument(
        "--input-root",
        default=_default_input_root(),
        help="Existing canonical data root containing reason_verdict/ and verdict_only/",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Where to write the shared-edge bundle. Defaults to finetuning/data/canonical_shared/<input-name>",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--prompt-variant", default="mechanistic")
    p.add_argument("--stratify-cols", default=DEFAULT_STRATIFY_COLS)
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output root",
    )
    return p.parse_args()


def _parse_cols(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_input_root(raw_input_root: str | None) -> Path:
    if raw_input_root is None:
        raise ValueError("Missing --input-root. Pass a canonical bundle path or set CANONICAL_DATA_TAG.")
    input_root = Path(raw_input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Canonical input root not found: {input_root}")
    return input_root


def _resolve_output_root(raw_output_root: str | None, input_root: Path) -> Path:
    if raw_output_root:
        return Path(raw_output_root)
    return _repo_root() / "finetuning" / "data" / "canonical_shared" / input_root.name


def _assert_output_root(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"Output root already exists and is not empty: {path}. "
            "Use --force only if you intentionally want to regenerate shared-edge artifacts."
        )
    path.mkdir(parents=True, exist_ok=True)


def _load_mode_inputs(mode_root: Path) -> tuple[pd.DataFrame, list[dict | None], pd.DataFrame, dict | None]:
    synth_train_path = mode_root / "judge_train.xlsx"
    synth_val_path = mode_root / "judge_val_synth.xlsx"
    lit_eval_path = mode_root / "judge_eval_gtlit.xlsx"

    for path in [synth_train_path, synth_val_path, lit_eval_path]:
        if not path.exists():
            raise FileNotFoundError(f"Expected canonical input file not found: {path}")

    synth_train = read_excel_with_retry(synth_train_path)
    synth_val = read_excel_with_retry(synth_val_path)
    lit_eval = read_excel_with_retry(lit_eval_path)

    synth_all = pd.concat([synth_train, synth_val], ignore_index=True)
    synth_all = synth_all.drop_duplicates().reset_index(drop=True)
    lit_eval = lit_eval.drop_duplicates().reset_index(drop=True)

    synth_meta = [
        load_json(sidecar_metadata_path(synth_train_path)),
        load_json(sidecar_metadata_path(synth_val_path)),
    ]
    lit_meta = load_json(sidecar_metadata_path(lit_eval_path))
    return synth_all, synth_meta, lit_eval, lit_meta


def _filter_prompt_variant(df: pd.DataFrame, prompt_variant: str | None) -> pd.DataFrame:
    if not prompt_variant or "prompt_variant" not in df.columns:
        return df.reset_index(drop=True)
    return df[df["prompt_variant"].astype(str).eq(prompt_variant)].reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _dominant_value(values: pd.Series) -> str:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return "unknown"
    top_count = counts.max()
    winners = sorted(counts[counts.eq(top_count)].index.tolist())
    return winners[0]


def _build_edge_summary(df: pd.DataFrame, verdict_col_name: str) -> pd.DataFrame:
    required = ["source", "target", "domain", "judge_verdict"]
    _require_columns(df, required, "canonical input")
    grouped = (
        df.groupby(["source", "target", "domain"], dropna=False)["judge_verdict"]
        .agg(_dominant_value)
        .reset_index(name=verdict_col_name)
    )
    return grouped


def _build_shared_edge_ids(synth_df: pd.DataFrame, lit_df: pd.DataFrame) -> pd.DataFrame:
    synth_edges = _build_edge_summary(synth_df, "judge_verdict_synth")
    lit_edges = _build_edge_summary(lit_df, "judge_verdict_lit")
    shared = synth_edges.merge(lit_edges, on=["source", "target", "domain"], how="inner")
    if shared.empty:
        raise ValueError("No shared edges found between GT Synth and GT Lit canonical inputs")
    return shared.drop_duplicates().reset_index(drop=True)


def _split_edge_ids(
    edge_ids: pd.DataFrame,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    stratify_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_fraction <= 0 or val_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Expected val_fraction > 0, test_fraction > 0, and val_fraction + test_fraction < 1")

    usable_cols = [c for c in stratify_cols if c in edge_ids.columns]
    labels = None
    if usable_cols:
        labels = edge_ids[usable_cols].fillna("unknown").astype(str).agg("||".join, axis=1)
        counts = labels.value_counts()
        n_test = max(1, int(round(len(edge_ids) * test_fraction)))
        if len(counts) <= 1 or counts.min() < 2 or n_test < len(counts):
            labels = None

    train_val, test = train_test_split(
        edge_ids,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    adjusted_val_fraction = val_fraction / (1.0 - test_fraction)

    labels_tv = None
    if usable_cols:
        labels_tv = train_val[usable_cols].fillna("unknown").astype(str).agg("||".join, axis=1)
        counts = labels_tv.value_counts()
        n_val = max(1, int(round(len(train_val) * adjusted_val_fraction)))
        if len(counts) <= 1 or counts.min() < 2 or n_val < len(counts):
            labels_tv = None

    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_fraction,
        random_state=seed,
        stratify=labels_tv,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _edge_key_set(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty:
        return set()
    return set(map(tuple, df[["source", "target", "domain"]].drop_duplicates().itertuples(index=False, name=None)))


def _filter_to_edge_keys(df: pd.DataFrame, edge_keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    keyed = df.assign(_edge_key=list(zip(df["source"], df["target"], df["domain"])))
    return keyed[keyed["_edge_key"].isin(edge_keys)].drop(columns="_edge_key").reset_index(drop=True)


def _validate_partitions(split_frames: dict[str, pd.DataFrame]) -> None:
    key_sets = {name: _edge_key_set(frame) for name, frame in split_frames.items()}
    splits = list(key_sets)
    for idx, left in enumerate(splits):
        for right in splits[idx + 1 :]:
            overlap = key_sets[left] & key_sets[right]
            if overlap:
                raise AssertionError(f"Edge leakage detected between {left} and {right}: {len(overlap)} overlapping edges")


def _pick_prompt_spec(metadata_items: list[dict | None]) -> dict | None:
    for item in metadata_items:
        if item and item.get("prompt_spec"):
            return item["prompt_spec"]
    return None


def _build_metadata(
    df: pd.DataFrame,
    *,
    split_name: str,
    source_name: str,
    objective_mode: str,
    seed: int,
    shared_edge_count: int,
    partition_edge_count: int,
    input_paths: list[str],
    input_metadata: list[dict | None],
    prompt_variant: str | None,
    stratify_cols: list[str],
) -> dict:
    metadata = {
        "split_name": split_name,
        "source_name": source_name,
        "objective_mode": objective_mode,
        "seed": seed,
        "row_count": int(len(df)),
        "unique_edge_count": int(df[["source", "target", "domain"]].drop_duplicates().shape[0]) if len(df) else 0,
        "shared_edge_count": int(shared_edge_count),
        "partition_edge_count": int(partition_edge_count),
        "domain_counts": summarize_counts(df["domain"].tolist()) if "domain" in df.columns else {},
        "judge_verdict_counts": summarize_counts(df["judge_verdict"].tolist()) if "judge_verdict" in df.columns else {},
        "prompt_variant_counts": summarize_counts(df["prompt_variant"].tolist()) if "prompt_variant" in df.columns else {},
        "input_paths": input_paths,
        "stratify_cols": stratify_cols,
    }
    if prompt_variant is not None:
        metadata["prompt_variant_filter"] = prompt_variant
    prompt_spec = _pick_prompt_spec(input_metadata)
    if prompt_spec is not None:
        metadata["prompt_spec"] = prompt_spec
    present_metadata = [item for item in input_metadata if item is not None]
    if present_metadata:
        metadata["source_dataset_metadata"] = present_metadata
    return metadata


def _save_dataset(df: pd.DataFrame, path: Path, metadata: dict) -> None:
    atomic_to_excel(df, path)
    write_json(
        sidecar_metadata_path(path),
        {
            **metadata,
            "file_name": path.name,
            "file_sha256": sha256_file(path),
        },
    )


def _mode_manifest(mode_root: Path) -> dict:
    files = {
        "shared_train_synth": mode_root / "shared_train_synth.xlsx",
        "shared_val_synth": mode_root / "shared_val_synth.xlsx",
        "shared_test_synth": mode_root / "shared_test_synth.xlsx",
        "shared_train_lit": mode_root / "shared_train_lit.xlsx",
        "shared_val_lit": mode_root / "shared_val_lit.xlsx",
        "shared_test_lit": mode_root / "shared_test_lit.xlsx",
    }
    payload = {}
    for key, path in files.items():
        payload[key] = {
            "path": str(path),
            "metadata": load_json(sidecar_metadata_path(path)),
        }
    return payload


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    input_root = _resolve_input_root(args.input_root)
    output_root = _resolve_output_root(args.output_root, input_root)
    _assert_output_root(output_root, args.force)
    stratify_cols = _parse_cols(args.stratify_cols)

    logger.info("Canonical input root: %s", input_root)
    logger.info("Shared-edge output root: %s", output_root)
    logger.info(
        "Shared-edge split fractions: val=%.2f test=%.2f seed=%d",
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    manifest = {
        "input_root": str(input_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "prompt_variant": args.prompt_variant,
        "stratify_cols": stratify_cols,
        "objective_modes": {},
    }

    canonical_manifest_path = input_root / "canonical_manifest.json"
    if canonical_manifest_path.exists():
        manifest["source_manifest_path"] = str(canonical_manifest_path)
        manifest["source_manifest"] = load_json(canonical_manifest_path)

    for objective_mode in OBJECTIVE_MODES:
        source_mode_root = input_root / objective_mode
        target_mode_root = output_root / objective_mode
        target_mode_root.mkdir(parents=True, exist_ok=True)

        synth_df, synth_meta, lit_df, lit_meta = _load_mode_inputs(source_mode_root)
        synth_df = _filter_prompt_variant(synth_df, args.prompt_variant)
        lit_df = _filter_prompt_variant(lit_df, args.prompt_variant)
        _require_columns(synth_df, ["prompt", "completion", "source", "target", "domain", "judge_verdict"], "GT Synth")
        _require_columns(lit_df, ["prompt", "completion", "source", "target", "domain", "judge_verdict"], "GT Lit")

        shared_edges = _build_shared_edge_ids(synth_df, lit_df)
        train_edges, val_edges, test_edges = _split_edge_ids(
            shared_edges,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            stratify_cols=stratify_cols,
        )

        edge_keys = {
            "train": set(map(tuple, train_edges[["source", "target", "domain"]].itertuples(index=False, name=None))),
            "val": set(map(tuple, val_edges[["source", "target", "domain"]].itertuples(index=False, name=None))),
            "test": set(map(tuple, test_edges[["source", "target", "domain"]].itertuples(index=False, name=None))),
        }
        _validate_partitions(
            {
                "train": _filter_to_edge_keys(shared_edges, edge_keys["train"]),
                "val": _filter_to_edge_keys(shared_edges, edge_keys["val"]),
                "test": _filter_to_edge_keys(shared_edges, edge_keys["test"]),
            }
        )

        split_payloads = {
            "shared_train_synth.xlsx": _filter_to_edge_keys(synth_df, edge_keys["train"]),
            "shared_val_synth.xlsx": _filter_to_edge_keys(synth_df, edge_keys["val"]),
            "shared_test_synth.xlsx": _filter_to_edge_keys(synth_df, edge_keys["test"]),
            "shared_train_lit.xlsx": _filter_to_edge_keys(lit_df, edge_keys["train"]),
            "shared_val_lit.xlsx": _filter_to_edge_keys(lit_df, edge_keys["val"]),
            "shared_test_lit.xlsx": _filter_to_edge_keys(lit_df, edge_keys["test"]),
        }

        for split in ["train", "val", "test"]:
            synth_keys = _edge_key_set(split_payloads[f"shared_{split}_synth.xlsx"])
            lit_keys = _edge_key_set(split_payloads[f"shared_{split}_lit.xlsx"])
            if synth_keys != lit_keys:
                raise AssertionError(f"Shared partition mismatch for {objective_mode} {split}: synth/lit edge sets differ")

        for file_name, df in split_payloads.items():
            source_name = "gt_synth" if file_name.endswith("_synth.xlsx") else "gt_lit"
            split_name = file_name.replace(".xlsx", "")
            split_key = split_name.split("_")[1]
            input_paths = [
                str(source_mode_root / "judge_train.xlsx"),
                str(source_mode_root / "judge_val_synth.xlsx"),
            ] if source_name == "gt_synth" else [str(source_mode_root / "judge_eval_gtlit.xlsx")]
            input_metadata = synth_meta if source_name == "gt_synth" else [lit_meta]
            _save_dataset(
                df,
                target_mode_root / file_name,
                _build_metadata(
                    df,
                    split_name=split_name,
                    source_name=source_name,
                    objective_mode=objective_mode,
                    seed=args.seed,
                    shared_edge_count=len(shared_edges),
                    partition_edge_count=len(edge_keys[split_key]),
                    input_paths=input_paths,
                    input_metadata=input_metadata,
                    prompt_variant=args.prompt_variant,
                    stratify_cols=stratify_cols,
                ),
            )
            logger.info("Saved: %s", target_mode_root / file_name)

        write_json(
            target_mode_root / "shared_edge_split_manifest.json",
            {
                "objective_mode": objective_mode,
                "shared_edge_count": int(len(shared_edges)),
                "train_edge_count": int(len(edge_keys["train"])),
                "val_edge_count": int(len(edge_keys["val"])),
                "test_edge_count": int(len(edge_keys["test"])),
                "stratify_cols": stratify_cols,
            },
        )
        manifest["objective_modes"][objective_mode] = _mode_manifest(target_mode_root)

    write_json(output_root / "shared_edge_canonical_manifest.json", manifest)
    logger.info("Shared-edge benchmark prepared at %s", output_root)


if __name__ == "__main__":
    main()
