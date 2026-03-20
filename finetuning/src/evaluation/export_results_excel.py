#!/usr/bin/env python3
"""Export evaluation result JSONs to an Excel workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

THESIS_BASELINE_ROWS = [
    {
        "result_dir": "thesis_baseline_reference",
        "model": "GPT-4.1 thesis baseline",
        "train_source": "none",
        "objective": "thesis setup",
        "dataset": "Shared GT Synth Test",
        "dataset_key": "synth_test",
        "f1": 0.74,
        "auc": 0.86,
        "loss": None,
        "skip_rate": None,
        "accuracy": None,
        "n_scored": None,
        "n_total": None,
        "eval_mode": "thesis_reference",
        "ground_truth_source": "thesis_reference",
        "f1_metric": "f1",
        "kind": "external_baseline",
        "source_type": "thesis_baseline",
        "source_ref": "finetuning/src/evaluation/evaluate_judge.py:THESIS_BASELINES",
        "notes": "Imported thesis GPT-4.1 reference for GT Synth.",
    },
    {
        "result_dir": "thesis_baseline_reference",
        "model": "GPT-4.1 thesis baseline",
        "train_source": "none",
        "objective": "thesis setup",
        "dataset": "Shared GT Lit Test",
        "dataset_key": "lit_test",
        "f1": 0.35,
        "auc": 0.60,
        "loss": None,
        "skip_rate": None,
        "accuracy": None,
        "n_scored": None,
        "n_total": None,
        "eval_mode": "thesis_reference",
        "ground_truth_source": "thesis_reference",
        "f1_metric": "f1",
        "kind": "external_baseline",
        "source_type": "thesis_baseline",
        "source_ref": "finetuning/src/evaluation/evaluate_judge.py:THESIS_BASELINES",
        "notes": "Imported thesis GPT-4.1 reference for GT Lit.",
    },
]

THESIS_BASELINE_SOURCE_ROWS = [
    {
        "source_name": "GPT-4.1 thesis baseline",
        "dataset": "GT Synth",
        "f1": 0.74,
        "auc": 0.86,
        "source_ref": "finetuning/src/evaluation/evaluate_judge.py:THESIS_BASELINES",
        "secondary_ref": "finetuning/src/evaluation/eval_classification_compare.py:THESIS_BASELINES",
        "notes": "Reference baseline carried forward from the thesis-era evaluation scripts.",
    },
    {
        "source_name": "GPT-4.1 thesis baseline",
        "dataset": "GT Lit",
        "f1": 0.35,
        "auc": 0.60,
        "source_ref": "finetuning/src/evaluation/evaluate_judge.py:THESIS_BASELINES",
        "secondary_ref": "finetuning/src/evaluation/eval_classification_compare.py:THESIS_BASELINES",
        "notes": "Reference baseline carried forward from the thesis-era evaluation scripts.",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export evaluation results to Excel")
    p.add_argument("--results-root", required=True, help="Directory containing eval result subdirectories")
    p.add_argument("--output-xlsx", required=True, help="Path to output .xlsx workbook")
    p.add_argument(
        "--glob-pattern",
        default="*",
        help="Glob pattern for result subdirectories under results-root",
    )
    p.add_argument(
        "--include-external-baselines",
        action="store_true",
        help="Include external baseline rows if present in evaluation_results.json",
    )
    p.add_argument(
        "--include-thesis-baseline",
        action="store_true",
        help="Append GPT-4.1 thesis baseline rows and provenance sheet",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_train_source(model_label: str) -> str:
    if "Shared GT Synth trained" in model_label:
        return "GT Synth"
    if "Shared GT Lit trained" in model_label:
        return "GT Lit"
    if "base" in model_label.lower():
        return "none"
    if "GPT-4.1" in model_label:
        return "none"
    return "unknown"


def _infer_objective(model_label: str, result_dir_name: str) -> str:
    label = model_label.lower()
    name = result_dir_name.lower()
    if "base" in label and "reason_verdict" in name:
        return "reason_verdict prompts"
    if "base" in label and "verdict_only" in name:
        return "verdict_only prompts"
    if "reason+verdict" in label or "reason_verdict" in name:
        return "reason_verdict"
    if "verdict-only" in label or "verdict_only" in name:
        return "verdict_only"
    if "gpt-4.1" in label:
        return "thesis setup"
    return "n/a"


def _dataset_prefix(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if "synth" in lowered:
        return "synth_test"
    if "lit" in lowered:
        return "lit_test"
    return lowered.replace(" ", "_")


def _collect_results(
    results_root: Path,
    glob_pattern: str,
    include_external_baselines: bool,
    include_thesis_baseline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict] = []
    metadata_rows: list[dict] = []
    source_rows: list[dict] = []

    for result_dir in sorted(results_root.glob(glob_pattern)):
        if not result_dir.is_dir():
            continue
        results_path = result_dir / "evaluation_results.json"
        metadata_path = result_dir / "evaluation_metadata.json"
        results_payload = _read_json(results_path)
        metadata_payload = _read_json(metadata_path)
        if not isinstance(results_payload, list):
            continue

        for row in results_payload:
            if row.get("kind") == "external_baseline" and not include_external_baselines:
                continue
            model_label = str(row.get("model_label", "unknown"))
            dataset = str(row.get("dataset", "unknown"))
            long_rows.append(
                {
                    "result_dir": result_dir.name,
                    "model": model_label,
                    "train_source": _infer_train_source(model_label),
                    "objective": _infer_objective(model_label, result_dir.name),
                    "dataset": dataset,
                    "dataset_key": _dataset_prefix(dataset),
                    "f1": row.get("f1_macro", row.get("f1")),
                    "auc": row.get("auc"),
                    "loss": row.get("eval_loss"),
                    "skip_rate": row.get("skip_rate"),
                    "accuracy": row.get("accuracy"),
                    "n_scored": row.get("n_scored"),
                    "n_total": row.get("n_total"),
                    "eval_mode": row.get("eval_mode"),
                    "ground_truth_source": row.get("ground_truth_source"),
                    "f1_metric": row.get("f1_metric"),
                    "kind": row.get("kind", "model"),
                }
            )

        if isinstance(metadata_payload, dict):
            datasets = metadata_payload.get("datasets", {})
            first_split = datasets.get("first_split", {})
            second_split = datasets.get("second_split", {})
            for split_name, split_data in [("first_split", first_split), ("second_split", second_split)]:
                if not isinstance(split_data, dict):
                    continue
                sidecar = split_data.get("sidecar", {})
                metadata_rows.append(
                    {
                        "result_dir": result_dir.name,
                        "split_slot": split_name,
                        "label": split_data.get("label"),
                        "path": split_data.get("path"),
                        "sha256": split_data.get("sha256"),
                        "row_count": sidecar.get("row_count"),
                        "unique_edge_count": sidecar.get("unique_edge_count"),
                        "shared_edge_count": sidecar.get("shared_edge_count"),
                        "partition_edge_count": sidecar.get("partition_edge_count"),
                        "objective_mode": sidecar.get("objective_mode"),
                        "source_name": sidecar.get("source_name"),
                    }
                )

    if include_thesis_baseline:
        long_rows.extend(THESIS_BASELINE_ROWS)
        source_rows.extend(THESIS_BASELINE_SOURCE_ROWS)
    long_df = pd.DataFrame(long_rows)
    metadata_df = pd.DataFrame(metadata_rows)
    source_df = pd.DataFrame(source_rows)
    return long_df, metadata_df, source_df


def _build_wide_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    value_cols = ["f1", "auc", "loss", "skip_rate", "accuracy", "n_scored", "eval_mode", "ground_truth_source"]
    index_cols = ["model", "objective", "train_source", "kind"]
    pivot = long_df.pivot_table(
        index=index_cols,
        columns="dataset_key",
        values=value_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{dataset}_{metric}" for metric, dataset in pivot.columns]
    wide_df = pivot.reset_index()
    run_dirs = (
        long_df.groupby(index_cols, dropna=False)["result_dir"]
        .agg(lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)})))
        .reset_index(name="result_dirs")
    )
    wide_df = wide_df.merge(run_dirs, on=index_cols, how="left")

    preferred_order = [
        "model",
        "objective",
        "train_source",
        "kind",
        "result_dirs",
        "synth_test_f1",
        "synth_test_auc",
        "synth_test_loss",
        "synth_test_skip_rate",
        "synth_test_accuracy",
        "synth_test_n_scored",
        "synth_test_ground_truth_source",
        "lit_test_f1",
        "lit_test_auc",
        "lit_test_loss",
        "lit_test_skip_rate",
        "lit_test_accuracy",
        "lit_test_n_scored",
        "lit_test_ground_truth_source",
        "synth_test_eval_mode",
        "lit_test_eval_mode",
    ]
    existing = [col for col in preferred_order if col in wide_df.columns]
    remaining = [col for col in wide_df.columns if col not in existing]
    return wide_df[existing + remaining]


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_xlsx = Path(args.output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    long_df, metadata_df, source_df = _collect_results(
        results_root=results_root,
        glob_pattern=args.glob_pattern,
        include_external_baselines=args.include_external_baselines,
        include_thesis_baseline=args.include_thesis_baseline,
    )
    wide_df = _build_wide_summary(long_df)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="summary_wide", index=False)
        long_df.to_excel(writer, sheet_name="summary_long", index=False)
        metadata_df.to_excel(writer, sheet_name="dataset_metadata", index=False)
        source_df.to_excel(writer, sheet_name="thesis_baseline_source", index=False)

    print(f"Wrote Excel workbook to {output_xlsx}")


if __name__ == "__main__":
    main()
