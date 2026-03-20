#!/usr/bin/env python3
"""Generate immutable train/val/test/eval artifacts for the canonical judge matrix."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ..metadata_utils import load_json, sidecar_metadata_path, write_json
from . import prepare_gt_lit_train_val, prepare_judge_data
from ..evaluation import create_eval_subsamples, create_gt_lit_test_subsample


OBJECTIVE_MODES = ["reason_verdict", "verdict_only"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=_repo_root(),
        ).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    default_root = repo_root / "finetuning" / "data" / "canonical" / _git_sha()
    p = argparse.ArgumentParser(description="Prepare canonical frozen judge matrix datasets")
    p.add_argument("--output-root", default=str(default_root))
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--target-n", type=int, default=1000)
    p.add_argument("--min-file-bytes", type=int, default=0)
    p.add_argument("--stratify-cols", default="domain,judge_verdict")
    p.add_argument("--prompt-variant", choices=prepare_judge_data.PROMPT_VARIANTS, default=None)
    p.add_argument("--balance-cols", default="")
    p.add_argument("--balance-samples-per-group", type=int, default=None)
    p.add_argument("--lit-eval-balance-cols", default="")
    p.add_argument("--lit-eval-balance-samples-per-group", type=int, default=None)
    p.add_argument("--preserve-split-group-cols", default="")
    p.add_argument("--eval-balance-cols", default="")
    p.add_argument("--eval-balance-samples-per-group", type=int, default=None)
    p.add_argument("--exclude-lit-overlap-from-synth", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output root",
    )
    return p.parse_args()


def _assert_output_root(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"Output root already exists and is not empty: {path}. "
            "Use --force only if you intentionally want to regenerate canonical artifacts."
        )
    path.mkdir(parents=True, exist_ok=True)


def _mode_manifest(mode_root: Path) -> dict:
    files = {
        "judge_train": mode_root / "judge_train.xlsx",
        "judge_val_synth": mode_root / "judge_val_synth.xlsx",
        "judge_eval_gtlit": mode_root / "judge_eval_gtlit.xlsx",
        "judge_val_synth_1k": mode_root / "judge_val_synth_1k_stratified.xlsx",
        "judge_eval_gtlit_1k": mode_root / "judge_eval_gtlit_1k_stratified.xlsx",
        "judge_train_gtlit": mode_root / "gt_lit_trainval" / "judge_train_gtlit.xlsx",
        "judge_val_gtlit": mode_root / "gt_lit_trainval" / "judge_val_gtlit.xlsx",
        "judge_test_gtlit": mode_root / "gt_lit_trainval" / "judge_test_gtlit.xlsx",
        "judge_test_gtlit_1k": mode_root / "gt_lit_trainval" / "judge_test_gtlit_1k_stratified.xlsx",
    }
    payload = {}
    for key, path in files.items():
        sidecar = load_json(sidecar_metadata_path(path))
        payload[key] = {
            "path": str(path),
            "metadata": sidecar,
        }
    return payload


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    output_root = Path(args.output_root)
    _assert_output_root(output_root, args.force)

    manifest = {
        "git_sha": _git_sha(),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "target_n": args.target_n,
        "stratify_cols": args.stratify_cols,
        "objective_modes": {},
    }

    for objective_mode in OBJECTIVE_MODES:
        mode_root = output_root / objective_mode
        mode_root.mkdir(parents=True, exist_ok=True)

        prepare_judge_data.main(
            argparse.Namespace(
                data_root=args.data_root,
                output_dir=str(mode_root),
                val_fraction=args.val_fraction,
                seed=args.seed,
                min_file_bytes=args.min_file_bytes,
                objective_mode=objective_mode,
                stratify_cols=args.stratify_cols,
                require_stratification=True,
                prompt_variant=getattr(args, "prompt_variant", None),
                balance_cols=getattr(args, "balance_cols", ""),
                balance_samples_per_group=getattr(args, "balance_samples_per_group", None),
                balance_lit_eval_cols=getattr(args, "lit_eval_balance_cols", ""),
                balance_lit_eval_samples_per_group=getattr(args, "lit_eval_balance_samples_per_group", None),
                preserve_split_group_cols=getattr(args, "preserve_split_group_cols", ""),
                exclude_lit_overlap_from_synth=getattr(args, "exclude_lit_overlap_from_synth", False),
            )
        )

        prepare_gt_lit_train_val.main(
            argparse.Namespace(
                input_path=str(mode_root / "judge_eval_gtlit.xlsx"),
                output_dir=str(mode_root / "gt_lit_trainval"),
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                stratify_cols=args.stratify_cols,
                require_stratification=True,
                balance_cols=getattr(args, "balance_cols", ""),
                balance_samples_per_group=None,
            )
        )

        create_eval_subsamples.main(
            argparse.Namespace(
                val_path=str(mode_root / "judge_val_synth.xlsx"),
                lit_path=str(mode_root / "judge_eval_gtlit.xlsx"),
                target_n=args.target_n,
                seed=args.seed,
                stratify_cols=args.stratify_cols,
                output_suffix="_1k_stratified",
                require_stratification=True,
                balance_cols=getattr(args, "eval_balance_cols", ""),
                balance_samples_per_group=getattr(args, "eval_balance_samples_per_group", None),
            )
        )

        create_gt_lit_test_subsample.main(
            argparse.Namespace(
                input_path=str(mode_root / "gt_lit_trainval" / "judge_test_gtlit.xlsx"),
                output_path=str(mode_root / "gt_lit_trainval" / "judge_test_gtlit_1k_stratified.xlsx"),
                target_n=args.target_n,
                seed=args.seed,
                stratify_cols=args.stratify_cols,
                require_stratification=True,
                balance_cols=getattr(args, "eval_balance_cols", ""),
                balance_samples_per_group=getattr(args, "eval_balance_samples_per_group", None),
            )
        )

        manifest["objective_modes"][objective_mode] = _mode_manifest(mode_root)

    write_json(output_root / "canonical_manifest.json", manifest)
    print(f"Canonical matrix data prepared at {output_root}")


if __name__ == "__main__":
    main()
