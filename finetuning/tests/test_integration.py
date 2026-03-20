"""Integration tests: data prep -> training data format compatibility."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_data_prep_output_compatible_with_training():
    """Data prep outputs can be loaded by training's load_split (format check)."""
    from finetuning.src.data.prepare_judge_data import main
    import argparse
    import pandas as pd

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
        )
        main(args=args)

        train_path = Path(out_dir) / "judge_train.xlsx"
        val_path = Path(out_dir) / "judge_val_synth.xlsx"

        for path in [train_path, val_path]:
            df = pd.read_excel(path).dropna(subset=["prompt", "completion"])
            assert len(df) > 0, f"{path.name} should have rows"
            assert "prompt" in df.columns and "completion" in df.columns
            for _, row in df.iterrows():
                assert isinstance(row["prompt"], str) and len(row["prompt"]) > 0
                assert isinstance(row["completion"], str) and len(row["completion"]) > 0
                assert "VERDICT:" in row["completion"] or "REASON:" in row["completion"]


def test_full_config_flow():
    """Config loads, paths resolve, and training args can be built (when deps available)."""
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")

    from hydra.core.global_hydra import GlobalHydra
    from hydra import compose, initialize_config_dir
    from finetuning.src.training.train_config import get_config_dir, get_repo_root, resolve_paths
    from finetuning.src.training.train_judge import build_training_args

    config_dir = get_config_dir()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="smoke", overrides=[])

    repo = get_repo_root()
    cfg = resolve_paths(cfg, repo)
    args = build_training_args(cfg)

    assert args.num_train_epochs == 1
    assert cfg.smoke_test is True


def test_prepare_shared_edge_matrix_data_builds_aligned_disjoint_splits():
    """Shared-edge prep should keep synth/lit partitions aligned without leakage."""
    import argparse
    import pandas as pd

    from finetuning.src.data.prepare_shared_edge_matrix_data import main

    prompt_spec = {
        "system_prompt": "Judge causal explanations.",
        "user_template": "SOURCE: {source}",
        "completion_template": "VERDICT: {verdict}",
    }

    def make_rows(prefix: str, verdicts: list[str], *, extra_edge: tuple[str, str] | None = None):
        rows = []
        for idx, verdict in enumerate(verdicts):
            rows.append(
                {
                    "prompt": f"{prefix} prompt {idx}",
                    "completion": f"VERDICT: {verdict}",
                    "source": f"S{idx}",
                    "target": f"T{idx}",
                    "domain": "depressive",
                    "judge_verdict": verdict,
                    "prompt_variant": "mechanistic",
                }
            )
        if extra_edge is not None:
            rows.append(
                {
                    "prompt": f"{prefix} extra",
                    "completion": "VERDICT: CORRECT",
                    "source": extra_edge[0],
                    "target": extra_edge[1],
                    "domain": "depressive",
                    "judge_verdict": "CORRECT",
                    "prompt_variant": "mechanistic",
                }
            )
        return pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_root = tmp_root / "canonical_input"
        output_root = tmp_root / "canonical_shared"

        for objective_mode in ["reason_verdict", "verdict_only"]:
            mode_root = input_root / objective_mode
            mode_root.mkdir(parents=True, exist_ok=True)

            synth_train = make_rows("synth_train", ["CORRECT", "INCORRECT", "PARTIALLY_CORRECT"])
            synth_val = make_rows("synth_val", ["CORRECT", "INCORRECT", "PARTIALLY_CORRECT"], extra_edge=("ONLY_SYNTH", "X"))
            lit_eval = make_rows("lit_eval", ["INCORRECT", "PARTIALLY_CORRECT", "CORRECT", "CORRECT", "INCORRECT", "PARTIALLY_CORRECT"], extra_edge=("ONLY_LIT", "Y"))

            synth_train.to_excel(mode_root / "judge_train.xlsx", index=False)
            synth_val.to_excel(mode_root / "judge_val_synth.xlsx", index=False)
            lit_eval.to_excel(mode_root / "judge_eval_gtlit.xlsx", index=False)

            for stem in ["judge_train.xlsx", "judge_val_synth.xlsx", "judge_eval_gtlit.xlsx"]:
                metadata_path = mode_root / f"{stem}.metadata.json"
                metadata_path.write_text(json.dumps({"prompt_spec": prompt_spec}), encoding="utf-8")

        main(
            args=argparse.Namespace(
                input_root=str(input_root),
                output_root=str(output_root),
                seed=42,
                val_fraction=0.17,
                test_fraction=0.17,
                prompt_variant="mechanistic",
                stratify_cols="domain",
                force=False,
            )
        )

        for objective_mode in ["reason_verdict", "verdict_only"]:
            mode_root = output_root / objective_mode
            synth_train = pd.read_excel(mode_root / "shared_train_synth.xlsx")
            synth_val = pd.read_excel(mode_root / "shared_val_synth.xlsx")
            synth_test = pd.read_excel(mode_root / "shared_test_synth.xlsx")
            lit_train = pd.read_excel(mode_root / "shared_train_lit.xlsx")
            lit_val = pd.read_excel(mode_root / "shared_val_lit.xlsx")
            lit_test = pd.read_excel(mode_root / "shared_test_lit.xlsx")

            def edge_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
                return set(map(tuple, df[["source", "target", "domain"]].drop_duplicates().itertuples(index=False, name=None)))

            train_keys = edge_keys(synth_train)
            val_keys = edge_keys(synth_val)
            test_keys = edge_keys(synth_test)

            assert train_keys
            assert val_keys
            assert test_keys
            assert train_keys.isdisjoint(val_keys)
            assert train_keys.isdisjoint(test_keys)
            assert val_keys.isdisjoint(test_keys)
            assert train_keys == edge_keys(lit_train)
            assert val_keys == edge_keys(lit_val)
            assert test_keys == edge_keys(lit_test)
            assert ("ONLY_SYNTH", "X", "depressive") not in train_keys | val_keys | test_keys
            assert ("ONLY_LIT", "Y", "depressive") not in train_keys | val_keys | test_keys
            assert (mode_root / "shared_edge_split_manifest.json").exists()
