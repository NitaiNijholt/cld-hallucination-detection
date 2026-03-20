"""Tests for data preparation."""

import tempfile
from pathlib import Path

import pytest

# Run from repo root so finetuning package is importable
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_prepare_judge_data_produces_expected_files():
    """Run prepare_judge_data with fixture; assert output files exist and have expected columns."""
    from finetuning.src.data.prepare_judge_data import main
    import argparse

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,  # 50% val for small fixture
            seed=42,
            min_file_bytes=0,  # include small fixture files
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
        )
        main(args=args)

        train_path = Path(out_dir) / "judge_train.xlsx"
        val_path = Path(out_dir) / "judge_val_synth.xlsx"
        lit_path = Path(out_dir) / "judge_eval_gtlit.xlsx"

        assert train_path.exists(), "judge_train.xlsx should exist"
        assert val_path.exists(), "judge_val_synth.xlsx should exist"
        assert lit_path.exists(), "judge_eval_gtlit.xlsx should exist"

        import pandas as pd
        for p, name in [(train_path, "train"), (val_path, "val"), (lit_path, "lit")]:
            df = pd.read_excel(p)
            required = ["prompt", "completion", "source", "target", "domain", "judge_verdict"]
            for col in required:
                assert col in df.columns, f"{name} should have column {col}"
            metadata_path = p.with_suffix(p.suffix + ".metadata.json")
            assert metadata_path.exists(), f"{metadata_path.name} should exist"


def test_prepare_judge_data_verdict_only_outputs_verdict_only_completion():
    """Verdict-only mode should build verdict-only completions and sidecar metadata."""
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
            objective_mode="verdict_only",
            stratify_cols="domain,judge_verdict",
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
        )
        main(args=args)

        train_path = Path(out_dir) / "judge_train.xlsx"
        train_df = pd.read_excel(train_path)
        assert train_df["completion"].str.startswith("VERDICT: ").all()
        assert not train_df["completion"].str.contains("REASON:").any()


def test_prepare_judge_data_can_filter_prompt_variant_and_balance_groups():
    """Prompt-variant filtering and exact balancing should be reflected in outputs."""
    from finetuning.src.data.prepare_judge_data import main
    import argparse
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        data_root = root / "final_runs"
        out_dir = root / "out"
        samples = [
            ("RQ1a_gt_synth_correctness", "depressive", "CORRECT"),
            ("RQ1a_gt_synth_correctness", "depressive", "INCORRECT"),
            ("RQ1a_gt_synth_correctness", "emergency_department", "CORRECT"),
            ("RQ1a_gt_synth_correctness", "emergency_department", "INCORRECT"),
            ("RQ1a_gt_lit_correctness", "depressive", "CORRECT"),
            ("RQ1a_gt_lit_correctness", "depressive", "INCORRECT"),
            ("RQ1a_gt_lit_correctness", "emergency_department", "CORRECT"),
            ("RQ1a_gt_lit_correctness", "emergency_department", "INCORRECT"),
        ]
        for dataset, domain, verdict in samples:
            run_dir = data_root / dataset / domain / "run_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                [
                    {
                        "Source": f"{domain}-{verdict}-1",
                        "Target": "T1",
                        "Relationship Type": "causes",
                        "Motivation": f"{domain} {verdict}",
                        "Judge Verdict": verdict,
                        "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
                    },
                    {
                        "Source": f"{domain}-{verdict}-2",
                        "Target": "T2",
                        "Relationship Type": "causes",
                        "Motivation": f"{domain} {verdict}",
                        "Judge Verdict": verdict,
                        "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
                    },
                ]
            )
            df.to_excel(run_dir / f"judged_demo_mechanistic_{domain}_{verdict}.xlsx", index=False)
            df.to_excel(run_dir / f"judged_demo_cot_{domain}_{verdict}.xlsx", index=False)

        args = argparse.Namespace(
            data_root=str(data_root),
            output_dir=str(out_dir),
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
            objective_mode="reason_verdict",
            stratify_cols="domain",
            require_stratification=False,
            prompt_variant="mechanistic",
            balance_cols="domain,judge_verdict",
            balance_samples_per_group=None,
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
        )
        main(args=args)

        train_df = pd.read_excel(out_dir / "judge_train.xlsx")
        lit_df = pd.read_excel(out_dir / "judge_eval_gtlit.xlsx")
        assert set(train_df["prompt_variant"]) == {"mechanistic"}
        assert set(lit_df["prompt_variant"]) == {"mechanistic"}
        assert train_df.groupby(["domain", "judge_verdict"]).size().nunique() == 1
        assert lit_df.groupby(["domain", "judge_verdict"]).size().nunique() == 1


def test_prepare_judge_data_can_exclude_gt_lit_overlap_from_synth():
    """Overlapping GT Synth edges should be removed when GT Lit is treated as held-out."""
    from finetuning.src.data.prepare_judge_data import main
    import argparse
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        data_root = root / "final_runs"
        out_dir = root / "out"

        def write_sheet(dataset: str, domain: str, filename: str, rows: list[dict]) -> None:
            run_dir = data_root / dataset / domain / "run_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_excel(run_dir / filename, index=False)

        shared_rows = [
            {
                "Source": "shared_s",
                "Target": "shared_t",
                "Relationship Type": "causes",
                "Motivation": "shared edge",
                "Judge Verdict": "CORRECT",
                "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
            },
            {
                "Source": "synth_only_s",
                "Target": "synth_only_t",
                "Relationship Type": "causes",
                "Motivation": "synth only edge",
                "Judge Verdict": "INCORRECT",
                "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
            },
        ]
        lit_rows = [
            {
                "Source": "shared_s",
                "Target": "shared_t",
                "Relationship Type": "causes",
                "Motivation": "shared edge",
                "Judge Verdict": "CORRECT",
                "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
            },
            {
                "Source": "lit_only_s",
                "Target": "lit_only_t",
                "Relationship Type": "causes",
                "Motivation": "lit only edge",
                "Judge Verdict": "PARTIALLY_CORRECT",
                "Judge Message": '{"judge_results": [{"reason": "Ok"}]}',
            },
        ]
        write_sheet(
            "RQ1a_gt_synth_correctness",
            "depressive",
            "judged_demo_mechanistic_synth.xlsx",
            shared_rows,
        )
        write_sheet(
            "RQ1a_gt_lit_correctness",
            "depressive",
            "judged_demo_mechanistic_lit.xlsx",
            lit_rows,
        )

        args = argparse.Namespace(
            data_root=str(data_root),
            output_dir=str(out_dir),
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
            objective_mode="reason_verdict",
            stratify_cols="domain",
            require_stratification=False,
            prompt_variant="mechanistic",
            balance_cols="",
            balance_samples_per_group=None,
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
            exclude_lit_overlap_from_synth=True,
        )
        main(args=args)

        train_df = pd.read_excel(out_dir / "judge_train.xlsx")
        val_df = pd.read_excel(out_dir / "judge_val_synth.xlsx")
        synth_edges = set(map(tuple, pd.concat([train_df, val_df])[["source", "target", "domain"]].drop_duplicates().itertuples(index=False, name=None)))
        lit_df = pd.read_excel(out_dir / "judge_eval_gtlit.xlsx")
        lit_edges = set(map(tuple, lit_df[["source", "target", "domain"]].drop_duplicates().itertuples(index=False, name=None)))

        assert ("shared_s", "shared_t", "depressive") not in synth_edges
        assert ("shared_s", "shared_t", "depressive") in lit_edges
        assert synth_edges.isdisjoint(lit_edges)


def test_prepare_gt_lit_train_val_produces_holdout_split():
    """GT Lit split creates train/val/test files with no edge leakage."""
    import argparse
    import pandas as pd

    from finetuning.src.data.prepare_gt_lit_train_val import main as split_main
    from finetuning.src.data.prepare_judge_data import main as prepare_main

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        prep_args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
        )
        prepare_main(args=prep_args)

        split_args = argparse.Namespace(
            input_path=str(Path(out_dir) / "judge_eval_gtlit.xlsx"),
            output_dir=out_dir,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
        split_main(args=split_args)

        train_path = Path(out_dir) / "judge_train_gtlit.xlsx"
        val_path = Path(out_dir) / "judge_val_gtlit.xlsx"
        test_path = Path(out_dir) / "judge_test_gtlit.xlsx"
        for path in [train_path, val_path, test_path]:
            assert path.exists(), f"{path.name} should exist"

        train_df = pd.read_excel(train_path)
        val_df = pd.read_excel(val_path)
        test_df = pd.read_excel(test_path)

        train_edges = set(zip(train_df["source"], train_df["target"], train_df["domain"]))
        val_edges = set(zip(val_df["source"], val_df["target"], val_df["domain"]))
        test_edges = set(zip(test_df["source"], test_df["target"], test_df["domain"]))

        assert train_edges.isdisjoint(val_edges)
        assert train_edges.isdisjoint(test_edges)
        assert val_edges.isdisjoint(test_edges)


def test_create_gt_lit_test_subsample_from_holdout():
    """Held-out GT Lit test subsample is created from the test split only."""
    import argparse
    import pandas as pd

    from finetuning.src.data.prepare_gt_lit_train_val import main as split_main
    from finetuning.src.data.prepare_judge_data import main as prepare_main
    from finetuning.src.evaluation.create_gt_lit_test_subsample import main as subsample_main

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        prep_args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
        )
        prepare_main(args=prep_args)

        split_args = argparse.Namespace(
            input_path=str(Path(out_dir) / "judge_eval_gtlit.xlsx"),
            output_dir=out_dir,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
        split_main(args=split_args)

        import sys
        orig_argv = sys.argv
        try:
            sys.argv = [
                "create_gt_lit_test_subsample",
                "--input_path", str(Path(out_dir) / "judge_test_gtlit.xlsx"),
                "--output_path", str(Path(out_dir) / "judge_test_gtlit_1k.xlsx"),
                "--target_n", "5",
                "--seed", "42",
            ]
            subsample_main()
        finally:
            sys.argv = orig_argv

        test_df = pd.read_excel(Path(out_dir) / "judge_test_gtlit.xlsx")
        sub_df = pd.read_excel(Path(out_dir) / "judge_test_gtlit_1k.xlsx")
        assert len(sub_df) <= min(5, len(test_df))
        metadata_path = Path(out_dir) / "judge_test_gtlit_1k.xlsx.metadata.json"
        assert metadata_path.exists()


def test_create_eval_subsamples_creates_stratified_outputs():
    """Frozen eval subsets should be created with the configured output suffix."""
    import argparse

    from finetuning.src.data.prepare_judge_data import main as prepare_main
    from finetuning.src.evaluation.create_eval_subsamples import main as subsample_main

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        prep_args = argparse.Namespace(
            data_root=str(fixture_root),
            output_dir=out_dir,
            val_fraction=0.5,
            seed=42,
            min_file_bytes=0,
            objective_mode="reason_verdict",
            stratify_cols="domain,judge_verdict",
            balance_lit_eval_cols="",
            balance_lit_eval_samples_per_group=None,
        )
        prepare_main(args=prep_args)

        subsample_args = argparse.Namespace(
            val_path=str(Path(out_dir) / "judge_val_synth.xlsx"),
            lit_path=str(Path(out_dir) / "judge_eval_gtlit.xlsx"),
            target_n=2,
            seed=42,
            stratify_cols="domain,judge_verdict",
            output_suffix="_1k_stratified",
        )
        subsample_main(args=subsample_args)

        assert (Path(out_dir) / "judge_val_synth_1k_stratified.xlsx").exists()
        assert (Path(out_dir) / "judge_eval_gtlit_1k_stratified.xlsx").exists()


def test_create_eval_subsamples_can_exact_balance_domain_and_verdict():
    import argparse
    import pandas as pd

    from finetuning.src.evaluation.create_eval_subsamples import main as subsample_main

    with tempfile.TemporaryDirectory() as out_dir:
        base = Path(out_dir)
        df = pd.DataFrame(
            [
                {"prompt": "p1", "completion": "c1", "domain": "d1", "judge_verdict": "CORRECT"},
                {"prompt": "p2", "completion": "c2", "domain": "d1", "judge_verdict": "CORRECT"},
                {"prompt": "p3", "completion": "c3", "domain": "d1", "judge_verdict": "INCORRECT"},
                {"prompt": "p4", "completion": "c4", "domain": "d1", "judge_verdict": "INCORRECT"},
                {"prompt": "p5", "completion": "c5", "domain": "d2", "judge_verdict": "CORRECT"},
                {"prompt": "p6", "completion": "c6", "domain": "d2", "judge_verdict": "CORRECT"},
                {"prompt": "p7", "completion": "c7", "domain": "d2", "judge_verdict": "INCORRECT"},
                {"prompt": "p8", "completion": "c8", "domain": "d2", "judge_verdict": "INCORRECT"},
            ]
        )
        val_path = base / "judge_val_synth.xlsx"
        lit_path = base / "judge_eval_gtlit.xlsx"
        df.to_excel(val_path, index=False)
        df.to_excel(lit_path, index=False)

        args = argparse.Namespace(
            val_path=str(val_path),
            lit_path=str(lit_path),
            target_n=1000,
            seed=42,
            stratify_cols="domain,judge_verdict",
            output_suffix="_balanced",
            require_stratification=True,
            balance_cols="domain,judge_verdict",
            balance_samples_per_group=2,
        )
        subsample_main(args=args)

        balanced = pd.read_excel(base / "judge_eval_gtlit_balanced.xlsx")
        assert balanced.groupby(["domain", "judge_verdict"]).size().to_dict() == {
            ("d1", "CORRECT"): 2,
            ("d1", "INCORRECT"): 2,
            ("d2", "CORRECT"): 2,
            ("d2", "INCORRECT"): 2,
        }


def test_create_gt_lit_test_subsample_can_exact_balance_domain_and_verdict():
    import argparse
    import pandas as pd

    from finetuning.src.evaluation.create_gt_lit_test_subsample import main as subsample_main

    with tempfile.TemporaryDirectory() as out_dir:
        base = Path(out_dir)
        df = pd.DataFrame(
            [
                {"prompt": "p1", "completion": "c1", "domain": "d1", "judge_verdict": "CORRECT"},
                {"prompt": "p2", "completion": "c2", "domain": "d1", "judge_verdict": "CORRECT"},
                {"prompt": "p3", "completion": "c3", "domain": "d1", "judge_verdict": "INCORRECT"},
                {"prompt": "p4", "completion": "c4", "domain": "d1", "judge_verdict": "INCORRECT"},
                {"prompt": "p5", "completion": "c5", "domain": "d2", "judge_verdict": "CORRECT"},
                {"prompt": "p6", "completion": "c6", "domain": "d2", "judge_verdict": "CORRECT"},
                {"prompt": "p7", "completion": "c7", "domain": "d2", "judge_verdict": "INCORRECT"},
                {"prompt": "p8", "completion": "c8", "domain": "d2", "judge_verdict": "INCORRECT"},
            ]
        )
        input_path = base / "judge_test_gtlit.xlsx"
        output_path = base / "judge_test_gtlit_balanced.xlsx"
        df.to_excel(input_path, index=False)

        args = argparse.Namespace(
            input_path=str(input_path),
            output_path=str(output_path),
            target_n=1000,
            seed=42,
            stratify_cols="domain,judge_verdict",
            require_stratification=True,
            balance_cols="domain,judge_verdict",
            balance_samples_per_group=2,
        )
        subsample_main(args=args)

        balanced = pd.read_excel(output_path)
        assert balanced.groupby(["domain", "judge_verdict"]).size().to_dict() == {
            ("d1", "CORRECT"): 2,
            ("d1", "INCORRECT"): 2,
            ("d2", "CORRECT"): 2,
            ("d2", "INCORRECT"): 2,
        }


def test_prepare_canonical_matrix_data_creates_frozen_bundle():
    """Canonical prep should generate immutable train/val/test/eval artifacts for both objectives."""
    import argparse

    from finetuning.src.data.prepare_canonical_matrix_data import main as canonical_main

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "final_runs"
    with tempfile.TemporaryDirectory() as out_dir:
        args = argparse.Namespace(
            output_root=str(Path(out_dir) / "canonical"),
            data_root=str(fixture_root),
            seed=42,
            val_fraction=0.2,
            test_fraction=0.2,
            target_n=1000,
            min_file_bytes=0,
            stratify_cols="",
            force=False,
            lit_eval_balance_cols="",
            lit_eval_balance_samples_per_group=None,
        )
        canonical_main(args=args)

        root = Path(out_dir) / "canonical"
        for objective_mode in ["reason_verdict", "verdict_only"]:
            mode_root = root / objective_mode
            assert (mode_root / "judge_train.xlsx").exists()
            assert (mode_root / "judge_val_synth.xlsx").exists()
            assert (mode_root / "judge_eval_gtlit.xlsx").exists()
            assert (mode_root / "judge_val_synth_1k_stratified.xlsx").exists()
            assert (mode_root / "judge_eval_gtlit_1k_stratified.xlsx").exists()
            assert (mode_root / "gt_lit_trainval" / "judge_train_gtlit.xlsx").exists()
            assert (mode_root / "gt_lit_trainval" / "judge_val_gtlit.xlsx").exists()
            assert (mode_root / "gt_lit_trainval" / "judge_test_gtlit.xlsx").exists()
            assert (mode_root / "gt_lit_trainval" / "judge_test_gtlit_1k_stratified.xlsx").exists()

        assert (root / "canonical_manifest.json").exists()
