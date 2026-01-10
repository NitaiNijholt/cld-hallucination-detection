#!/usr/bin/env python3
"""
RQ2 Unified Analysis Script
Single entry point for all RQ2-related analyses:
0. Input inventory + deduplication (reproducible file list)
1. Phases 1-3: Individual Metrics & Batch Analysis
1b. Thesis Table: Single-metric meta-analysis (block-level AUC vs 0.5)
2. Phase 4: RFE Feature Selection
3. Phase 5: Ensemble Comparison
4. Phase 6: Cross-Domain Generalization
5. Master Report Generation

Usage:
    python3 final_runs/RQ2_unified_analysis.py
"""

import sys
import os
import subprocess
import argparse
import shutil
import glob
from datetime import datetime
from pathlib import Path

def run_command(cmd, description, *, env=None):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
        print(f"\n✅ Successfully completed: {description}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running: {description}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="RQ2 unified analysis runner")
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional output run directory. When set, all RQ2 scripts will write their "
            "analysis outputs to RUN_DIR/analyses and artifacts to RUN_DIR/artifacts via env overrides."
        ),
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")

    print("\n" + "#"*80)
    print("RQ2: Context-Insensitive Hallucination Detection")
    print("#"*80)
    
    # Check new structure first, then fall back to old
    rq2_scripts = PROJECT_ROOT / "final_runs/RQ2_uq_hallucination_detection/analysis_scripts"
    if not rq2_scripts.exists():
        rq2_scripts = PROJECT_ROOT / "final_runs/RQ2_uq_hallucination_detection/scripts"

    # Optional: coerce all sub-script outputs into a single run folder.
    # This is implemented via environment-variable overrides that the RQ2 scripts honor.
    env = os.environ.copy()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = None

    if run_dir is not None:
        analyses_dir = run_dir / "analyses"
        artifacts_dir = run_dir / "artifacts"
        analyses_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        env["RQ2_ANALYSES_DIR"] = str(analyses_dir)
        env["RQ2_OUTPUT_DIR"] = str(artifacts_dir)
        print(f"\nRQ2 output override enabled:")
        print(f"  RQ2_ANALYSES_DIR={analyses_dir}")
        print(f"  RQ2_OUTPUT_DIR={artifacts_dir}")

    # 0. Build reproducible inventory (and deduplicate)
    script_inventory = rq2_scripts / "rq2_file_inventory.py"
    run_command(f"python3 {script_inventory}", "RQ2 (Prep): Build file inventory", env=env)

    script_dedup = rq2_scripts / "rq2_deduplicate_files.py"
    run_command(f"python3 {script_dedup}", "RQ2 (Prep): Deduplicate file inventory", env=env)

    # 1. Phases 1-3: Batch Analysis
    # Generates metric statistics, correlations, and per-file diagnostics
    script_batch = rq2_scripts / "rq2_simple_batch_analyzer.py"
    run_command(f"python3 {script_batch}", "RQ2 Phases 1-3: Batch Analysis", env=env)

    # 1b. Thesis table: Single-metric meta-analysis (block-level AUC vs 0.5)
    script_single_metric = rq2_scripts / "rq2_single_metric_table.py"
    run_command(f"python3 {script_single_metric}", "RQ2 Thesis Table: Single-metric meta-analysis", env=env)

    # 2. Phase 4: RFE Feature Selection
    script_phase4 = rq2_scripts / "rq2_phase4_rfe_multiclassifier.py"
    run_command(f"python3 {script_phase4}", "RQ2 Phase 4: RFE Feature Selection", env=env)

    # 2b. Regenerate thesis-ready Phase 4 figure (uses latest Phase 4 JSON + permutation importance)
    script_phase4_fig = rq2_scripts / "generate_rfe_figure_4_features.py"
    run_command(f"python3 {script_phase4_fig}", "RQ2 Phase 4 Figure: RFE + Permutation Importance (4 features)", env=env)

    # 3. Phase 5: Ensemble Comparison
    script_phase5 = rq2_scripts / "rq2_phase5_ensemble_with_rfe_features.py"
    run_command(f"python3 {script_phase5}", "RQ2 Phase 5: Ensemble Comparison", env=env)

    # 4. Phase 6: Cross-Domain Generalization
    script_phase6 = rq2_scripts / "rq2_phase6_cross_cld_with_best.py"
    run_command(f"python3 {script_phase6}", "RQ2 Phase 6: Cross-Domain Generalization", env=env)

    # 5. Master Report Generation
    script_report = rq2_scripts / "rq2_master_report_v2_enhanced.py"
    run_command(f"python3 {script_report}", "RQ2 Master Report Generation", env=env)

    # 6. Copy figures to thesis-compatible names for reproducibility
    if run_dir is not None:
        print("\n" + "="*80)
        print("Creating thesis-compatible figure copies...")
        print("="*80)
        
        # Copy RFE figure to thesis-compatible name
        rfe_src = artifacts_dir / "rfe_4_features_figure.png"
        rfe_dst = artifacts_dir / "rfe_clean_figure.png"
        if rfe_src.exists():
            shutil.copy(rfe_src, rfe_dst)
            print(f"  ✓ Copied: rfe_4_features_figure.png → rfe_clean_figure.png")
        
        # Find and copy feature distributions figure from Phase 6
        phase6_dirs = list(analyses_dir.glob("rq2_phase6_cross_cld_*"))
        if phase6_dirs:
            latest_phase6 = max(phase6_dirs, key=lambda p: p.stat().st_mtime)
            feat_dist_src = latest_phase6 / "figures" / "phase6_feature_distributions_by_cld.png"
            feat_dist_dst = artifacts_dir / "feature_distributions_by_cld_4x3.png"
            if feat_dist_src.exists():
                shutil.copy(feat_dist_src, feat_dist_dst)
                print(f"  ✓ Copied: phase6_feature_distributions_by_cld.png → feature_distributions_by_cld_4x3.png")
        
        # Create a 'latest' symlink for the entire run
        latest_link = run_dir.parent / "RQ2_latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
        print(f"  ✓ Created symlink: RQ2_latest → {run_dir.name}")

    print("\n" + "="*80)
    print("✅ All RQ2 analyses completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()


