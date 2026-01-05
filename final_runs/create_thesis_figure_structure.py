#!/usr/bin/env python3
"""
Create Thesis Figure Structure
==============================

Maps reproduced outputs to the exact paths expected by the thesis LaTeX files.
Creates the `Figures/final_runs/` directory structure with proper timestamps/symlinks.

Usage:
    # From reproduction output, create thesis-compatible structure
    uv run python final_runs/create_thesis_figure_structure.py \
        --source /tmp/thesis_repro_20260104 \
        --target thesis/final_thesis/def_submission_template/Figures/final_runs

    # Or create in a new directory for review
    uv run python final_runs/create_thesis_figure_structure.py \
        --source /tmp/thesis_repro_20260104 \
        --target /tmp/thesis_figures_for_review
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


# Mapping from thesis expected paths to reproduction output patterns
# Format: (thesis_relative_path, (repro_category, repro_subdir_pattern, filename))
THESIS_FIGURE_MAPPING = [
    # RQ1a GT Synth Correctness
    ("RQ1a_gt_synth_correctness/Output/enhanced_analysis_latest/rq1a_row1_performance_metrics.png",
     ("RQ1", "RQ1a_gt_synth_correctness/enhanced_analysis_*", "rq1a_row1_performance_metrics.png")),
    ("RQ1a_gt_synth_correctness/Output/enhanced_analysis_latest/rq1a_row2_detailed_analysis.png",
     ("RQ1", "RQ1a_gt_synth_correctness/enhanced_analysis_*", "rq1a_row2_detailed_analysis.png")),
    ("RQ1a_gt_synth_correctness/Output/enhanced_analysis_latest/rq1a_roc_curves.png",
     ("RQ1", "RQ1a_gt_synth_correctness/enhanced_analysis_*", "rq1a_roc_curves.png")),
    
    # RQ1a GT Synth Citation
    ("RQ1a_gt_synth_citation/Output/enhanced_analysis_latest/rq1a_row1_performance_metrics.png",
     ("RQ1", "RQ1a_gt_synth_citation/enhanced_analysis_*", "rq1a_row1_performance_metrics.png")),
    ("RQ1a_gt_synth_citation/Output/enhanced_analysis_latest/rq1a_row2_detailed_analysis.png",
     ("RQ1", "RQ1a_gt_synth_citation/enhanced_analysis_*", "rq1a_row2_detailed_analysis.png")),
    ("RQ1a_gt_synth_citation/Output/enhanced_analysis_latest/rq1a_roc_curves.png",
     ("RQ1", "RQ1a_gt_synth_citation/enhanced_analysis_*", "rq1a_roc_curves.png")),
    
    # RQ1a GT Lit Correctness
    ("RQ1a_gt_lit_correctness/Output/enhanced_analysis_latest/rq1a_ground_truth_row1_performance_metrics.png",
     ("RQ1", "RQ1a_gt_lit_correctness/enhanced_analysis_*", "rq1a_ground_truth_row1_performance_metrics.png")),
    ("RQ1a_gt_lit_correctness/Output/enhanced_analysis_latest/rq1a_ground_truth_row2_detailed_analysis.png",
     ("RQ1", "RQ1a_gt_lit_correctness/enhanced_analysis_*", "rq1a_ground_truth_row2_detailed_analysis.png")),
    ("RQ1a_gt_lit_correctness/Output/enhanced_analysis_latest/rq1a_ground_truth_score_heatmap.png",
     ("RQ1", "RQ1a_gt_lit_correctness/enhanced_analysis_*", "rq1a_ground_truth_correctness_score_heatmap.png")),
    
    # RQ1a GT Lit Citation
    ("RQ1a_gt_lit_citation/Output/enhanced_analysis_latest/rq1a_ground_truth_row1_performance_metrics.png",
     ("RQ1", "RQ1a_gt_lit_citation/enhanced_analysis_*", "rq1a_ground_truth_row1_performance_metrics.png")),
    ("RQ1a_gt_lit_citation/Output/enhanced_analysis_latest/rq1a_ground_truth_row2_detailed_analysis.png",
     ("RQ1", "RQ1a_gt_lit_citation/enhanced_analysis_*", "rq1a_ground_truth_row2_detailed_analysis.png")),
    ("RQ1a_gt_lit_citation/Output/enhanced_analysis_latest/rq1a_ground_truth_score_heatmap.png",
     ("RQ1", "RQ1a_gt_lit_citation/enhanced_analysis_*", "rq1a_ground_truth_citation_score_heatmap.png")),
    
    # Appendix: Legacy directory names (same data, different naming)
    # RQ1a_corruption_detection_correctness_final -> maps to gt_synth_correctness
    ("RQ1a_corruption_detection_correctness_final/Output/enhanced_analysis_latest/rq1a_corruption_detection_correctness_score_heatmap.png",
     ("RQ1", "RQ1a_gt_synth_correctness/enhanced_analysis_*", "rq1a_corruption_detection_correctness_score_heatmap.png")),
    ("RQ1a_corruption_detection_correctness_final/Output/enhanced_analysis_latest/rq1a_roc_curves.png",
     ("RQ1", "RQ1a_gt_synth_correctness/enhanced_analysis_*", "rq1a_roc_curves.png")),
    
    # RQ1a_corruption_detection_citation_gpt5mini -> maps to gt_synth_citation
    ("RQ1a_corruption_detection_citation_gpt5mini/Output/enhanced_analysis_latest/rq1a_corruption_detection_citation_score_heatmap.png",
     ("RQ1", "RQ1a_gt_synth_citation/enhanced_analysis_*", "rq1a_corruption_detection_citation_score_heatmap.png")),
    ("RQ1a_corruption_detection_citation_gpt5mini/Output/enhanced_analysis_latest/rq1a_roc_curves.png",
     ("RQ1", "RQ1a_gt_synth_citation/enhanced_analysis_*", "rq1a_roc_curves.png")),
    
    # RQ1a_ground_truth_correctness -> maps to gt_lit_correctness
    ("RQ1a_ground_truth_correctness/Output/enhanced_analysis_latest/rq1a_ground_truth_correctness_score_heatmap.png",
     ("RQ1", "RQ1a_gt_lit_correctness/enhanced_analysis_*", "rq1a_ground_truth_correctness_score_heatmap.png")),
    ("RQ1a_ground_truth_correctness/Output/enhanced_analysis_latest/rq1a_ground_truth_roc_curves.png",
     ("RQ1", "RQ1a_gt_lit_correctness/enhanced_analysis_*", "rq1a_ground_truth_roc_curves.png")),
    
    # RQ1a_ground_truth_citation -> maps to gt_lit_citation
    ("RQ1a_ground_truth_citation/Output/enhanced_analysis_latest/rq1a_ground_truth_citation_score_heatmap.png",
     ("RQ1", "RQ1a_gt_lit_citation/enhanced_analysis_*", "rq1a_ground_truth_citation_score_heatmap.png")),
    ("RQ1a_ground_truth_citation/Output/enhanced_analysis_latest/rq1a_ground_truth_roc_curves.png",
     ("RQ1", "RQ1a_gt_lit_citation/enhanced_analysis_*", "rq1a_ground_truth_roc_curves.png")),
    
    # RQ2 UQ Hallucination Detection
    ("RQ2_uq_hallucination_detection/rfe_clean_figure.png",
     ("RQ2", "artifacts", "rfe_clean_figure.png")),
    ("RQ2_uq_hallucination_detection/feature_distributions_by_cld_4x3.png",
     ("RQ2", "artifacts", "feature_distributions_by_cld_4x3.png")),
    
    # RQ3 Deep Research Validation
    ("RQ3_deep_research_validation/Output/rq3_combined_figure.png",
     ("RQ3", "figures", "rq3_combined_figure.png")),
    ("RQ3_deep_research_validation/validation/all_edges_threshold_tradeoff.png",
     ("RQ3", "validation", "all_edges_threshold_tradeoff.png")),
    ("RQ3_deep_research_validation/validation/dr_cost_efficiency_comparison.png",
     ("RQ3", "validation", "dr_cost_efficiency_comparison.png")),
    
    # Supplementary
    ("supp_parallelization_benchmark/Output/parallelization_thesis_figure.png",
     ("SUPP", ".", "parallelization_thesis_figure.png")),
    ("supp_time_cost_scaling/Output/figure3_generation_vs_judging.png",
     ("SUPP", ".", "figure3_generation_vs_judging.png")),
    
    # Prompt Sensitivity (Appendix)
    ("supp_prompt_sensitivity/prompt_sensitivity_figure.png",
     ("SUPP", ".", "prompt_sensitivity_figure.png")),
    ("supp_prompt_sensitivity/corrector_sensitivity_figure.png",
     ("SUPP", ".", "corrector_sensitivity_figure.png")),
]

# Tables that need to be copied from source final_runs (not reproduced, static)
THESIS_TABLE_MAPPING = [
    # Prompt sensitivity tables
    ("supp_prompt_sensitivity/prompt_sensitivity_table.tex",
     "supp_prompt_sensitivity/prompt_sensitivity_table.tex"),
    ("supp_prompt_sensitivity/prompt_sensitivity_assumptions_table.tex",
     "supp_prompt_sensitivity/prompt_sensitivity_assumptions_table.tex"),
    ("supp_prompt_sensitivity/corrector_sensitivity_table.tex",
     "supp_prompt_sensitivity/corrector_sensitivity_table.tex"),
    ("supp_prompt_sensitivity/corrector_sensitivity_assumptions_table.tex",
     "supp_prompt_sensitivity/corrector_sensitivity_assumptions_table.tex"),
    
    # RQ1b corrector tables
    ("RQ1b_corrector_ablation/baseline_table_groundtruth.tex",
     "RQ1b_corrector_ablation/baseline_table_groundtruth.tex"),
    ("RQ1b_corrector_ablation/baseline_table_synthetic.tex",
     "RQ1b_corrector_ablation/baseline_table_synthetic.tex"),
    ("RQ1b_corrector_ablation/ablation_table_groundtruth.tex",
     "RQ1b_corrector_ablation/ablation_table_groundtruth.tex"),
    ("RQ1b_corrector_ablation/ablation_table_synthetic.tex",
     "RQ1b_corrector_ablation/ablation_table_synthetic.tex"),
    
    # RQ2 UQ tables
    ("RQ2_uq_hallucination_detection/single_metric_table.tex",
     "RQ2_uq_hallucination_detection/single_metric_table.tex"),
    ("RQ2_uq_hallucination_detection/ensemble_performance_table.tex",
     "RQ2_uq_hallucination_detection/ensemble_performance_table.tex"),
    ("RQ2_uq_hallucination_detection/normality_tests_table.tex",
     "RQ2_uq_hallucination_detection/normality_tests_table.tex"),
    
    # RQ3 tables
    ("RQ3_deep_research_validation/rq3_main_table.tex",
     "RQ3_deep_research_validation/rq3_main_table.tex"),
    ("RQ3_deep_research_validation/rq3_per_cld_table.tex",
     "RQ3_deep_research_validation/rq3_per_cld_table.tex"),
    ("RQ3_deep_research_validation/rq3_enriched_gt_table.tex",
     "RQ3_deep_research_validation/rq3_enriched_gt_table.tex"),
    
    # Scaling table
    ("latex/table21_scaling_summary.tex",
     "latex/table21_scaling_summary.tex"),
    
    # Static figures (preliminary, not reproduced)
    ("prelim_random_baseline_generator/Output/random_baseline_comparison.png",
     "prelim_random_baseline_generator/Output/random_baseline_comparison.png"),
    ("RQ2_uq_hallucination_detection/phase4_rfe_complete.png",
     "RQ2_uq_hallucination_detection/phase4_rfe_complete.png"),
]


def find_source_file(source_dir: Path, category: str, subdir_pattern: str, filename: str) -> Path | None:
    """Find a source file matching the pattern."""
    category_dir = source_dir / category
    if not category_dir.exists():
        return None
    
    # Handle glob patterns in subdir
    if "*" in subdir_pattern:
        # Find matching directories
        parts = subdir_pattern.split("*")
        matching_dirs = list(category_dir.glob(subdir_pattern))
        if matching_dirs:
            # Use the most recent one
            latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
            candidate = latest_dir / filename
            if candidate.exists():
                return candidate
    else:
        candidate = category_dir / subdir_pattern / filename
        if candidate.exists():
            return candidate
    
    # Try recursive search as fallback
    matches = list(category_dir.rglob(filename))
    if matches:
        return matches[0]
    
    return None


def create_thesis_structure(source_dir: Path, target_dir: Path, final_runs_dir: Path | None = None, dry_run: bool = False):
    """Create the thesis-compatible figure structure."""
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    if final_runs_dir:
        print(f"Static tables from: {final_runs_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'COPY'}")
    print()
    
    success_count = 0
    missing_count = 0
    
    # Copy figures from reproduction output
    for thesis_path, (category, subdir_pattern, filename) in THESIS_FIGURE_MAPPING:
        source_file = find_source_file(source_dir, category, subdir_pattern, filename)
        target_file = target_dir / thesis_path
        
        if source_file:
            if not dry_run:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source_file, target_file)
            print(f"✓ {thesis_path}")
            print(f"  ← {source_file.relative_to(source_dir)}")
            success_count += 1
        else:
            print(f"✗ {thesis_path}")
            print(f"  NOT FOUND: {category}/{subdir_pattern}/{filename}")
            missing_count += 1
    
    # Copy static tables from original final_runs (if provided)
    if final_runs_dir and THESIS_TABLE_MAPPING:
        print("\n--- Static Tables ---")
        for thesis_path, source_path in THESIS_TABLE_MAPPING:
            source_file = final_runs_dir / source_path
            target_file = target_dir / thesis_path
            
            if source_file.exists():
                if not dry_run:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(source_file, target_file)
                print(f"✓ {thesis_path}")
                print(f"  ← {source_path}")
                success_count += 1
            else:
                print(f"✗ {thesis_path}")
                print(f"  NOT FOUND: {source_path}")
                missing_count += 1
    
    total = len(THESIS_FIGURE_MAPPING) + (len(THESIS_TABLE_MAPPING) if final_runs_dir else 0)
    print()
    print("="*60)
    print(f"Copied: {success_count}/{total}")
    print(f"Missing: {missing_count}/{total}")
    
    if missing_count > 0:
        print("\n⚠️ Some assets could not be mapped. Check the source paths.")
    else:
        print("\n✅ All thesis assets mapped successfully!")
    
    return success_count, missing_count


def main():
    parser = argparse.ArgumentParser(description="Create thesis-compatible figure structure")
    parser.add_argument("--source", required=True, help="Source reproduction directory")
    parser.add_argument("--target", required=True, help="Target thesis Figures/final_runs directory")
    parser.add_argument("--final-runs", default=None, help="Original final_runs directory for static tables (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    args = parser.parse_args()
    
    source_dir = Path(args.source).expanduser().resolve()
    target_dir = Path(args.target).expanduser().resolve()
    
    # Default final_runs location
    if args.final_runs:
        final_runs_dir = Path(args.final_runs).expanduser().resolve()
    else:
        # Auto-detect from script location
        final_runs_dir = Path(__file__).parent.resolve()
    
    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1
    
    create_thesis_structure(source_dir, target_dir, final_runs_dir=final_runs_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    exit(main())

