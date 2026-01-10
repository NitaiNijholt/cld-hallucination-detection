#!/usr/bin/env python3
"""
Create Thesis Figure Structure

Maps reproduced outputs to the exact paths expected by the thesis LaTeX files.
Run this AFTER reproduce_all_thesis_assets.py to populate thesis/reproducible_version/Figures/final_runs/

Usage:
    python create_thesis_figure_structure.py --source /tmp/thesis_reproducibility_TIMESTAMP --target thesis/reproducible_version/Figures/final_runs
    
Or with symlinks (recommended):
    python create_thesis_figure_structure.py --source /tmp/thesis_latest --target thesis/reproducible_version/Figures/final_runs
"""

import argparse
import shutil
from pathlib import Path

# Mapping from reproduction output paths to thesis expected paths
# Format: (source_pattern, thesis_path)
# Use ** for recursive glob, * for single-level glob
FIGURE_MAPPINGS = [
    # RQ1a figures (nested in enhanced_analysis_TIMESTAMP subdirs)
    ("RQ1/RQ1a_gt_synth_correctness/**/rq1a_*.png", "RQ1a_gt_synth_correctness/enhanced_analysis_latest/"),
    ("RQ1/RQ1a_gt_lit_correctness/**/rq1a_*.png", "RQ1a_gt_lit_correctness/enhanced_analysis_latest/"),
    ("RQ1/RQ1a_gt_synth_citation/**/rq1a_*.png", "RQ1a_gt_synth_citation/enhanced_analysis_latest/"),
    ("RQ1/RQ1a_gt_lit_citation/**/rq1a_*.png", "RQ1a_gt_lit_citation/enhanced_analysis_latest/"),
    
    # RQ1b corrector ablation tables
    ("RQ1/RQ1b_corrector_ablation/*.tex", "RQ1b_corrector_ablation/"),
    
    # RQ2 figures and tables (in artifacts subdir)
    ("RQ2/artifacts/phase4_rfe_complete.png", "RQ2_uq_hallucination_detection/phase4_rfe_complete.png"),
    ("RQ2/analyses/**/phase6_distributions_with_effect_sizes.png", "RQ2_uq_hallucination_detection/phase6_distributions_with_effect_sizes.png"),
    ("RQ2/artifacts/*.tex", "RQ2_uq_hallucination_detection/"),
    ("RQ2/artifacts/*.png", "RQ2_uq_hallucination_detection/"),
    
    # RQ3 figures and tables
    ("RQ3/figures/*.png", "RQ3_deep_research_validation/Output/"),
    ("RQ3/rq3_combined_figure.png", "RQ3_deep_research_validation/Output/rq3_combined_figure.png"),
    ("RQ3/validation/*.png", "RQ3_deep_research_validation/validation/"),
    ("RQ3/tables/*.tex", "RQ3_deep_research_validation/"),
    ("RQ3/*.tex", "RQ3_deep_research_validation/"),
    
    # SUPP time/cost
    ("SUPP/figure3_generation_vs_judging.png", "supp_time_cost_scaling/Output/figure3_generation_vs_judging.png"),
    ("SUPP/time_cost_scaling/figures/*.png", "supp_time_cost_scaling/Output/"),
    ("SUPP/*.tex", "latex/"),
    
    # SUPP parallelization
    ("SUPP/parallelization_thesis_figure.png", "supp_parallelization_benchmark/Output/parallelization_thesis_figure.png"),
    ("SUPP/parallelization/*.png", "supp_parallelization_benchmark/Output/"),
    ("SUPP/parallelization/*.tex", "supp_parallelization_benchmark/"),
    
    # SUPP prompt sensitivity
    ("SUPP/prompt_sensitivity_figure.png", "supp_prompt_sensitivity/prompt_sensitivity_figure.png"),
    ("SUPP/corrector_sensitivity_figure.png", "supp_prompt_sensitivity/corrector_sensitivity_figure.png"),
    ("SUPP/*sensitivity*.tex", "supp_prompt_sensitivity/"),
    
    # SUPP edge ablation
    ("SUPP/edge_ablation_f1_ordered.png", "supp_edge_ablation/edge_ablation_f1_ordered.png"),
    
    # PRELIM
    ("PRELIM/random_baseline_comparison.png", "prelim_random_baseline_generator/Output/random_baseline_comparison.png"),
    ("PRELIM/random_baseline_table.tex", "prelim_random_baseline_generator/Output/random_baseline_table.tex"),
    ("PRELIM/temperature_sensitivity/*.png", "prelim_temperature_sensitivity/Output/"),
    ("PRELIM/temperature_sensitivity/*.tex", "prelim_temperature_sensitivity/Output/"),
    ("PRELIM/*temperature*.tex", "prelim_temperature_sensitivity/Output/"),
    ("PRELIM/generator_model_comparison.tex", "prelim_generator_model_comparison/Output/generator_model_comparison.tex"),
    ("PRELIM/generator_model_comparison.png", "prelim_generator_model_comparison/Output/generator_model_comparison.png"),
    
    # VALIDATION
    ("VALIDATION/truthqa_verification_table.tex", "validation_RQ1_truthfulqa_judge/truthqa_verification_table.tex"),
    ("VALIDATION/human_validation_table.tex", "validation_RQ1a_human_agreement/human_validation_table.tex"),
    ("VALIDATION/human_validation_confusion_table.tex", "validation_RQ1a_human_agreement/human_validation_confusion_table.tex"),
    ("VALIDATION/search_provider_comparison_table.tex", "validation_RQ1a_ulemans_external/search_provider_comparison_table.tex"),
    ("VALIDATION/physics_comparison_table.tex", "validation_RQ1a_physics_clds/physics_comparison_table.tex"),
    ("VALIDATION/retrieval_comparison_table.tex", "validation_RQ1a_physics_clds/retrieval_comparison_table.tex"),
    ("VALIDATION/effect_size_summary_table.tex", "effect_size_summary_table.tex"),
]


def copy_with_glob(source_dir: Path, pattern: str, target_dir: Path, target_subpath: str):
    """Copy files matching pattern from source to target."""
    # Handle recursive glob patterns (**)
    if '**' in pattern:
        # Use full pattern from source_dir
        files = list(source_dir.glob(pattern))
    else:
        source_path = source_dir / pattern.rsplit('/', 1)[0] if '/' in pattern else source_dir
        file_pattern = pattern.rsplit('/', 1)[-1]
        
        # Handle glob patterns
        if '*' in file_pattern:
            files = list(source_path.glob(file_pattern))
        else:
            files = [source_path / file_pattern] if (source_path / file_pattern).exists() else []
    
    copied = 0
    for src_file in files:
        if src_file.is_file():
            # Determine target path
            if target_subpath.endswith('/'):
                # Directory target - keep original filename
                dst = target_dir / target_subpath / src_file.name
            else:
                # Specific file target
                dst = target_dir / target_subpath
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)
            copied += 1
            print(f"  ✓ {src_file.name} → {dst.relative_to(target_dir)}")
    
    return copied


def main():
    parser = argparse.ArgumentParser(description="Map reproduction outputs to thesis figure structure")
    parser.add_argument("--source", required=True, help="Path to reproduction output (e.g., /tmp/thesis_latest)")
    parser.add_argument("--target", required=True, help="Path to thesis Figures/final_runs directory")
    args = parser.parse_args()
    
    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    
    if not source.exists():
        print(f"❌ Source directory not found: {source}")
        return 1
    
    print("=" * 80)
    print("CREATE THESIS FIGURE STRUCTURE")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print()
    
    # NOTE: We do NOT clear the target directory to preserve pre-populated static files
    # (e.g., latex/table21_scaling_summary.tex, supp_prompt_sensitivity/*.tex)
    # The copy operation will overwrite any existing files with the same name
    
    target.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    for pattern, thesis_path in FIGURE_MAPPINGS:
        copied = copy_with_glob(source, pattern, target, thesis_path)
        total_copied += copied
    
    print()
    print("=" * 80)
    print(f"✅ Copied {total_copied} files to thesis figure structure")
    print("=" * 80)
    
    # Verify key files exist
    key_files = [
        "RQ1a_gt_synth_correctness/enhanced_analysis_latest/rq1a_row1_performance_metrics.png",
        "RQ2_uq_hallucination_detection/phase4_rfe_complete.png",
        "RQ3_deep_research_validation/Output/rq3_combined_figure.png",
        "supp_time_cost_scaling/Output/figure3_generation_vs_judging.png",
        "prelim_random_baseline_generator/Output/random_baseline_comparison.png",
    ]
    
    print()
    print("Key files verification:")
    for kf in key_files:
        exists = (target / kf).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {kf}")
    
    return 0


if __name__ == "__main__":
    exit(main())
