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
    # Also copy phase6_distributions to root for lowercase figures/ reference in thesis
    ("RQ2/artifacts/phase6_distributions_with_effect_sizes.png", "phase6_distributions_with_effect_sizes.png"),
    
    # RQ3 figures and tables
    ("RQ3/figures/*.png", "RQ3_deep_research_validation/Output/"),
    ("RQ3/rq3_combined_figure.png", "RQ3_deep_research_validation/Output/rq3_combined_figure.png"),
    ("RQ3/validation/*.png", "RQ3_deep_research_validation/validation/"),
    ("RQ3/tables/*.tex", "RQ3_deep_research_validation/"),
    ("RQ3/*.tex", "RQ3_deep_research_validation/"),
    
    # SUPP time/cost
    ("SUPP/figure3_generation_vs_judging.png", "supp_time_cost_scaling/Output/figure3_generation_vs_judging.png"),
    ("SUPP/time_cost_scaling/figures/*.png", "supp_time_cost_scaling/Output/"),
    ("SUPP/table21_scaling_summary.tex", "latex/table21_scaling_summary.tex"),
    # Also copy figure1/2 to root for direct Figures/ reference in appendix
    ("SUPP/time_cost_scaling/figures/figure1_scaling_analysis.png", "figure1_scaling_analysis.png"),
    ("SUPP/time_cost_scaling/figures/figure2_configuration_comparison.png", "figure2_configuration_comparison.png"),
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

# Static assets that are copied from the main thesis (illustrations, not generated)
STATIC_ASSETS = [
    # Introduction illustration (hand-created PDF)
    ("thesis/reproducible_version/Figures/figure4_expert_time_vs_llm_generation.pdf", "figure4_expert_time_vs_llm_generation.pdf"),
]

# Figures that need to be copied DIRECTLY to thesis/reproducible_version/Figures/
# (because thesis uses lowercase 'figures/' path which resolves there)
THESIS_FIGURES_MAPPINGS = [
    # From SUPP - time/cost scaling
    ("SUPP/time_cost_scaling/figures/figure1_scaling_analysis.png", "figure1_scaling_analysis.png"),
    ("SUPP/time_cost_scaling/figures/figure2_configuration_comparison.png", "figure2_configuration_comparison.png"),
    ("SUPP/figure3_generation_vs_judging.png", "figure3_generation_vs_judging.png"),
    # From SUPP - sensitivity
    ("SUPP/prompt_sensitivity_figure.png", "prompt_sensitivity_figure.png"),
    ("SUPP/corrector_sensitivity_figure.png", "corrector_sensitivity_figure.png"),
    # From RQ2 (phase6 distributions)
    ("RQ2/analyses/**/phase6_correct_vs_hallucination_distributions.png", "phase6_distributions_with_effect_sizes.png"),
    # From RQ3
    ("RQ3/validation/dr_cost_efficiency_comparison.png", "dr_cost_efficiency_comparison.png"),
    # From PRELIM
    ("PRELIM/generator_model_comparison.png", "generator_model_comparison.png"),
    ("PRELIM/generator_precision_recall.png", "generator_precision_recall.png"),
    ("PRELIM/random_baseline_comparison.png", "random_baseline_comparison.png"),
]

# Static figures that can't be regenerated (require private data or are hand-made)
# These are copied from their existing locations in final_runs/
STATIC_FIGURES = [
    ("supp_edge_ablation/edge_ablation_f1_ordered.png", "edge_ablation_f1_ordered.png"),
]

# Generated tables that go to thesis/reproducible_version/generated/ (not Figures/final_runs/)
# These are LaTeX macros and tables generated by RQ3 scripts
# Format: (source_pattern, destination_name, [alternative_glob_patterns])
GENERATED_TABLE_MAPPINGS = [
    # Smart trigger tables from RQ3
    ("RQ3/validation/rq3_smart_trigger_numbers.tex", "rq3_smart_trigger_numbers.tex"),
    ("RQ3/validation/rq3_smart_trigger_table.tex", "rq3_smart_trigger_table.tex"),
    # Human validation tables
    ("RQ3/validation/rq3_human_validation_numbers.tex", "rq3_human_validation_numbers.tex"),
    ("RQ3/validation/rq3_human_validation_table.tex", "rq3_human_validation_table.tex"),
    # Enrichment validation tables  
    ("RQ3/validation/rq3_enrichment_validation_numbers.tex", "rq3_enrichment_validation_numbers.tex"),
    ("RQ3/validation/rq3_enrichment_validation_table.tex", "rq3_enrichment_validation_table.tex"),
    # Distribution tables (may have timestamps)
    ("RQ3/validation/rq3_verdict_distribution_table.tex", "rq3_verdict_distribution_table.tex"),
    ("RQ3/validation/rq3_applicability_distribution_table.tex", "rq3_applicability_distribution_table.tex"),
    # Effect size summary
    ("VALIDATION/effect_size_summary_table.tex", "effect_size_summary_table.tex"),
    ("VALIDATION/effect_size_summary_rq3_enrichment_rows.tex", "effect_size_summary_rq3_enrichment_rows.tex"),
]

# Alternative paths to check for tables (in order of preference)
TABLE_ALTERNATIVE_PATHS = {
    "rq3_verdict_distribution_table.tex": ["RQ3/tables/verdict_distribution_table_*.tex", "RQ3/validation/verdict_distribution_table_*.tex"],
    "rq3_applicability_distribution_table.tex": ["RQ3/tables/applicability_distribution_table_*.tex", "RQ3/validation/applicability_distribution_table_*.tex"],
    "rq3_smart_trigger_numbers.tex": ["RQ3/**/rq3_smart_trigger_numbers.tex"],
    "rq3_smart_trigger_table.tex": ["RQ3/**/rq3_smart_trigger_table.tex"],
    "rq3_human_validation_numbers.tex": ["RQ3/**/rq3_human_validation_numbers.tex"],
    "rq3_human_validation_table.tex": ["RQ3/**/rq3_human_validation_table.tex"],
    "rq3_enrichment_validation_numbers.tex": ["RQ3/**/rq3_enrichment_validation_numbers.tex"],
    "rq3_enrichment_validation_table.tex": ["RQ3/**/rq3_enrichment_validation_table.tex"],
    "effect_size_summary_table.tex": ["VALIDATION/**/effect_size_summary_table.tex", "**/effect_size_summary_table.tex"],
    "effect_size_summary_rq3_enrichment_rows.tex": ["VALIDATION/**/effect_size_summary_rq3_enrichment_rows.tex"],
}


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
    
    # Copy static assets from the main thesis (illustrations that aren't generated)
    # Find project root by going up from target directory until we find thesis/
    project_root = target.parent.parent.parent.parent  # Figures/final_runs -> Figures -> reproducible_version -> thesis -> project_root
    if not (project_root / "thesis").exists():
        # Fallback: try from source
        project_root = source.parent
    if (project_root / "thesis").exists():
        print("\nCopying static assets (illustrations):")
        for static_src, static_dst in STATIC_ASSETS:
            src_path = project_root / static_src
            if src_path.exists():
                dst_path = target / static_dst
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ {static_src} → {static_dst}")
                total_copied += 1
    
    # Copy key figures to parent Figures/ directory for direct Figures/xxx references in thesis
    # (Thesis uses lowercase 'figures/' path which resolves to thesis/reproducible_version/Figures/)
    figures_parent = target.parent  # Figures/ (parent of final_runs/)
    print("\nCopying to Figures/ for direct references:")
    for src_pattern, dst_name in THESIS_FIGURES_MAPPINGS:
        # Handle glob patterns
        if '**' in src_pattern or '*' in src_pattern:
            matches = list(source.glob(src_pattern))
            if matches:
                src_file = sorted(matches)[-1]  # Use most recent
                dst_path = figures_parent / dst_name
                shutil.copy2(src_file, dst_path)
                print(f"  ✓ {src_file.name} → Figures/{dst_name}")
                total_copied += 1
        else:
            src_path = source / src_pattern
            if src_path.exists():
                dst_path = figures_parent / dst_name
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ {src_pattern.split('/')[-1]} → Figures/{dst_name}")
                total_copied += 1
            else:
                print(f"  ⚠ NOT FOUND: {src_pattern}")
    
    # Also copy static figure4 to Figures/
    for static_src, static_dst in STATIC_ASSETS:
        src_path = target / static_dst
        if src_path.exists():
            dst_path = figures_parent / static_dst
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ {static_dst} → Figures/{static_dst}")
            total_copied += 1
    
    # Copy static figures from final_runs/ (figures that can't be regenerated)
    print("\nCopying static figures from final_runs/:")
    for src_rel, dst_name in STATIC_FIGURES:
        src_path = target / src_rel
        if src_path.exists() and src_path.stat().st_size > 10000:  # Must be real, not placeholder
            dst_path = figures_parent / dst_name
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ {src_rel} → Figures/{dst_name}")
            total_copied += 1
        else:
            # Check if there's a real version in thesis/Figures already
            existing = figures_parent / dst_name
            if existing.exists() and existing.stat().st_size > 10000:
                print(f"  ✓ {dst_name} already present ({existing.stat().st_size} bytes)")
            else:
                print(f"  ⚠ {dst_name}: no real version found (static asset, requires private data)")
    
    # Copy generated tables to thesis/reproducible_version/generated/
    generated_dir = target.parent.parent / "generated"  # Figures/final_runs -> Figures -> reproducible_version -> generated
    generated_dir.mkdir(parents=True, exist_ok=True)
    print("\nCopying to generated/ for LaTeX macros and tables:")
    for src_pattern, dst_name in GENERATED_TABLE_MAPPINGS:
        src_path = source / src_pattern
        copied = False
        if src_path.exists():
            dst_path = generated_dir / dst_name
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ {src_pattern} → generated/{dst_name}")
            total_copied += 1
            copied = True
        
        # Check glob patterns for timestamped or nested files
        if not copied and dst_name in TABLE_ALTERNATIVE_PATHS:
            for glob_pattern in TABLE_ALTERNATIVE_PATHS[dst_name]:
                matches = list(source.glob(glob_pattern))
                if matches:
                    # Use the most recent file (last in sorted order)
                    src_file = sorted(matches)[-1]
                    shutil.copy2(src_file, generated_dir / dst_name)
                    print(f"  ✓ {src_file.name} → generated/{dst_name} (from glob)")
                    total_copied += 1
                    copied = True
                    break
        
        if not copied:
            # Check reproduction output directories
            alt_paths = [
                source / "RQ3" / dst_name,
                source / "RQ3" / "validation" / dst_name,
                source / "RQ3" / "tables" / dst_name,
                source / "SUPP" / dst_name,
                source / "VALIDATION" / dst_name,
            ]
            for alt in alt_paths:
                if alt.exists():
                    shutil.copy2(alt, generated_dir / dst_name)
                    print(f"  ✓ {alt.name} → generated/{dst_name} (from reproduction output)")
                    total_copied += 1
                    copied = True
                    break
        
        if not copied:
            # Check if file already exists in generated/ with real content (written directly by scripts)
            existing = generated_dir / dst_name
            if existing.exists() and existing.stat().st_size > 200:  # Not a placeholder
                content = existing.read_text()
                if "Placeholder" not in content and "[Table:" not in content:
                    print(f"  ✓ {dst_name} already exists with content (written directly)")
                    total_copied += 1
                    copied = True
            if not copied:
                print(f"  ⚠️ Not found in reproduction output: {dst_name}")
    
    # Also scan for any additional .tex files in key output directories and copy them
    print("\nScanning for additional .tex tables in reproduction output:")
    additional_tex_dirs = [
        (source / "RQ3" / "tables", "rq3_"),
        (source / "RQ3" / "validation", "rq3_"),
        (source / "VALIDATION", ""),
    ]
    for scan_dir, prefix in additional_tex_dirs:
        if scan_dir.exists():
            for tex_file in scan_dir.glob("*.tex"):
                # Skip timestamped duplicates, only copy the most recent version
                base_name = tex_file.stem
                # Check if it's a timestamped file (ends with _YYYYMMDD_HHMMSS)
                import re
                timestamp_match = re.match(r"(.+)_\d{8}_\d{6}$", base_name)
                if timestamp_match:
                    base_name = timestamp_match.group(1)
                dst_name = f"{prefix}{base_name}.tex" if prefix and not base_name.startswith(prefix) else f"{base_name}.tex"
                dst_path = generated_dir / dst_name
                if not dst_path.exists():
                    shutil.copy2(tex_file, dst_path)
                    print(f"  ✓ {tex_file.name} → generated/{dst_name}")
                    total_copied += 1
    
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
