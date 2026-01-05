#!/usr/bin/env python3
"""
Restructure final_runs experiments to unified CLD → runs → prompts structure.

For RQ1a: Move files from run folders into prompt subfolders based on filename patterns
For RQ1b: Merge separate prompt variant folders into single experiment with prompt subfolders
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict

FINAL_RUNS = Path("/home/nitai/code/causalix.ai/final_runs")

# Prompt variants to look for
PROMPT_VARIANTS = ["baseline", "cot", "mechanistic"]

def restructure_rq1a_folder(experiment_path: Path):
    """
    Restructure RQ1a folder: move files into prompt subfolders based on filename.
    
    Current: Data/CLD/run_X/files_with_prompt_in_name
    Target:  Data/CLD/run_X/prompt/files
    """
    data_path = experiment_path / "Data"
    if not data_path.exists():
        print(f"  No Data folder in {experiment_path.name}")
        return
    
    # Find all CLD folders
    for cld_folder in data_path.iterdir():
        if not cld_folder.is_dir():
            continue
        
        # Find all run folders
        for run_folder in cld_folder.iterdir():
            if not run_folder.is_dir() or not run_folder.name.startswith("run_"):
                continue
            
            print(f"  Processing {cld_folder.name}/{run_folder.name}")
            
            # Create prompt subfolders
            for prompt in PROMPT_VARIANTS:
                (run_folder / prompt).mkdir(exist_ok=True)
            
            # Also create 'shared' for files that don't belong to a specific prompt
            (run_folder / "shared").mkdir(exist_ok=True)
            
            # Move files based on filename patterns
            for file_path in list(run_folder.iterdir()):
                if file_path.is_dir():
                    # If it's an existing subfolder (like 'analysis'), move to shared
                    if file_path.name not in PROMPT_VARIANTS and file_path.name != "shared":
                        target = run_folder / "shared" / file_path.name
                        if not target.exists():
                            shutil.move(str(file_path), str(target))
                    continue
                
                # Check filename for prompt variant
                filename = file_path.name.lower()
                moved = False
                
                for prompt in PROMPT_VARIANTS:
                    # Look for patterns like _baseline_, _cot_, _mechanistic_ in filename
                    if f"_{prompt}_" in filename or filename.startswith(f"{prompt}_"):
                        target = run_folder / prompt / file_path.name
                        shutil.move(str(file_path), str(target))
                        moved = True
                        break
                
                if not moved:
                    # Move to shared folder
                    target = run_folder / "shared" / file_path.name
                    shutil.move(str(file_path), str(target))


def merge_rq1b_variants(base_name: str, variants: list, target_name: str):
    """
    Merge multiple RQ1b variant folders into one with prompt subfolders.
    
    Current: 
      RQ1b_..._baseline/Data/CLD/run_X/files
      RQ1b_..._cot/Data/CLD/run_X/files
      RQ1b_..._mechanistic/Data/CLD/run_X/files
    
    Target:
      RQ1b_.../Data/CLD/run_X/baseline/files
      RQ1b_.../Data/CLD/run_X/cot/files
      RQ1b_.../Data/CLD/run_X/mechanistic/files
    """
    target_path = FINAL_RUNS / target_name
    target_data = target_path / "Data"
    target_data.mkdir(parents=True, exist_ok=True)
    
    # Also create other standard folders
    (target_path / "Output").mkdir(exist_ok=True)
    (target_path / "analysis_scripts").mkdir(exist_ok=True)
    
    print(f"\nMerging into {target_name}:")
    
    for variant_folder, prompt_name in variants:
        variant_path = FINAL_RUNS / variant_folder
        if not variant_path.exists():
            print(f"  Skipping {variant_folder} (not found)")
            continue
        
        print(f"  Processing {variant_folder} -> {prompt_name}/")
        
        variant_data = variant_path / "Data"
        if not variant_data.exists():
            continue
        
        # Process each CLD folder
        for cld_folder in variant_data.iterdir():
            if not cld_folder.is_dir():
                # If it's a file at Data level (like rq1b_all_results.xlsx), move to Output
                if cld_folder.suffix in ['.xlsx', '.csv', '.json']:
                    target_file = target_path / "Output" / f"{prompt_name}_{cld_folder.name}"
                    if not target_file.exists():
                        shutil.copy2(str(cld_folder), str(target_file))
                continue
            
            # Create target CLD folder
            target_cld = target_data / cld_folder.name
            target_cld.mkdir(exist_ok=True)
            
            # Process each run folder
            for run_folder in cld_folder.iterdir():
                if not run_folder.is_dir() or not run_folder.name.startswith("run_"):
                    continue
                
                # Create target run and prompt folders
                target_run = target_cld / run_folder.name
                target_run.mkdir(exist_ok=True)
                target_prompt = target_run / prompt_name
                target_prompt.mkdir(exist_ok=True)
                
                # Copy all files from source run to target prompt folder
                for item in run_folder.iterdir():
                    target_item = target_prompt / item.name
                    if item.is_dir():
                        if not target_item.exists():
                            shutil.copytree(str(item), str(target_item))
                    else:
                        if not target_item.exists():
                            shutil.copy2(str(item), str(target_item))
        
        # Copy analysis_scripts and Output folders
        for folder_name in ["analysis_scripts", "Output"]:
            src_folder = variant_path / folder_name
            if src_folder.exists():
                for item in src_folder.iterdir():
                    target_item = target_path / folder_name / f"{prompt_name}_{item.name}" if item.is_file() else target_path / folder_name / item.name
                    if item.is_file() and not target_item.exists():
                        shutil.copy2(str(item), str(target_item))


def main():
    print("=" * 60)
    print("RESTRUCTURING FINAL_RUNS TO CLD → RUNS → PROMPTS")
    print("=" * 60)
    
    # ========================================
    # RQ1a EXPERIMENTS - restructure in place
    # ========================================
    print("\n" + "=" * 40)
    print("RESTRUCTURING RQ1a EXPERIMENTS")
    print("=" * 40)
    
    rq1a_experiments = [
        "RQ1a_gt_synth_citation",
        "RQ1a_gt_synth_correctness",
        "RQ1a_gt_lit_citation",
        "RQ1a_gt_lit_correctness",
        "RQ1a_gt_lit_correctness_test",
    ]
    
    for exp_name in rq1a_experiments:
        exp_path = FINAL_RUNS / exp_name
        if exp_path.exists():
            print(f"\nRestructuring {exp_name}:")
            restructure_rq1a_folder(exp_path)
    
    # ========================================
    # RQ1b EXPERIMENTS - merge variant folders
    # ========================================
    print("\n" + "=" * 40)
    print("MERGING RQ1b EXPERIMENTS")
    print("=" * 40)
    
    # Ground truth correctness with baseline judge
    merge_rq1b_variants(
        "RQ1b_corrector_experiment_ground_truth_correctness",
        [
            ("RQ1b_corrector_experiment_ground_truth_correctness_baseline", "baseline"),
            ("RQ1b_corrector_experiment_ground_truth_correctness_cot", "cot"),
            ("RQ1b_corrector_experiment_ground_truth_correctness_mechanistic", "mechanistic"),
        ],
        "RQ1b_corrector_ground_truth_correctness"
    )
    
    # Ground truth correctness with mechanistic judge
    merge_rq1b_variants(
        "RQ1b_corrector_experiment_ground_truth_correctness_mechanistic_judge",
        [
            ("RQ1b_corrector_experiment_ground_truth_correctness_baseline_mechanistic_judge", "baseline"),
            ("RQ1b_corrector_experiment_ground_truth_correctness_cot_mechanistic_judge", "cot"),
            ("RQ1b_corrector_experiment_ground_truth_correctness_mechanistic_mechanistic_judge", "mechanistic"),
        ],
        "RQ1b_corrector_ground_truth_correctness_mechanistic_judge"
    )
    
    # Corruption detection correctness final
    merge_rq1b_variants(
        "RQ1b_corrector_experiment_corruption_detection_correctness_final",
        [
            ("RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline", "baseline"),
            ("RQ1b_corrector_experiment_corruption_detection_correctness_final_cot", "cot"),
            ("RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic", "mechanistic"),
        ],
        "RQ1b_corrector_corruption_detection_correctness"
    )
    
    print("\n" + "=" * 60)
    print("RESTRUCTURING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify the new structure is correct")
    print("2. Move old RQ1b variant folders to deprecated/")
    print("3. Commit the changes")


if __name__ == "__main__":
    main()

