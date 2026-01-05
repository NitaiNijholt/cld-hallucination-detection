#!/usr/bin/env python3
"""
Add judge level to RQ1b_corrector_ground_truth_correctness structure.

Current:  Data/CLD/run_X/prompt/files
Target:   Data/CLD/run_X/prompt/judge/files
"""

import os
import shutil
from pathlib import Path

FINAL_RUNS = Path("/home/nitai/code/causalix.ai/final_runs")
PROMPTS = ["baseline", "cot", "mechanistic"]

def add_judge_level_to_folder(experiment_path: Path, judge_name: str):
    """Add judge subfolder level to existing prompt folders."""
    data_path = experiment_path / "Data"
    if not data_path.exists():
        print(f"  No Data folder in {experiment_path.name}")
        return
    
    for cld_folder in data_path.iterdir():
        if not cld_folder.is_dir():
            continue
        
        for run_folder in cld_folder.iterdir():
            if not run_folder.is_dir() or not run_folder.name.startswith("run_"):
                continue
            
            print(f"  Processing {cld_folder.name}/{run_folder.name}")
            
            for prompt in PROMPTS:
                prompt_folder = run_folder / prompt
                if not prompt_folder.exists():
                    continue
                
                # Create temp folder to hold files
                temp_folder = run_folder / f"_temp_{prompt}"
                
                # Move all contents to temp
                if prompt_folder.exists() and any(prompt_folder.iterdir()):
                    shutil.move(str(prompt_folder), str(temp_folder))
                    
                    # Recreate prompt folder and add judge subfolder
                    prompt_folder.mkdir(exist_ok=True)
                    judge_folder = prompt_folder / judge_name
                    
                    # Move contents from temp to judge folder
                    shutil.move(str(temp_folder), str(judge_folder))


def merge_mechanistic_judge(source_path: Path, target_path: Path):
    """Merge mechanistic judge data into main experiment folder."""
    source_data = source_path / "Data"
    target_data = target_path / "Data"
    
    if not source_data.exists():
        print(f"  No Data folder in {source_path.name}")
        return
    
    for cld_folder in source_data.iterdir():
        if not cld_folder.is_dir():
            continue
        
        target_cld = target_data / cld_folder.name
        target_cld.mkdir(exist_ok=True)
        
        for run_folder in cld_folder.iterdir():
            if not run_folder.is_dir() or not run_folder.name.startswith("run_"):
                continue
            
            print(f"  Merging {cld_folder.name}/{run_folder.name}")
            
            target_run = target_cld / run_folder.name
            target_run.mkdir(exist_ok=True)
            
            for prompt in PROMPTS:
                prompt_folder = run_folder / prompt
                if not prompt_folder.exists():
                    continue
                
                target_prompt = target_run / prompt
                target_prompt.mkdir(exist_ok=True)
                
                # Create mechanistic_judge folder and copy contents
                target_judge = target_prompt / "mechanistic_judge"
                if not target_judge.exists():
                    shutil.copytree(str(prompt_folder), str(target_judge))


def main():
    print("=" * 60)
    print("ADDING JUDGE LEVEL TO STRUCTURE")
    print("=" * 60)
    
    main_exp = FINAL_RUNS / "RQ1b_corrector_ground_truth_correctness"
    mech_judge_exp = FINAL_RUNS / "RQ1b_corrector_ground_truth_correctness_mechanistic_judge"
    
    # Step 1: Add baseline_judge level to main experiment
    print(f"\nStep 1: Adding baseline_judge level to {main_exp.name}")
    add_judge_level_to_folder(main_exp, "baseline_judge")
    
    # Step 2: Merge mechanistic_judge data
    print(f"\nStep 2: Merging {mech_judge_exp.name} into main experiment")
    merge_mechanistic_judge(mech_judge_exp, main_exp)
    
    print("\n" + "=" * 60)
    print("DONE - New structure: Data/CLD/run_X/prompt/judge/files")
    print("=" * 60)


if __name__ == "__main__":
    main()

