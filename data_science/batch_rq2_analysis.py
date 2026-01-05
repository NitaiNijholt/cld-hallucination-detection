#!/usr/bin/env python3
"""
Batch RQ2 analysis for experiment exp_20251008_180446_a0d1b2de
"""

import subprocess
import sys
from pathlib import Path

# Excel files to analyze
EXCEL_FILES = [
    "parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run1_7a66e382/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_181257.xlsx",
    "parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults/run2_0d57ed9a/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_20251008_200816.xlsx",
    "parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run1_86b016e1/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_Social_norms_and_obesity_prevalence_20251008_213554.xlsx",
    "parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/Social_norms_and_obesity_prevalence/run2_d09234b7/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_Social_norms_and_obesity_prevalence_20251008_221139.xlsx",
    "parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run1_d45241a5/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run1_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251008_230700.xlsx",
    "parameter_tuning_experiments/results/exp_20251008_180446_a0d1b2de/combo_1/prompts_Nitai_C_andrew_nodegen/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet/run2_96605199/results_exp_20251008_180446_a0d1b2de_param_combo_nr_1_prompts_Nitai_C_andrew_nodegen_run2_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_20251009_082149.xlsx",
]

def main():
    print("=" * 80)
    print(f"Starting RQ2 batch analysis for {len(EXCEL_FILES)} files")
    print("=" * 80)
    print()
    
    results = []
    
    for i, file_path in enumerate(EXCEL_FILES, 1):
        filename = Path(file_path).name
        print(f"[{i}/{len(EXCEL_FILES)}] Processing: {filename}")
        print("-" * 80)
        
        cmd = [
            sys.executable,
            "run_rq2_single_file_v2.py",
            "--metric-source", "generator",
            "--hallucination-def", "ground_truth_FP",
            file_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            
            print("✅ SUCCESS")
            results.append((file_path, "SUCCESS"))
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED with return code {e.returncode}")
            results.append((file_path, f"FAILED (code {e.returncode})"))
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append((file_path, f"ERROR: {e}"))
        
        print()
    
    print("=" * 80)
    print("BATCH ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Results Summary:")
    for file_path, status in results:
        filename = Path(file_path).name
        print(f"  {status:20s} {filename}")
    
    print()
    print("Analysis directories:")
    subprocess.run([
        "ls", "-td",
        "parameter_tuning_experiments/rq2_analyses/rq2_analysis_*"
    ])

if __name__ == "__main__":
    main()
