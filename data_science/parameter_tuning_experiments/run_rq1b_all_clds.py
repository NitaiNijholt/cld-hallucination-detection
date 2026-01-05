#!/usr/bin/env python3
"""
RQ1b Batch Controller - Run Corrector on All CLDs

Runs correction experiments on all runs from RQ1a_gt_synth_correctness,
aggregating results to CSV.

Usage:
    python run_rq1b_all_clds.py --clds social_norms depressive --yes
"""

import argparse
import subprocess
import sys
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_judged_session_id(rq1a_dir: Path, judge_variant: str = 'baseline') -> str:
    """Extract judged session ID from RQ1a Excel file for specific judge variant."""
    # Look for Excel file with the specified judge variant
    judged_files = list(rq1a_dir.glob(f"judged_*_{judge_variant}_*.xlsx"))
    if not judged_files:
        raise FileNotFoundError(f"No judged files with variant '{judge_variant}' in {rq1a_dir}")
    
    # Use the most recent file if multiple exist
    judged_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    excel_file = judged_files[0]
    
    params_df = pd.read_excel(excel_file, sheet_name='Params')
    session_id_row = params_df[params_df['Parameter'] == 'session_id']
    
    if session_id_row.empty:
        raise ValueError(f"No session_id in {excel_file}")
    
    return session_id_row.iloc[0]['Value']

def extract_base_session_id(rq1a_dir: Path) -> str:
    """Extract base session ID from base_session_info.txt or session_info.txt."""
    # Try base_session_info.txt first (ground truth scenarios)
    info_file = rq1a_dir / 'base_session_info.txt'
    if not info_file.exists():
        # Fall back to session_info.txt (corruption detection scenario)
        info_file = rq1a_dir / 'session_info.txt'
    
    with open(info_file, 'r') as f:
        for line in f:
            if line.startswith('base_session_id='):
                return line.split('=')[1].strip()
    raise ValueError(f"No base_session_id in {info_file}")

def extract_corrected_session_id(judged_id: str) -> str:
    """Extract corrected session ID from mapping file."""
    mapping_file = f"corrector_session_mapping_{judged_id[:8]}.txt"
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.startswith('corrected_session_id='):
                return line.split('=')[1].strip()
    raise ValueError(f"No corrected_session_id in {mapping_file}")

def count_actions(outcomes_file: str) -> dict:
    """Count correction actions from outcomes JSON.
    
    All action types tracked:
    - revise: Motivation text was revised
    - change_type: Edge type was changed (POSITIVE/NEGATIVE/NONE)
    - flip_polarity: Edge polarity was flipped (deprecated, now uses change_type)
    - remove: Edge was removed
    - none: No correction needed
    - error: Correction failed (API error, timeout, parsing error, etc.)
    """
    with open(outcomes_file, 'r') as f:
        outcomes = json.load(f)
    
    # Track ALL action types including error
    counts = {
        'remove': 0, 
        'flip_polarity': 0, 
        'change_type': 0, 
        'revise': 0, 
        'none': 0,
        'error': 0  # NEW: Track error actions explicitly
    }
    for o in outcomes:
        action = o.get('action', 'none')
        if action in counts:
            counts[action] += 1
        else:
            # Unknown action type - log and count as error
            print(f"WARNING: Unknown action type '{action}' - counting as error")
            counts['error'] += 1
    
    return counts

def get_judge_scores(session_id: str) -> dict:
    """Get judge scores from Neo4j session."""
    from neo4j import GraphDatabase
    import os
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            query = '''
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND r.aggregate_score IS NOT NULL
            RETURN avg(r.aggregate_score) as avg_score,
                   count(*) as edges_with_scores,
                   sum(CASE WHEN r.judge_verdict = 'CORRECT' THEN 1 ELSE 0 END) as correct,
                   sum(CASE WHEN r.judge_verdict = 'PARTIALLY_CORRECT' THEN 1 ELSE 0 END) as partial,
                   sum(CASE WHEN r.judge_verdict = 'INCORRECT' THEN 1 ELSE 0 END) as incorrect
            '''
            result = session.run(query, {'sid': session_id}).single()
            
            return {
                'avg_score': result['avg_score'] if result and result['avg_score'] else 0.0,
                'edges_with_scores': result['edges_with_scores'] if result else 0,
                'correct': result['correct'] if result else 0,
                'partial': result['partial'] if result else 0,
                'incorrect': result['incorrect'] if result else 0
            }
    finally:
        driver.close()

def main():
    parser = argparse.ArgumentParser(description="RQ1b Batch Controller")
    parser.add_argument(
        "--clds",
        nargs="+",
        default=["social_norms", "depressive", "emergency_department"],
        choices=["social_norms", "depressive", "emergency_department"],
        help="CLDs to process"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per CLD (end run)")
    parser.add_argument("--start-run", type=int, default=1, help="Starting run number (default: 1)")
    parser.add_argument(
        "--judge-variant",
        default="baseline",
        choices=["baseline", "cot", "mechanistic"],
        help="Judge prompt variant (baseline/cot/mechanistic)"
    )
    parser.add_argument(
        "--corrector-prompt",
        default="baseline",
        choices=["baseline", "cot", "mechanistic"],
        help="Corrector prompt variant (baseline/cot/mechanistic)"
    )
    parser.add_argument(
        "--rq1a-folder",
        default="RQ1a_gt_synth_correctness",
        help="RQ1a base folder name (e.g., RQ1a_gt_synth_correctness)"
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--append", action="store_true", 
                       help="Append to existing results (update/replace matching CLD/Run entries)")
    args = parser.parse_args()
    
    # Derive scenario from folder name
    if 'corruption_detection' in args.rq1a_folder:
        scenario = 'corruption_detection'
    elif 'ground_truth_correctness' in args.rq1a_folder:
        scenario = 'ground_truth_correctness'
    elif 'ground_truth_citation' in args.rq1a_folder or 'citation' in args.rq1a_folder:
        scenario = 'ground_truth_citation'
    else:
        scenario = 'corruption_detection'  # default
    
    # Infer judge approach from folder name
    if 'citation' in args.rq1a_folder.lower():
        judge_approach = 'per_citation_aggregate'
    elif 'correctness' in args.rq1a_folder.lower():
        judge_approach = 'correctness'
    else:
        # Default to correctness for corruption detection
        judge_approach = 'correctness'
    
    # Create output directory name from RQ1a folder, corrector prompt, and judge variant
    output_folder = f"RQ1b_corrector_experiment_{args.rq1a_folder.replace('RQ1a_', '')}_{args.corrector_prompt}_{args.judge_variant}_judge"
    
    print("="*80)
    print("RQ1b BATCH CONTROLLER - ALL CLDs")
    print("="*80)
    print(f"RQ1a Folder: {args.rq1a_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Scenario: {scenario}")
    print(f"Corrector Prompt: {args.corrector_prompt}")
    print(f"Judge Approach: {judge_approach}")
    print(f"Judge Variant: {args.judge_variant}")
    print(f"CLDs: {', '.join(args.clds)}")
    print(f"Runs: {args.start_run} to {args.runs}")
    print()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    rq1a_base = project_root / "final_runs" / args.rq1a_folder
    output_dir = project_root / "final_runs" / output_folder
    output_xlsx = output_dir / "rq1b_all_results.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Excel files for ground truth comparison (if needed)
    excel_files = {
        'social_norms': project_root / 'data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx',
        'depressive': project_root / 'data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx',
        'emergency_department': project_root / 'data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx'
    }
    
    # Plan experiments
    experiments = []
    for cld in args.clds:
        for run_num in range(args.start_run, args.runs + 1):
            rq1a_dir = rq1a_base / cld / f"run_{run_num}"
            if rq1a_dir.exists():
                experiments.append((cld, run_num, rq1a_dir, scenario))
    
    print(f"Found {len(experiments)} experiments to run")
    for cld, run_num, _, _ in experiments:
        print(f"  - {cld}/run_{run_num}")
    print()
    
    # Estimate time
    edge_counts = {'social_norms': 110, 'depressive': 210, 'emergency_department': 1190}
    est_minutes = sum(edge_counts.get(cld, 100) / 40 for cld, _, _, _ in experiments)
    print(f"Estimated time: ~{int(est_minutes)} minutes")
    
    if not args.yes:
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    # Run experiments
    results = []
    
    # If append mode, load existing results first
    existing_df = None
    if args.append and output_xlsx.exists():
        print(f"📂 APPEND MODE: Loading existing results from {output_xlsx}")
        try:
            existing_df = pd.read_excel(output_xlsx, sheet_name='All_Results')
            print(f"   Loaded {len(existing_df)} existing rows")
        except Exception as e:
            print(f"   Warning: Could not load existing results: {e}")
            existing_df = None
    
    for idx, (cld, run_num, rq1a_dir, scenario) in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(experiments)}] {cld.upper()} - RUN {run_num} - {scenario}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 1. Extract session IDs
            print(f"1. Extracting session IDs (judge={args.judge_variant})...")
            judged_id = extract_judged_session_id(rq1a_dir, args.judge_variant)
            base_id = extract_base_session_id(rq1a_dir)
            print(f"   Judged: {judged_id[:8]}...")
            print(f"   Base: {base_id[:8]}...")
            
            # 2. Run corrector
            print("2. Running corrector...")
            corrector_cmd = ['python3', 'data_science/run_corrector_only.py', judged_id, 'gpt-4.1', judge_approach, args.corrector_prompt]
            print(f"   DEBUG: Full session ID being passed: {judged_id}")
            print(f"   DEBUG: Judge approach: {judge_approach}")
            print(f"   DEBUG: Corrector prompt: {args.corrector_prompt}")
            print(f"   DEBUG: Command: {' '.join(corrector_cmd)}")
            result = subprocess.run(
                corrector_cmd,
                capture_output=True,
                text=True,
                cwd='/home/nitai/code/causalix.ai'
            )
            
            if result.returncode != 0:
                print(f"   ❌ Corrector failed (exit code {result.returncode})")
                print(f"   STDERR: {result.stderr[:500]}")
                print(f"   STDOUT: {result.stdout[:500]}")
                continue
            
            print(f"   ✅ Corrector completed")
            
            # 3. Extract corrected session ID
            print("3. Extracting corrected session ID...")
            corrected_id = extract_corrected_session_id(judged_id)
            print(f"   Corrected: {corrected_id[:8]}...")
            
            # 4. Calculate metrics (choose GT based on scenario)
            print("4. Calculating metrics...")
            metrics_xlsx = f"temp_metrics_{cld}_run{run_num}.xlsx"
            
            cmd = [
                'python3', 'data_science/parameter_tuning_experiments/calculate_correction_metrics.py',
                '--corrected-session', corrected_id,
                '--judged-session', judged_id,
                '--cld-name', cld,
                '--run-number', str(run_num),
                '--corrector-model', 'gpt-4.1',
                '--corrector-variant', 'original',
                '--judge-model', 'gpt-4.1',
                '--judge-type', 'correctness',
                '--judge-variant', args.judge_variant,
                '--output', metrics_xlsx
            ]
            
            if scenario == 'corruption_detection':
                # Scenario 1: Compare to synthetic base session
                cmd.extend(['--gt-session', base_id])
            else:
                # Scenarios 2 & 3: Compare to Excel validation
                excel_file = str(excel_files[cld])
                cmd.extend(['--gt-excel', excel_file])
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ❌ Metrics calculation failed!")
                print(f"   STDERR: {result.stderr}")
                print(f"   STDOUT: {result.stdout}")
                continue
            
            # 5. Read metrics
            metrics_df = pd.read_excel(metrics_xlsx)
            metrics_row = metrics_df.iloc[0].to_dict()
            
            # 6. Count actions
            outcomes_file = f"correction_outcomes_{corrected_id[:8]}.json"
            action_counts = count_actions(outcomes_file)
            
            # 7. Get judge scores before and after
            print("5. Getting judge scores...")
            scores_before = get_judge_scores(judged_id)
            scores_after = get_judge_scores(corrected_id)
            
            print(f"   Judge Score Before: {scores_before['avg_score']:.3f}")
            print(f"   Judge Score After: {scores_after['avg_score']:.3f}")
            print(f"   Delta: {scores_after['avg_score'] - scores_before['avg_score']:+.3f}")
            
            # 8. Combine results (merge metrics with additional data)
            runtime = time.time() - start_time
            
            # Metrics row already has most fields, just add what's missing
            result_row = metrics_row.copy()
            result_row.update({
                'Scenario': scenario,
                'Corrector_Prompt': args.corrector_prompt,
                'Actions_Remove': action_counts['remove'],
                'Actions_Flip': action_counts['flip_polarity'],
                'Actions_Change': action_counts['change_type'],
                'Actions_Revise': action_counts['revise'],
                'Actions_None': action_counts['none'],       # NEW: Track none actions
                'Actions_Error': action_counts['error'],     # NEW: Track error actions
                'Actions_Total': sum(action_counts.values()),
                'Judge_Score_Before': scores_before['avg_score'],
                'Judge_Score_After': scores_after['avg_score'],
                'Judge_Score_Delta': scores_after['avg_score'] - scores_before['avg_score'],
                'Correct_Before': scores_before['correct'],
                'Correct_After': scores_after['correct'],
                'Partial_Before': scores_before['partial'],
                'Partial_After': scores_after['partial'],
                'Incorrect_Before': scores_before['incorrect'],
                'Incorrect_After': scores_after['incorrect'],
                'Runtime_Seconds': runtime
            })
            
            results.append(result_row)
            
            # Copy individual run files to output directory
            run_output_dir = output_dir / cld / f"run_{run_num}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            import glob
            
            print("6. Copying output files to run directory...")
            
            # 1. Copy metrics summary Excel
            run_metrics_file = run_output_dir / f"metrics_{cld}_run{run_num}.xlsx"
            shutil.copy(metrics_xlsx, run_metrics_file)
            print(f"   ✅ Metrics: {run_metrics_file.name}")
            
            # 2. Copy judged (pre-corrected) session Excel
            judged_excel_pattern = f"judged_{judged_id[:8]}_before_correction*.xlsx"
            judged_excel_files = glob.glob(judged_excel_pattern)
            if judged_excel_files:
                for judged_file in judged_excel_files:
                    dest = run_output_dir / Path(judged_file).name
                    shutil.copy(judged_file, dest)
                    print(f"   ✅ Judged (pre-corrected) session: {dest.name}")
            else:
                print(f"   ⚠️  No judged session Excel found (pattern: {judged_excel_pattern})")
            
            # 3. Copy corrected session Excel (detailed edge data - created by corrector)
            corrected_excel_pattern = f"corrected_{judged_id[:8]}_to_{corrected_id[:8]}*.xlsx"
            corrected_excel_files = glob.glob(corrected_excel_pattern)
            if corrected_excel_files:
                for corrected_file in corrected_excel_files:
                    dest = run_output_dir / Path(corrected_file).name
                    shutil.copy(corrected_file, dest)
                    print(f"   ✅ Corrected session: {dest.name}")
            else:
                print(f"   ⚠️  No corrected session Excel found (pattern: {corrected_excel_pattern})")
            
            # 4. Copy correction outcomes JSON
            outcomes_pattern = f"*correction_outcomes*{corrected_id[:8]}*.json"
            outcomes_files = glob.glob(outcomes_pattern)
            if outcomes_files:
                for outcomes_file in outcomes_files:
                    shutil.copy(outcomes_file, run_output_dir / Path(outcomes_file).name)
                print(f"   ✅ Correction outcomes: {len(outcomes_files)} JSON file(s)")
            
            # Also try the simpler pattern
            simple_outcomes = f"correction_outcomes_{corrected_id[:8]}.json"
            if Path(simple_outcomes).exists() and simple_outcomes not in [Path(f).name for f in outcomes_files]:
                shutil.copy(simple_outcomes, run_output_dir / simple_outcomes)
            
            # 5. Copy session mapping files
            mapping_pattern = f"*correction_session_mapping*{judged_id[:8]}*.txt"
            mapping_files = glob.glob(mapping_pattern)
            if mapping_files:
                for mapping_file in mapping_files:
                    shutil.copy(mapping_file, run_output_dir / Path(mapping_file).name)
                print(f"   ✅ Session mappings: {len(mapping_files)} file(s)")
            
            # 6. Clean up temporary files from project root
            print("7. Cleaning up temporary files...")
            temp_files_to_remove = []
            
            # Add judged Excel files
            temp_files_to_remove.extend(judged_excel_files)
            # Add corrected Excel files
            temp_files_to_remove.extend(corrected_excel_files)
            # Add outcomes JSON files
            temp_files_to_remove.extend(outcomes_files)
            # Add mapping files
            temp_files_to_remove.extend(mapping_files)
            # Add metrics file
            temp_files_to_remove.append(metrics_xlsx)
            
            removed_count = 0
            for temp_file in temp_files_to_remove:
                try:
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
                        removed_count += 1
                except Exception as e:
                    print(f"   ⚠️  Could not remove {temp_file}: {e}")
            
            if removed_count > 0:
                print(f"   ✅ Removed {removed_count} temporary file(s)")
            
            print(f"   ✅ Complete! F1 Δ: {metrics_row['F1_Delta']:+.3f}")
            print(f"   📁 All run data saved to: {run_output_dir}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save aggregate results with statistics
    if results:
        df = pd.DataFrame(results)
        
        # If append mode, merge with existing results
        if args.append and existing_df is not None:
            # Remove existing rows that match the CLD/Run we just processed
            for _, new_row in df.iterrows():
                mask = (existing_df['CLD'] == new_row['CLD']) & (existing_df['Run'] == new_row['Run'])
                existing_df = existing_df[~mask]
            
            # Append new results
            df = pd.concat([existing_df, df], ignore_index=True)
            print(f"   Merged: {len(existing_df)} existing + {len(results)} new = {len(df)} total rows")
        
        # Calculate statistics per CLD
        stats_rows = []
        for cld in df['CLD'].unique():
            cld_data = df[df['CLD'] == cld]
            n = len(cld_data)
            
            # Calculate mean and 95% CI for key metrics
            import scipy.stats as stats
            
            for metric in ['F1_Delta', 'Precision_Delta', 'Recall_Delta', 'Judge_Score_Delta',
                          'Actions_Total', 'Actions_Revise', 'Actions_Change', 'Actions_None', 'Actions_Error']:
                values = cld_data[metric].values
                mean = values.mean()
                std = values.std(ddof=1) if n > 1 else 0
                if n > 1:
                    sem = stats.sem(values)
                    if sem > 0:  # Only calculate CI if there's variance
                        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
                        ci_lower, ci_upper = ci
                    else:
                        # All values identical, no variance
                        ci_lower = ci_upper = mean
                else:
                    ci_lower = ci_upper = mean
                
                stats_rows.append({
                    'CLD': cld,
                    'Metric': metric,
                    'N': n,
                    'Mean': mean,
                    'Std': std,
                    'CI_95_Lower': ci_lower,
                    'CI_95_Upper': ci_upper,
                    'CI_95_Range': f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                })
        
        stats_df = pd.DataFrame(stats_rows)
        
        # Calculate overall statistics (including action metrics)
        overall_stats_rows = []
        for metric in ['F1_Delta', 'Precision_Delta', 'Recall_Delta', 'Judge_Score_Delta', 
                       'Actions_Total', 'Actions_Revise', 'Actions_Change', 'Actions_None', 'Actions_Error']:
            values = df[metric].values
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0
            n = len(values)
            if n > 1:
                sem = stats.sem(values)
                if sem > 0:  # Only calculate CI if there's variance
                    ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
                    ci_lower, ci_upper = ci
                else:
                    # All values identical, no variance
                    ci_lower = ci_upper = mean
            else:
                ci_lower = ci_upper = mean
            
            overall_stats_rows.append({
                'Metric': metric,
                'N': n,
                'Mean': mean,
                'Std': std,
                'CI_95_Lower': ci_lower,
                'CI_95_Upper': ci_upper,
                'CI_95_Range': f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            })
        
        overall_stats_df = pd.DataFrame(overall_stats_rows)
        
        # Calculate MACRO-AVERAGED statistics (mean of per-CLD means)
        # This treats each CLD equally regardless of sample size
        macro_stats_rows = []
        
        # First, get per-CLD means
        per_cld_means = df.groupby('CLD').agg({
            'F1_Delta': 'mean',
            'Precision_Delta': 'mean',
            'Recall_Delta': 'mean',
            'Judge_Score_Delta': 'mean',
            'Actions_Total': 'sum',
            'Actions_Revise': 'sum',
            'Actions_Change': 'sum',
            'Actions_None': 'sum',
            'Actions_Error': 'sum'
        }).reset_index()
        
        n_clds = len(per_cld_means)
        
        for metric in ['F1_Delta', 'Precision_Delta', 'Recall_Delta', 'Judge_Score_Delta']:
            values = per_cld_means[metric].values
            macro_mean = values.mean()
            macro_std = values.std(ddof=0)  # Population std since we have all CLDs
            
            macro_stats_rows.append({
                'Metric': metric,
                'N_CLDs': n_clds,
                'Macro_Mean': macro_mean,
                'Macro_Std': macro_std,
                'Description': f'Mean of {n_clds} per-CLD means (treats each CLD equally)'
            })
        
        # For action metrics, sum across all CLDs
        for metric in ['Actions_Total', 'Actions_Revise', 'Actions_Change']:
            total = per_cld_means[metric].sum()
            macro_stats_rows.append({
                'Metric': metric,
                'N_CLDs': n_clds,
                'Macro_Mean': total,
                'Macro_Std': 0,
                'Description': f'Total sum across all CLDs'
            })
        
        macro_stats_df = pd.DataFrame(macro_stats_rows)
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Results', index=False)
            stats_df.to_excel(writer, sheet_name='Statistics_Per_CLD', index=False)
            overall_stats_df.to_excel(writer, sheet_name='Statistics_Overall', index=False)
            macro_stats_df.to_excel(writer, sheet_name='Statistics_Macro_Average', index=False)
            per_cld_means.to_excel(writer, sheet_name='Per_CLD_Means', index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ ALL EXPERIMENTS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_xlsx}")
        print(f"  - Sheet 1: All_Results (raw data)")
        print(f"  - Sheet 2: Statistics_Per_CLD (mean & 95% CI by CLD)")
        print(f"  - Sheet 3: Statistics_Overall (micro-averaged: pooled across all runs)")
        print(f"  - Sheet 4: Statistics_Macro_Average (macro-averaged: mean of per-CLD means)")
        print(f"  - Sheet 5: Per_CLD_Means (per-CLD aggregates for macro averaging)")
        print(f"\nIndividual run data saved to: {output_dir}/[cld]/run_[n]/")
        print(f"\nSummary:")
        print(df[['CLD', 'Run', 'F1_Delta', 'Actions_Total']].to_string(index=False))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

