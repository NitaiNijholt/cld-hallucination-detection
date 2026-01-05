#!/usr/bin/env python3
"""
RQ1 Corruption Experiment - Analysis Script

Analyzes results from the corruption experiment by comparing:
1. Baseline (clean, no corruption)
2. Corrupted (50% corruption, judged but not corrected)
3. Corrected (50% corruption, judged and corrected)

Usage:
    python analyze_corruption_experiment.py <experiment_directory>

Example:
    python analyze_corruption_experiment.py parameter_tuning_experiments/results/rq1_corruption_experiment_20251011_120000
"""

import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def load_metrics_from_excel(excel_path: Path) -> dict:
    """Extract precision, recall, F1 from Excel summary sheet."""
    try:
        # Try multiple possible sheet names
        sheet_names_to_try = ['Metrics', 'Cited Edges Only Summary', 'All Edges Summary', 'Summary']
        df = None
        
        for sheet_name in sheet_names_to_try:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                break
            except:
                continue
        
        if df is None:
            return {}
        
        # Find rows with the key metrics
        metrics = {}
        
        # Look for precision, recall, F1 in the Metric column
        # Priority: exact matches first, then with 'edge', then with 'node'
        for _, row in df.iterrows():
            metric_name = str(row.get('Metric', '')).lower().strip()
            value = row.get('Value', None)
            
            # Try exact matches first (edge-level metrics)
            if metric_name == 'precision':
                metrics['precision'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif metric_name == 'recall':
                metrics['recall'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif metric_name == 'f1 score':
                metrics['f1'] = float(value) if value is not None and pd.notna(value) else 0.0
            # Fallback: look for 'edge' keyword
            elif 'precision' in metric_name and 'edge' in metric_name and 'precision' not in metrics:
                metrics['precision'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif 'recall' in metric_name and 'edge' in metric_name and 'recall' not in metrics:
                metrics['recall'] = float(value) if value is not None and pd.notna(value) else 0.0
            elif 'f1' in metric_name and 'edge' in metric_name and 'f1' not in metrics:
                metrics['f1'] = float(value) if value is not None and pd.notna(value) else 0.0
        
        # Calculate F1 if not found
        if 'f1' not in metrics and 'precision' in metrics and 'recall' in metrics:
            p, r = metrics['precision'], metrics['recall']
            if p + r > 0:
                metrics['f1'] = 2 * (p * r) / (p + r)
            else:
                metrics['f1'] = 0.0
        
        return metrics
    
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        return {}

def find_phase0_baseline(exp_dir: Path) -> Path:
    """Find the baseline Excel file from Phase 0."""
    # Check if session_info.txt has the base session ID
    session_info = exp_dir / "session_info.txt"
    if session_info.exists():
        with open(session_info, 'r') as f:
            for line in f:
                if line.startswith('base_session_id='):
                    base_session_id = line.split('=')[1].strip()
                    
                    # Search for Phase 0 results with this session ID
                    results_dir = exp_dir.parent
                    for phase0_dir in results_dir.glob('rq1_phase0_generation_*'):
                        for excel_file in phase0_dir.glob('*.xlsx'):
                            # Check if this Excel has the matching session ID
                            try:
                                params_df = pd.read_excel(excel_file, sheet_name='Params')
                                for _, row in params_df.iterrows():
                                    if row.get('Parameter') == 'session_id' and row.get('Value') == base_session_id:
                                        return excel_file
                            except:
                                continue
    
    return None

def extract_corruption_type_breakdown(excel_path: Path) -> dict:
    """
    Extract breakdown of corruption types (motivation vs structural).
    
    Returns dict with counts:
    - motivation_corruption: Edges with spurious motivations
    - structural_new_edge: Entirely spurious edges (not in ground truth)
    - structural_polarity_flip: Ground truth edges with flipped polarity
    - total_corrupted: Total corrupted edges
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        
        # Count different corruption types
        motivation_corruption = 0
        structural_new_edge = 0
        structural_polarity_flip = 0
        total_corrupted = 0
        
        for _, row in df.iterrows():
            is_corrupted = row.get('Is Corrupted', False)
            
            if is_corrupted:
                total_corrupted += 1
                
                # Check corruption type (if column exists)
                corruption_type = row.get('Corruption Type', '').lower() if pd.notna(row.get('Corruption Type')) else ''
                corruption_subtype = row.get('Corruption Subtype', '').lower() if pd.notna(row.get('Corruption Subtype')) else ''
                
                if 'structural' in corruption_type:
                    if 'new_edge' in corruption_subtype:
                        structural_new_edge += 1
                    elif 'polarity_flip' in corruption_subtype or 'flip' in corruption_subtype:
                        structural_polarity_flip += 1
                else:
                    # Default to motivation corruption if not structural
                    motivation_corruption += 1
        
        return {
            'total_corrupted': total_corrupted,
            'motivation_corruption': motivation_corruption,
            'structural_new_edge': structural_new_edge,
            'structural_polarity_flip': structural_polarity_flip
        }
    
    except Exception as e:
        print(f"Error extracting corruption types from {excel_path}: {e}")
        return {}

def extract_correction_details(before_path: Path, after_path: Path) -> pd.DataFrame:
    """
    Extract edge-level correction details by comparing before and after correction files.
    
    Returns DataFrame with:
    - Source, Target, Relationship Type
    - Is Corrupted (ground truth)
    - Initial Judge Verdict (before correction)
    - Corrected (was correction attempted?)
    - Correction Action (revise/recite/remove)
    - Final Judge Verdict (after re-judging)
    - Verdict Changed (did re-judging change the verdict?)
    """
    try:
        # Read both files - use "All Edges" to include corrected edges that may have been removed (no citations)
        df_before = pd.read_excel(before_path, sheet_name='All Edges')
        df_after = pd.read_excel(after_path, sheet_name='All Edges')
        
        # Create edge ID for merging
        df_before['Edge_ID'] = df_before['Source'] + ' -> ' + df_before['Target']
        df_after['Edge_ID'] = df_after['Source'] + ' -> ' + df_after['Target']
        
        # Extract initial verdicts from before file
        initial_verdicts = {}
        for _, row in df_before.iterrows():
            edge_id = row['Edge_ID']
            judge_msg = row.get('Judge Message', '')
            try:
                judge_data = json.loads(judge_msg)
                initial_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
            except:
                initial_verdict = row.get('Judge Verdict', 'UNKNOWN')
            initial_verdicts[edge_id] = initial_verdict
        
        # Extract final verdicts and correction info from after file
        correction_data = []
        for _, row in df_after.iterrows():
            edge_id = row['Edge_ID']
            
            # Get final verdict
            judge_msg = row.get('Judge Message', '')
            try:
                judge_data = json.loads(judge_msg)
                final_verdict = judge_data.get('aggregate_verdict', 'UNKNOWN')
            except:
                final_verdict = row.get('Judge Verdict', 'UNKNOWN')
            
            # Get initial verdict
            initial_verdict = initial_verdicts.get(edge_id, 'NOT_FOUND')
            
            # Check if verdict changed
            verdict_changed = (initial_verdict != final_verdict) and (initial_verdict != 'NOT_FOUND')
            
            # Extract new edge properties
            corruption_type = row.get('Corruption Type', '') if pd.notna(row.get('Corruption Type')) else ''
            corruption_subtype = row.get('Corruption Subtype', '') if pd.notna(row.get('Corruption Subtype')) else ''
            added_by_corrector = row.get('Added by Corrector', False)
            removed_by_corrector = row.get('Removed by Corrector', False)
            removal_reasoning = row.get('Removal Reasoning', '') if pd.notna(row.get('Removal Reasoning')) else ''
            corrector_reasoning = row.get('Corrector Reasoning', '') if pd.notna(row.get('Corrector Reasoning')) else ''
            
            correction_data.append({
                'Source': row['Source'],
                'Target': row['Target'],
                'Relationship Type': row.get('Relationship Type', 'N/A'),
                'Is Corrupted': row.get('Is Corrupted', False),
                'Corruption Type': corruption_type,
                'Corruption Subtype': corruption_subtype,
                'Initial Verdict': initial_verdict,
                'Corrected': row.get('Corrected', False),
                'Correction Action': row.get('Correction Action', ''),
                'Corrector Message': row.get('Corrector Message', ''),
                'Corrector Reasoning': corrector_reasoning,
                'Added by Corrector': added_by_corrector,
                'Removed by Corrector': removed_by_corrector,
                'Removal Reasoning': removal_reasoning,
                'Final Verdict': final_verdict,
                'Verdict Changed': verdict_changed,
                'Corrupted_and_Corrected': row.get('Is Corrupted', False) and row.get('Corrected', False)
            })
        
        correction_df = pd.DataFrame(correction_data)
        return correction_df
    
    except Exception as e:
        print(f"Error extracting correction details: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def extract_judge_performance(excel_path: Path) -> dict:
    """
    Extract judge performance metrics from judged Excel file.
    
    Compares:
    - Ground Truth: Is Corrupted column (True = corrupted, False = clean)
    - Judge Prediction: Judge Message aggregate_verdict (INCORRECT = judge says corrupted, CORRECT = judge says clean)
    
    Returns:
    - precision: Of edges flagged as incorrect, how many were actually corrupted
    - recall: Of actually corrupted edges, how many did judge flag as incorrect
    - f1: Harmonic mean of precision and recall
    - accuracy: Overall % of correct judgments
    - confusion matrix: TP, TN, FP, FN
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='Cited Edges Only')
        
        # Extract aggregate_score from Judge Message JSON (or Aggregate Score column)
        verdicts = []
        for _, row in df.iterrows():
            judge_msg = row.get('Judge Message', '')
            is_corrupted = row.get('Is Corrupted', False)
            
            # Try to get aggregate_score from Aggregate Score column first, then from JSON
            aggregate_score = row.get('Aggregate Score', np.nan)
            try:
                judge_data = json.loads(judge_msg)
                if pd.isna(aggregate_score):
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
            except:
                pass
            
            verdicts.append({
                'is_corrupted': is_corrupted,
                'aggregate_score': aggregate_score
            })
        
        verdicts_df = pd.DataFrame(verdicts)
        
        # Map to binary using aggregate_score threshold:
        # Ground truth: True (corrupted) = 1, False (clean) = 0
        # Judge prediction: score <= 0.5 = 1 (flagged as hallucination), score > 0.5 = 0 (accepted as correct)
        verdicts_df['ground_truth'] = verdicts_df['is_corrupted'].astype(int)
        
        # Score-based threshold: 0 and 0.5 are hallucination (1), only >0.5 is correct (0)
        # Handle NaN scores by defaulting to 0.5 (neutral, treated as hallucination)
        verdicts_df['aggregate_score'] = verdicts_df['aggregate_score'].fillna(0.5)
        verdicts_df['judge_pred'] = (verdicts_df['aggregate_score'] <= 0.5).astype(int)
        
        # Calculate confusion matrix
        tp = ((verdicts_df['ground_truth'] == 1) & (verdicts_df['judge_pred'] == 1)).sum()
        tn = ((verdicts_df['ground_truth'] == 0) & (verdicts_df['judge_pred'] == 0)).sum()
        fp = ((verdicts_df['ground_truth'] == 0) & (verdicts_df['judge_pred'] == 1)).sum()
        fn = ((verdicts_df['ground_truth'] == 1) & (verdicts_df['judge_pred'] == 0)).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(verdicts_df) if len(verdicts_df) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_total': len(verdicts_df),
            'n_corrupted': int(verdicts_df['ground_truth'].sum()),
            'n_clean': int((verdicts_df['ground_truth'] == 0).sum())
        }
    
    except Exception as e:
        print(f"Error extracting judge performance from {excel_path}: {e}")
        return {}

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_corruption_experiment.py <experiment_directory>")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    
    if not exp_dir.exists():
        print(f"Error: Directory not found: {exp_dir}")
        sys.exit(1)
    
    print("="*80)
    print("RQ1 CORRUPTION EXPERIMENT ANALYSIS")
    print("="*80)
    print(f"Experiment directory: {exp_dir}")
    print()
    
    # Find all Excel files
    excel_files = {}
    
    # 1. Baseline (from Phase 0)
    print("Looking for baseline (Phase 0) file...")
    baseline_file = find_phase0_baseline(exp_dir)
    if baseline_file:
        print(f"  ✓ Found: {baseline_file.name}")
        excel_files['baseline'] = baseline_file
    else:
        print("  ✗ Not found - looking in experiment directory...")
        # Fallback: look for baseline in exp_dir
        for f in exp_dir.glob('baseline*.xlsx'):
            excel_files['baseline'] = f
            print(f"  ✓ Found: {f.name}")
            break
    
    # 2. Corrupted (Phase 1 + Phase 2a judging without correction)
    print("Looking for corrupted+judged file...")
    judged_files = sorted(exp_dir.glob('judged_no_correction*.xlsx'), key=lambda x: x.stat().st_mtime, reverse=True)
    if judged_files:
        excel_files['corrupted'] = judged_files[0]
        print(f"  ✓ Found: {judged_files[0].name} (latest of {len(judged_files)} files)")
    else:
        print("  ✗ No judged file found")
    
    # 3. Corrected (Phase 2b: judging with correction)
    print("Looking for corrected file...")
    corrected_files = sorted(exp_dir.glob('corrected_*.xlsx'), key=lambda x: x.stat().st_mtime, reverse=True)
    if corrected_files:
        excel_files['corrected'] = corrected_files[0]
        print(f"  ✓ Found: {corrected_files[0].name} (latest of {len(corrected_files)} files)")
    else:
        print("  ✗ No corrected file found")
    
    print()
    
    if len(excel_files) < 3:
        print("Error: Not all required files found!")
        print(f"Found: {list(excel_files.keys())}")
        print("Expected: ['baseline', 'corrupted', 'corrected']")
        sys.exit(1)
    
    # Load metrics from each file
    print("="*80)
    print("EXTRACTING METRICS")
    print("="*80)
    
    results = {}
    for condition, excel_path in excel_files.items():
        print(f"\n{condition.upper()}: {excel_path.name}")
        metrics = load_metrics_from_excel(excel_path)
        results[condition] = metrics
        
        if metrics:
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {metrics.get('recall', 0):.4f}")
            print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
        else:
            print("  ✗ Failed to extract metrics")
    
    # Create comparison table
    print()
    print("="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    comparison_data = []
    for condition in ['baseline', 'corrupted', 'corrected']:
        metrics = results.get(condition, {})
        comparison_data.append({
            'Condition': condition.capitalize(),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements
    print()
    print("="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    baseline_f1 = results['baseline'].get('f1', 0)
    corrupted_f1 = results['corrupted'].get('f1', 0)
    corrected_f1 = results['corrected'].get('f1', 0)
    
    degradation = ((corrupted_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
    recovery = ((corrected_f1 - corrupted_f1) / corrupted_f1 * 100) if corrupted_f1 > 0 else 0
    overall_gap = ((corrected_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
    
    print(f"\nF1 Score Changes:")
    print(f"  Baseline → Corrupted: {degradation:+.2f}% (degradation from corruption)")
    print(f"  Corrupted → Corrected: {recovery:+.2f}% (improvement from correction)")
    print(f"  Baseline → Corrected: {overall_gap:+.2f}% (overall gap)")
    print()
    
    # RQ1b Analysis
    print()
    print("="*80)
    print("RQ1b: DOES CORRECTOR IMPROVE CLD ACCURACY?")
    print("="*80)
    print("Comparing structural CLD metrics before and after correction\n")
    
    baseline_precision = results['baseline'].get('precision', 0)
    corrupted_precision = results['corrupted'].get('precision', 0)
    corrected_precision = results['corrected'].get('precision', 0)
    
    baseline_recall = results['baseline'].get('recall', 0)
    corrupted_recall = results['corrupted'].get('recall', 0)
    corrected_recall = results['corrected'].get('recall', 0)
    
    # Calculate changes
    precision_recovery = ((corrected_precision - corrupted_precision) / corrupted_precision * 100) if corrupted_precision > 0 else 0
    recall_recovery = ((corrected_recall - corrupted_recall) / corrupted_recall * 100) if corrupted_recall > 0 else 0
    
    print(f"📊 CLD Structural Accuracy Changes:")
    print(f"  Precision: {corrupted_precision:.4f} → {corrected_precision:.4f} ({precision_recovery:+.2f}%)")
    print(f"  Recall:    {corrupted_recall:.4f} → {corrected_recall:.4f} ({recall_recovery:+.2f}%)")
    print(f"  F1 Score:  {corrupted_f1:.4f} → {corrected_f1:.4f} ({recovery:+.2f}%)")
    
    # Determine if hypothesis is confirmed
    if corrected_f1 > corrupted_f1:
        improvement_magnitude = "strong" if recovery > 10 else "moderate" if recovery > 5 else "slight"
        print(f"\n✅ RQ1b ANSWER: YES - Corrector shows {improvement_magnitude} improvement in CLD accuracy")
        print(f"   Correction improved structural F1 by {recovery:.2f}%")
    elif corrected_f1 == corrupted_f1:
        print(f"\n⚠️  RQ1b ANSWER: NO CHANGE - Corrector did not affect CLD accuracy")
        print(f"   F1 score remained at {corrupted_f1:.4f}")
    else:
        print(f"\n✗ RQ1b ANSWER: NO - Corrector did not improve CLD accuracy")
        print(f"   F1 score decreased by {abs(recovery):.2f}%")
        print(f"   (This may indicate over-correction or removal of valid edges)")
    
    # Extract judge performance metrics
    print()
    print("="*80)
    print("JUDGE PERFORMANCE ANALYSIS")
    print("="*80)
    print("Analyzing judge ability to detect synthetic corruptions...")
    print("Comparing judge verdicts (CORRECT/INCORRECT) against ground truth (Is Corrupted)")
    print("Note: Edges are never deleted from Neo4j; removed edges are marked as 'None' type\n")
    
    judge_results = {}
    condition_labels = {
        'corrupted': 'BEFORE CORRECTION (Phase 2a)',
        'corrected': 'AFTER CORRECTION (Phase 2b)'
    }
    
    for condition in ['corrupted', 'corrected']:
        if condition in excel_files:
            label = condition_labels[condition]
            print(f"\n{label}")
            print(f"  File: {excel_files[condition].name}")
            judge_perf = extract_judge_performance(excel_files[condition])
            judge_results[condition] = judge_perf
            
            if judge_perf:
                print(f"  Dataset: {judge_perf['n_corrupted']} corrupted + {judge_perf['n_clean']} clean = {judge_perf['n_total']} edges")
                print(f"  Confusion Matrix:")
                print(f"    TP (Correctly flagged corrupted): {judge_perf['tp']}")
                print(f"    TN (Correctly accepted clean):    {judge_perf['tn']}")
                print(f"    FP (Incorrectly flagged clean):   {judge_perf['fp']}")
                print(f"    FN (Missed corrupted):            {judge_perf['fn']}")
                print(f"  Metrics:")
                print(f"    Precision: {judge_perf['precision']:.4f} (of flagged edges, % actually corrupted)")
                print(f"    Recall:    {judge_perf['recall']:.4f} (of corrupted edges, % detected)")
                print(f"    F1 Score:  {judge_perf['f1']:.4f} (harmonic mean)")
                print(f"    Accuracy:  {judge_perf['accuracy']:.4f} (overall % correct)")
            else:
                print("  ✗ Failed to extract judge performance")
    
    # Compare judge performance between conditions
    if 'corrupted' in judge_results and 'corrected' in judge_results:
        corrupted_judge_f1 = judge_results['corrupted'].get('f1', 0)
        corrected_judge_f1 = judge_results['corrected'].get('f1', 0)
        judge_f1_change = ((corrected_judge_f1 - corrupted_judge_f1) / corrupted_judge_f1 * 100) if corrupted_judge_f1 > 0 else 0
        
        print()
        print("Judge Performance Comparison:")
        print(f"  Corrupted → Corrected: {judge_f1_change:+.2f}% change in judge F1")
        
        if corrupted_judge_f1 > 0.7:
            print(f"✅ Judge demonstrates strong hallucination detection on synthetic corruptions (F1={corrupted_judge_f1:.3f})")
        elif corrupted_judge_f1 > 0.5:
            print(f"⚠️  Judge shows moderate hallucination detection on synthetic corruptions (F1={corrupted_judge_f1:.3f})")
        else:
            print(f"✗ Judge struggles with hallucination detection on synthetic corruptions (F1={corrupted_judge_f1:.3f})")
    
    # Extract correction details by comparing before and after files
    print()
    print("="*80)
    print("CORRECTION & RE-JUDGING DETAILS")
    print("="*80)
    print("Comparing judge verdicts before correction (Phase 2a) vs after correction (Phase 2b)")
    
    correction_details_df = pd.DataFrame()
    if 'corrupted' in excel_files and 'corrected' in excel_files:
        print(f"  Before: {excel_files['corrupted'].name}")
        print(f"  After:  {excel_files['corrected'].name}")
        correction_details_df = extract_correction_details(excel_files['corrupted'], excel_files['corrected'])
        
        if not correction_details_df.empty:
            n_total = len(correction_details_df)
            print(f"\n  Total edges: {n_total}")
            
            # Count correction actions
            if 'Corrected' in correction_details_df.columns:
                n_corrected = correction_details_df['Corrected'].sum()
                print(f"  Edges corrected: {n_corrected} ({n_corrected/n_total*100:.1f}%)")
                
                if 'Correction Action' in correction_details_df.columns:
                    action_counts = correction_details_df[correction_details_df['Corrected'] == True]['Correction Action'].value_counts()
                    print(f"    Breakdown by action:")
                    for action, count in action_counts.items():
                        if action and str(action).strip():
                            print(f"      - {action}: {count}")
                
                if 'Corrupted_and_Corrected' in correction_details_df.columns:
                    n_corrupt_and_corrected = correction_details_df['Corrupted_and_Corrected'].sum()
                    n_corrupted = correction_details_df['Is Corrupted'].sum() if 'Is Corrupted' in correction_details_df.columns else 0
                    print(f"  Corrupted edges that were corrected: {n_corrupt_and_corrected}/{n_corrupted}")
            
            # Show corruption type breakdown
            if 'Corruption Type' in correction_details_df.columns:
                corruption_breakdown = extract_corruption_type_breakdown(excel_files['corrected'])
                if corruption_breakdown:
                    total_corrupt = corruption_breakdown.get('total_corrupted', 0)
                    print(f"\n  Corruption type breakdown (total: {total_corrupt} corrupted edges):")
                    print(f"    - Motivation corruption: {corruption_breakdown.get('motivation_corruption', 0)}")
                    print(f"    - Structural (new spurious edges): {corruption_breakdown.get('structural_new_edge', 0)}")
                    print(f"    - Structural (polarity flips): {corruption_breakdown.get('structural_polarity_flip', 0)}")
            
            # Show corrector agent actions
            if 'Added by Corrector' in correction_details_df.columns and 'Removed by Corrector' in correction_details_df.columns:
                n_added = correction_details_df['Added by Corrector'].sum()
                n_removed = correction_details_df['Removed by Corrector'].sum()
                print(f"\n  Corrector agent actions:")
                print(f"    - Edges added: {n_added}")
                print(f"    - Edges removed: {n_removed}")
                
                # Show removal reasons if available
                if n_removed > 0 and 'Removal Reasoning' in correction_details_df.columns:
                    removed_edges = correction_details_df[correction_details_df['Removed by Corrector'] == True]
                    if not removed_edges.empty:
                        print(f"    Removal reasoning examples:")
                        for idx, (_, row) in enumerate(removed_edges.head(3).iterrows(), 1):
                            reasoning = row['Removal Reasoning']
                            if reasoning:
                                print(f"      {idx}. {row['Source']} → {row['Target']}: {reasoning[:100]}...")
            
            # Show verdict changes
            if 'Verdict Changed' in correction_details_df.columns:
                n_verdict_changed = correction_details_df['Verdict Changed'].sum()
                print(f"\n  Verdict changes after re-judging: {n_verdict_changed}/{n_total} ({n_verdict_changed/n_total*100:.1f}%)")
                
                # Show breakdown of verdict changes
                verdict_changes = correction_details_df[correction_details_df['Verdict Changed'] == True]
                if not verdict_changes.empty and 'Initial Verdict' in verdict_changes.columns and 'Final Verdict' in verdict_changes.columns:
                    print(f"    Verdict transitions:")
                    for initial, final in verdict_changes[['Initial Verdict', 'Final Verdict']].value_counts().items():
                        print(f"      - {initial[0]} → {initial[1]}: {final} edges")
                
                # Show if corrections led to verdict improvements
                if 'Is Corrupted' in correction_details_df.columns:
                    corrected_edges = correction_details_df[correction_details_df['Corrected'] == True]
                    if not corrected_edges.empty:
                        verdict_mapping = {'INCORRECT': 1, 'PARTIALLY_CORRECT': 1, 'CORRECT': 0, 'UNKNOWN': 0}
                        corrected_edges['initial_flagged'] = corrected_edges['Initial Verdict'].map(verdict_mapping).fillna(0)
                        corrected_edges['final_flagged'] = corrected_edges['Final Verdict'].map(verdict_mapping).fillna(0)
                        
                        # Count edges where correction improved the verdict
                        improved = ((corrected_edges['Is Corrupted'] == True) & 
                                   (corrected_edges['initial_flagged'] == 0) & 
                                   (corrected_edges['final_flagged'] == 1)).sum()
                        fixed = ((corrected_edges['Is Corrupted'] == True) & 
                                (corrected_edges['initial_flagged'] == 1) & 
                                (corrected_edges['final_flagged'] == 0)).sum()
                        
                        if improved > 0 or fixed > 0:
                            print(f"\n  Impact of correction on corrupted edges:")
                            if improved > 0:
                                print(f"    - Initially missed, now detected: {improved}")
                            if fixed > 0:
                                print(f"    - Initially flagged, now accepted after fix: {fixed}")
        else:
            print("  ✗ No correction details found")
    
    # Save results
    output_file = exp_dir / "analysis_results.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        # Add detailed metrics sheet
        detailed_data = []
        for condition, excel_path in excel_files.items():
            detailed_data.append({
                'Condition': condition.capitalize(),
                'Excel File': excel_path.name,
                'Precision': results[condition].get('precision', 0),
                'Recall': results[condition].get('recall', 0),
                'F1 Score': results[condition].get('f1', 0)
            })
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # Add improvement analysis sheet
        improvement_df = pd.DataFrame([{
            'Metric': 'F1 Degradation (Baseline → Corrupted)',
            'Value': degradation,
            'Unit': '%'
        }, {
            'Metric': 'F1 Recovery (Corrupted → Corrected)',
            'Value': recovery,
            'Unit': '%'
        }, {
            'Metric': 'Overall Gap (Baseline → Corrected)',
            'Value': overall_gap,
            'Unit': '%'
        }])
        improvement_df.to_excel(writer, sheet_name='Improvement Analysis', index=False)
        
        # Add judge performance sheet
        if judge_results:
            judge_perf_data = []
            for condition in ['corrupted', 'corrected']:
                if condition in judge_results and judge_results[condition]:
                    perf = judge_results[condition]
                    # Use clearer labels for the conditions
                    condition_label = {
                        'corrupted': 'Before Correction (Phase 2a)',
                        'corrected': 'After Correction (Phase 2b)'
                    }.get(condition, condition.capitalize())
                    
                    judge_perf_data.append({
                        'Condition': condition_label,
                        'Precision': perf.get('precision', 0),
                        'Recall': perf.get('recall', 0),
                        'F1 Score': perf.get('f1', 0),
                        'Accuracy': perf.get('accuracy', 0),
                        'TP': perf.get('tp', 0),
                        'TN': perf.get('tn', 0),
                        'FP': perf.get('fp', 0),
                        'FN': perf.get('fn', 0),
                        'N Corrupted': perf.get('n_corrupted', 0),
                        'N Clean': perf.get('n_clean', 0),
                        'N Total': perf.get('n_total', 0)
                    })
            
            if judge_perf_data:
                judge_perf_df = pd.DataFrame(judge_perf_data)
                judge_perf_df.to_excel(writer, sheet_name='Judge Performance', index=False)
        
        # Add correction details sheet (edge-level data showing verdict transitions)
        if not correction_details_df.empty:
            correction_details_df.to_excel(writer, sheet_name='Correction Details', index=False)
        
        # Add RQ1b Analysis sheet
        rq1b_data = pd.DataFrame([
            {'Metric': 'Baseline Precision', 'Before Correction': baseline_precision, 'After Correction': baseline_precision, 'Change (%)': 0.0},
            {'Metric': 'Corrupted→Corrected Precision', 'Before Correction': corrupted_precision, 'After Correction': corrected_precision, 'Change (%)': precision_recovery},
            {'Metric': 'Baseline Recall', 'Before Correction': baseline_recall, 'After Correction': baseline_recall, 'Change (%)': 0.0},
            {'Metric': 'Corrupted→Corrected Recall', 'Before Correction': corrupted_recall, 'After Correction': corrected_recall, 'Change (%)': recall_recovery},
            {'Metric': 'Baseline F1 Score', 'Before Correction': baseline_f1, 'After Correction': baseline_f1, 'Change (%)': 0.0},
            {'Metric': 'Corrupted→Corrected F1 Score', 'Before Correction': corrupted_f1, 'After Correction': corrected_f1, 'Change (%)': recovery}
        ])
        rq1b_data.to_excel(writer, sheet_name='RQ1b CLD Accuracy', index=False)
        
        # Add RQ1b summary
        if corrected_f1 > corrupted_f1:
            improvement_magnitude = "Strong" if recovery > 10 else "Moderate" if recovery > 5 else "Slight"
            answer = f"YES - {improvement_magnitude} improvement"
        elif corrected_f1 == corrupted_f1:
            answer = "NO CHANGE"
        else:
            answer = "NO - Decreased accuracy"
        
        rq1b_summary = pd.DataFrame([{
            'Research Question': 'RQ1b: Does corrector improve CLD accuracy?',
            'Answer': answer,
            'F1 Improvement': f"{recovery:+.2f}%",
            'Precision Improvement': f"{precision_recovery:+.2f}%",
            'Recall Improvement': f"{recall_recovery:+.2f}%"
        }])
        rq1b_summary.to_excel(writer, sheet_name='RQ1b Summary', index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Create visualization
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: F1 Score comparison
    ax1 = axes[0]
    conditions = ['Baseline', 'Corrupted', 'Corrected']
    f1_scores = [baseline_f1, corrupted_f1, corrected_f1]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = ax1.bar(conditions, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score Comparison: Baseline vs Corrupted vs Corrected', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: All metrics comparison
    ax2 = axes[1]
    metrics_comparison = comparison_df.set_index('Condition')[['Precision', 'Recall', 'F1 Score']]
    metrics_comparison.plot(kind='bar', ax=ax2, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.legend(title='Metric', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = exp_dir / "comparison_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {exp_dir}")
    print(f"  - {output_file.name}")
    print(f"  - {plot_file.name}")
    print()

if __name__ == "__main__":
    main()
