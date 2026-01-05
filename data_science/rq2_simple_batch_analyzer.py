#!/usr/bin/env python3
"""
Simple RQ2 Batch Analyzer
Runs basic RQ2 analysis (correlations, AUC, statistical tests) on all valid Excel files.
Does not require rq2_master_pipeline - implements analysis directly.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings('ignore')

# CI Metrics to analyze
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]

def detect_experiment_type(filepath: str):
    """Detect experiment type from file path."""
    path_str = str(filepath)
    if 'RQ1a_gt_lit_citation' in path_str:
        return 'citation', 'FP_OR_FN'
    elif 'RQ1a_gt_lit_correctness' in path_str:
        return 'correctness', 'FP_OR_FN'
    elif 'RQ1a_corruption_detection' in path_str:
        return 'corruption', 'IS_CORRUPTED'
    else:
        return 'unknown', 'FP_OR_FN'

def extract_metadata(filepath: str):
    """Extract metadata from file path."""
    path = Path(filepath)
    filename = path.name.lower()
    path_str = str(filepath)
    
    metadata = {
        'filepath': str(filepath),
        'filename': path.name
    }
    
    # Detect experiment type and hallucination def
    exp_type, halluc_def = detect_experiment_type(filepath)
    metadata['experiment_type'] = exp_type
    metadata['hallucination_def'] = halluc_def
    
    # Detect CLD
    if 'depressive' in path_str.lower():
        metadata['cld'] = 'depressive'
    elif 'social_norms' in path_str.lower() or 'Social_norms' in path_str:
        metadata['cld'] = 'social_norms'
    elif 'emergency_department' in path_str.lower() or 'older_persons' in path_str.lower():
        metadata['cld'] = 'emergency_department'
    else:
        metadata['cld'] = 'unknown'
    
    # Detect run (e.g., 'run_1', 'run_2', 'run_3')
    for part in path.parts:
        if part.startswith('run_') or part == 'run':
            metadata['run'] = part  # Keep full run identifier (run_1, run_2, run_3)
            break
    else:
        metadata['run'] = 'unknown'
    
    # Detect prompt type (check specific variants first)
    if 'mechanistic_original' in filename:
        metadata['prompt_type'] = 'mechanistic_original'
    elif 'mechanistic_lit' in filename:
        metadata['prompt_type'] = 'mechanistic_lit'
    elif 'mechanistic' in filename:
        metadata['prompt_type'] = 'mechanistic'
    elif 'cot' in filename:
        metadata['prompt_type'] = 'cot'
    elif 'baseline' in filename:
        metadata['prompt_type'] = 'baseline'
    else:
        metadata['prompt_type'] = 'unknown'
    
    return metadata

def load_and_prepare_data(filepath: str):
    """Load Excel file and prepare for RQ2 analysis."""
    metadata = extract_metadata(filepath)
    
    try:
        # Load data
        df = pd.read_excel(filepath, sheet_name='All Edges')
        
        # Check for required columns
        if 'Classification' not in df.columns:
            return None, metadata, "Missing Classification column"
        
        # Check for CI metrics
        missing_metrics = [m for m in CI_METRICS if m not in df.columns]
        if missing_metrics:
            return None, metadata, f"Missing CI metrics: {missing_metrics}"
        
        # Define hallucination based on experiment type
        if metadata['hallucination_def'] == 'FP_OR_FN':
            df['is_hallucination'] = (df['Classification'] == 'FP') | (df['Classification'] == 'FN')
        elif metadata['hallucination_def'] == 'IS_CORRUPTED':
            if 'Is Corrupted' in df.columns:
                df['is_hallucination'] = df['Is Corrupted'].fillna(False).astype(bool)
            elif 'is_corrupted' in df.columns:
                df['is_hallucination'] = df['is_corrupted'].fillna(False).astype(bool)
            else:
                return None, metadata, "Missing is_corrupted column for corruption experiment"
        else:
            df['is_hallucination'] = (df['Classification'] == 'FP')
        
        # Add metadata to dataframe
        for key, value in metadata.items():
            df[f'meta_{key}'] = value
        
        return df, metadata, None
        
    except Exception as e:
        return None, metadata, str(e)

def calculate_metric_stats(df, metric_name):
    """Calculate statistics for a single CI metric."""
    # Filter out NaN and inf values
    df_clean = df[df[metric_name].notna()].copy()
    df_clean = df_clean[np.isfinite(df_clean[metric_name])]
    
    if len(df_clean) < 10:
        return None
    
    # Separate by hallucination status
    halluc = df_clean[df_clean['is_hallucination']][metric_name]
    correct = df_clean[~df_clean['is_hallucination']][metric_name]
    
    if len(halluc) < 2 or len(correct) < 2:
        return None
    
    # Calculate statistics
    stats_dict = {
        'metric': metric_name,
        'n_total': len(df_clean),
        'n_halluc': len(halluc),
        'n_correct': len(correct),
        'halluc_mean': float(halluc.mean()),
        'halluc_std': float(halluc.std()),
        'correct_mean': float(correct.mean()),
        'correct_std': float(correct.std()),
    }
    
    # Mann-Whitney U test (non-parametric, appropriate for non-normal data)
    try:
        U_stat, p_value = stats.mannwhitneyu(halluc, correct, alternative='two-sided')
        # Rank-biserial correlation as effect size: r = 1 - (2U)/(n1*n2)
        n1, n2 = len(halluc), len(correct)
        rank_biserial = 1 - (2 * U_stat) / (n1 * n2)
        stats_dict['mwu_U'] = float(U_stat)
        stats_dict['mwu_p'] = float(p_value)
        stats_dict['rank_biserial'] = float(rank_biserial)
        stats_dict['significant'] = p_value < 0.05
    except:
        stats_dict['mwu_U'] = None
        stats_dict['mwu_p'] = None
        stats_dict['rank_biserial'] = None
        stats_dict['significant'] = False
    
    # Keep t-test for backwards compatibility (but mark as deprecated)
    try:
        t_stat, t_p_value = stats.ttest_ind(halluc, correct, equal_var=False)
        stats_dict['t_statistic'] = float(t_stat)
        stats_dict['t_p_value'] = float(t_p_value)
    except:
        stats_dict['t_statistic'] = None
        stats_dict['t_p_value'] = None
    
    # Correlation
    try:
        r, p = stats.pearsonr(df_clean[metric_name], df_clean['is_hallucination'].astype(int))
        stats_dict['correlation_r'] = float(r)
        stats_dict['correlation_p'] = float(p)
    except:
        stats_dict['correlation_r'] = None
        stats_dict['correlation_p'] = None
    
    # ROC AUC
    try:
        auc = roc_auc_score(df_clean['is_hallucination'], df_clean[metric_name])
        stats_dict['auc'] = float(auc)
    except:
        stats_dict['auc'] = None
    
    # Cohen's d (effect size)
    try:
        n1, n2 = len(halluc), len(correct)
        var1, var2 = halluc.var(ddof=1), correct.var(ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        d = (halluc.mean() - correct.mean()) / pooled_std if pooled_std > 0 else 0
        stats_dict['cohens_d'] = float(d)
    except:
        stats_dict['cohens_d'] = None
    
    return stats_dict

def analyze_file(filepath: str, output_dir: Path):
    """Analyze a single file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(filepath).name}")
    print(f"{'='*80}")
    
    # Load data
    df, metadata, error = load_and_prepare_data(filepath)
    
    if df is None:
        print(f"   ❌ Failed to load: {error}")
        return {
            'success': False,
            'error': error,
            'metadata': metadata
        }
    
    print(f"   Experiment Type: {metadata['experiment_type']}")
    print(f"   CLD: {metadata['cld']}")
    print(f"   Run: {metadata['run']}")
    print(f"   Prompt: {metadata['prompt_type']}")
    print(f"   Hallucination Def: {metadata['hallucination_def']}")
    print(f"   Total Edges: {len(df)}")
    print(f"   Hallucinations: {df['is_hallucination'].sum()} ({df['is_hallucination'].sum()/len(df)*100:.1f}%)")
    
    # Calculate stats for each metric
    results = []
    for metric in CI_METRICS:
        if metric in df.columns:
            metric_stats = calculate_metric_stats(df, metric)
            if metric_stats:
                results.append(metric_stats)
    
    if not results:
        print(f"   ❌ No valid metrics to analyze")
        return {
            'success': False,
            'error': 'No valid CI metrics',
            'metadata': metadata
        }
    
    # Print summary
    print(f"\n   Metric Statistics:")
    for stat in results:
        auc_str = f"AUC={stat['auc']:.3f}" if stat['auc'] is not None else "AUC=N/A"
        sig_str = "***" if stat.get('significant', False) else ""
        print(f"      {stat['metric']:30s}: {auc_str} {sig_str}")
    
    # Create output subdirectory
    file_out_dir = output_dir / f"{metadata['experiment_type']}_{metadata['cld']}_{metadata['run']}_{metadata['prompt_type']}"
    file_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_out_dir / 'metric_statistics.csv', index=False)
    
    # Save metadata
    with open(file_out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Results saved to: {file_out_dir.name}/")
    
    return {
        'success': True,
        'metadata': metadata,
        'results': results,
        'output_dir': str(file_out_dir)
    }

def main():
    print("="*80)
    print("RQ2 SIMPLE BATCH ANALYZER")
    print("="*80)
    
    # Find the inventory file (prefer deduplicated if exists)
    inventory_dir = Path("parameter_tuning_experiments/rq2_analyses")
    
    # Check for deduplicated list first
    deduplicated_files = list(inventory_dir.glob("rq2_valid_files_deduplicated_*.json"))
    if deduplicated_files:
        inventory_file = max(deduplicated_files, key=lambda p: p.stat().st_mtime)
        print(f"   Using DEDUPLICATED file list")
    else:
        # Fall back to regular inventory
        inventory_files = list(inventory_dir.glob("rq2_valid_files_*.json"))
        if not inventory_files:
            print("❌ No valid files inventory found!")
            print(f"   Run rq2_file_inventory.py first")
            sys.exit(1)
        inventory_file = max(inventory_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📋 Loading inventory: {inventory_file.name}")
    
    with open(inventory_file, 'r') as f:
        valid_files = json.load(f)
    
    print(f"   Total valid files: {len(valid_files)}")
    
    # Create batch output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = inventory_dir / f"rq2_batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Batch output directory: {batch_dir}")
    
    # Process each file
    all_results = []
    successful = 0
    failed = 0
    
    for i, file_info in enumerate(valid_files, 1):
        filepath = file_info['filepath']
        
        print(f"\n[{i}/{len(valid_files)}]", end=" ")
        
        result = analyze_file(filepath, batch_dir)
        result['index'] = i
        result['timestamp'] = datetime.now().isoformat()
        
        if result['success']:
            successful += 1
        else:
            failed += 1
        
        all_results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTotal files:  {len(valid_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"Success rate: {successful/len(valid_files)*100:.1f}%")
    
    # Save batch results (convert numpy types to native Python)
    def convert_to_serializable(obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    batch_results_file = batch_dir / "batch_results.json"
    with open(batch_results_file, 'w') as f:
        json.dump(convert_to_serializable({
            'timestamp': timestamp,
            'total_files': len(valid_files),
            'successful': successful,
            'failed': failed,
            'results': all_results
        }), f, indent=2)
    
    print(f"\n📄 Batch results saved to: {batch_results_file}")
    
    # Print failed files if any
    if failed > 0:
        print(f"\n⚠️  Failed files:")
        for result in all_results:
            if not result['success']:
                meta = result['metadata']
                print(f"   - {meta['cld']}/{meta['run']}/{meta['prompt_type']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n✅ All individual results saved to: {batch_dir}/")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

