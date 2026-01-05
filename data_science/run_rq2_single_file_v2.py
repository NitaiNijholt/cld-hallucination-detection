#!/usr/bin/env python3
"""
Run RQ2 analysis pipeline on a single Excel file (Enhanced Version).

Improvements:
1. Creates dedicated output folder per analysis with clear naming
2. Adds experimental metadata to all CSV outputs
3. Explicitly specifies CI metric source (generator vs judge)
4. Documents hallucination definition clearly

Usage:
    python run_rq2_single_file_v2.py <excel_file_path> [options]
"""
import sys
import os
sys.path.insert(0, 'parameter_tuning_experiments/analysis')

import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import json


def extract_metadata_from_path(excel_file: str) -> dict:
    """
    Extract comprehensive metadata from the Excel file path.
    
    Args:
        excel_file: Path to the Excel file
        
    Returns:
        Dictionary with metadata including experiment_type and prompt_type
    """
    file_path = Path(excel_file)
    parts = file_path.parts
    filename = file_path.name.lower()
    
    metadata = {
        'excel_file_path': str(excel_file),
        'excel_file_name': file_path.name
    }
    
    # Detect experiment type from path
    path_str = str(excel_file)
    if 'RQ1a_gt_lit_citation' in path_str:
        metadata['experiment_type'] = 'citation'
    elif 'RQ1a_gt_lit_correctness' in path_str:
        metadata['experiment_type'] = 'correctness'
    elif 'RQ1a_corruption_detection' in path_str:
        metadata['experiment_type'] = 'corruption'
    else:
        metadata['experiment_type'] = 'unknown'
    
    # Detect CLD from path
    if 'depressive' in path_str.lower():
        metadata['cld_name'] = 'depressive'
    elif 'social_norms' in path_str.lower() or 'Social_norms' in path_str:
        metadata['cld_name'] = 'social_norms'
    elif 'emergency_department' in path_str.lower() or 'older_persons' in path_str.lower():
        metadata['cld_name'] = 'emergency_department'
    else:
        metadata['cld_name'] = 'unknown'
        
        # Extract run number
    try:
        run_idx = [i for i, p in enumerate(parts) if p.startswith('run')][0]
        run_part = parts[run_idx]
        metadata['run_number'] = run_part.split('_')[0]  # e.g., 'run1' or 'run2'
    except:
        metadata['run_number'] = 'run_unknown'
    
    # Extract prompt type from filename (check specific variants first)
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
    
    # Try to extract other metadata (for backward compatibility)
    try:
        exp_idx = [i for i, p in enumerate(parts) if p.startswith('exp_')][0]
        metadata['experiment_id'] = parts[exp_idx]
    except:
        metadata['experiment_id'] = f"{metadata['experiment_type']}_exp"
    
    try:
        combo_idx = [i for i, p in enumerate(parts) if p.startswith('combo_')][0]
        metadata['combo'] = parts[combo_idx]
    except:
        metadata['combo'] = 'combo_unknown'
        
    try:
        prompt_idx = [i for i, p in enumerate(parts) if 'prompts_' in p][0]
        metadata['prompt_set'] = parts[prompt_idx]
    except:
        metadata['prompt_set'] = 'prompts_unknown'
    
    return metadata


def prepare_data_from_excel(
    excel_file: str,
    metric_source: str = 'generator',
    hallucination_def: str = None
):
    """
    Load and prepare data from a single Excel file for RQ2 analysis.
    
    Args:
        excel_file: Path to the Excel file
        metric_source: Which CI metrics to use ('generator' or 'judge')
        hallucination_def: How to define hallucinations:
            - None: Auto-detect based on experiment type (default)
            - 'ground_truth_FP_OR_FN': Classification == 'FP' OR 'FN'
            - 'ground_truth_FP': Classification == 'FP' 
            - 'is_corrupted': is_corrupted == True
            - 'judge_aggregate_low': aggregate_score < threshold
        
    Returns:
        Tuple of (prepared DataFrame, metadata dict)
    """
    print("="*80)
    print("RQ2 DATA PREPARATION")
    print("="*80)
    
    # Extract metadata
    metadata = extract_metadata_from_path(excel_file)
    metadata['metric_source'] = metric_source
    
    # Auto-detect hallucination definition based on experiment type if not specified
    if hallucination_def is None:
        if metadata['experiment_type'] in ['citation', 'correctness']:
            hallucination_def = 'ground_truth_FP_OR_FN'
        elif metadata['experiment_type'] == 'corruption':
            hallucination_def = 'is_corrupted'
        else:
            hallucination_def = 'ground_truth_FP_OR_FN'  # Default fallback
        print(f"\n🔍 Auto-detected hallucination definition: {hallucination_def} (based on {metadata['experiment_type']})")
    
    metadata['hallucination_definition'] = hallucination_def
    metadata['analysis_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📁 Loading data from: {Path(excel_file).name}")
    print(f"\n📋 Analysis Configuration:")
    print(f"   Experiment Type:      {metadata['experiment_type']}")
    print(f"   CLD Name:             {metadata['cld_name']}")
    print(f"   Run:                  {metadata['run_number']}")
    print(f"   Prompt Type:          {metadata['prompt_type']}")
    print(f"   CI Metric Source:     {metric_source.upper()}")
    print(f"   Hallucination Def:    {hallucination_def}")
    
    # Load All Edges sheet
    df = pd.read_excel(excel_file, sheet_name="All Edges")
    print(f"\n✅ Loaded {len(df)} edges")
    
    # Normalize column names (title case to snake_case with lowercase)
    column_mapping = {
        'Source': 'source',
        'Target': 'target',
        'Classification': 'classification',
        'In Session Graph': 'in_session_graph',
        'In Validation Graph': 'in_validation_graph',
        'Judge Verdict': 'judge_verdict',
        'Motivation': 'motivation',
        'Citation': 'citation',
        'Judge Message': 'judge_message',
        'Aggregate Score': 'aggregate_score',
        'Corrected': 'corrected',
        'Correction Action': 'correction_action',
        'Corrected Motivation': 'corrected_motivation',
        'Corrector Message': 'corrector_message',
        'Is Corrupted': 'is_corrupted',
        'Spurious Motivation': 'spurious_motivation',
        'Perplexity': 'perplexity',
        'Min Prob': 'min_prob',
        'Max Window Entropy': 'max_window_entropy',
        'Gen Perplexity': 'gen_perplexity',
        'Gen Min Prob': 'gen_min_prob',
        'Gen Max Window Entropy': 'gen_max_window_entropy',
        'Judge Perplexity': 'judge_perplexity',
        'Judge Min Prob': 'judge_min_prob',
        'Judge Max Window Entropy': 'judge_max_window_entropy',
        'Cosine Similarity': 'cosine_similarity',
        'Gen Cosine Similarity': 'gen_cosine_similarity',
        'Judge Cosine Similarity': 'judge_cosine_similarity',
    }
    
    df = df.rename(columns=column_mapping)
    
    # Define hallucination based on specified method
    if hallucination_def == 'ground_truth_FP_OR_FN':
        # Ground truth: FP OR FN = Incorrect predictions
        df['is_hallucination'] = (df['classification'] == 'FP') | (df['classification'] == 'FN')
        print(f"\n🎯 Hallucination Definition: Ground Truth (Classification == 'FP' OR 'FN')")
        print(f"   FP = False Positive (generated but shouldn't exist)")
        print(f"   FN = False Negative (should exist but wasn't generated)")
    elif hallucination_def == 'ground_truth_FP':
        # Ground truth: FP = False Positive = Hallucination
        df['is_hallucination'] = (df['classification'] == 'FP')
        print(f"\n🎯 Hallucination Definition: Ground Truth (Classification == 'FP')")
    elif hallucination_def == 'is_corrupted':
        # Corruption experiment: Corrupted edges are synthetic hallucinations
        df['is_hallucination'] = df['is_corrupted'].fillna(False).astype(bool)
        print(f"\n🎯 Hallucination Definition: Corruption (is_corrupted == True)")
    elif hallucination_def == 'judge_aggregate_low':
        # Judge-based: Low aggregate score indicates hallucination
        threshold = 0.5  # Can be made configurable
        df['is_hallucination'] = (df['aggregate_score'] < threshold)
        print(f"\n🎯 Hallucination Definition: Judge Aggregate Score < {threshold}")
    else:
        raise ValueError(f"Unknown hallucination definition: {hallucination_def}")
    
    # Rename CI metric columns based on source
    if metric_source == 'generator':
        # Use generator CI metrics (rename to standard names for pipeline)
        df['perplexity'] = df['gen_perplexity']
        df['min_prob'] = df['gen_min_prob']
        df['max_window_entropy'] = df['gen_max_window_entropy']
        df['cosine_similarity'] = df['gen_cosine_similarity']
        print(f"\n📊 Using GENERATOR CI metrics (gen_*)")
    elif metric_source == 'judge':
        # Use judge CI metrics (rename to standard names for pipeline)
        df['perplexity'] = df['judge_perplexity']
        df['min_prob'] = df['judge_min_prob']
        df['max_window_entropy'] = df['judge_max_window_entropy']
        df['cosine_similarity'] = df['judge_cosine_similarity']
        print(f"\n📊 Using JUDGE CI metrics (judge_*)")
    else:
        raise ValueError(f"Unknown metric source: {metric_source}. Use 'generator' or 'judge'.")
    
    # Add metadata columns to dataframe
    for key, value in metadata.items():
        df[f'meta_{key}'] = value
    
    # Also add experiment_id and cld_name without meta_ prefix (required by RQ2 pipeline)
    df['experiment_id'] = metadata['experiment_id']
    df['cld_name'] = metadata['cld_name']
    
    # Print distribution
    print(f"\n📊 Hallucination Distribution:")
    halluc_count = df['is_hallucination'].sum()
    non_halluc_count = (~df['is_hallucination']).sum()
    print(f"   Hallucinations:      {halluc_count:3d} ({halluc_count/len(df)*100:5.1f}%)")
    print(f"   Non-Hallucinations:  {non_halluc_count:3d} ({non_halluc_count/len(df)*100:5.1f}%)")
    
    if 'classification' in df.columns:
        print(f"\n📊 Ground Truth Classification:")
        print(f"   TP (Correct):        {(df['classification'] == 'TP').sum():3d} ({(df['classification'] == 'TP').sum()/len(df)*100:5.1f}%)")
        print(f"   FP (Hallucination):  {(df['classification'] == 'FP').sum():3d} ({(df['classification'] == 'FP').sum()/len(df)*100:5.1f}%)")
        print(f"   FN (Missed):         {(df['classification'] == 'FN').sum():3d} ({(df['classification'] == 'FN').sum()/len(df)*100:5.1f}%)")
        print(f"   TN (Correct None):   {(df['classification'] == 'TN').sum():3d} ({(df['classification'] == 'TN').sum()/len(df)*100:5.1f}%)")
    
    return df, metadata


def create_output_directory(metadata: dict) -> Path:
    """
    Create a dedicated output directory for this analysis run.
    
    Format: rq2_analysis_<timestamp>_<exp_id>_<cld_name>_<run>_<metric_source>_<halluc_def>
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Path to the output directory
    """
    timestamp = metadata['analysis_timestamp']
    exp_id_short = metadata['experiment_id'].replace('exp_', '')[:12]  # Shorten for readability
    cld_short = metadata['cld_name'][:30]  # Truncate if too long
    run = metadata['run_number']
    metric_src = metadata['metric_source'][:3]  # 'gen' or 'jud'
    halluc_def_short = 'gt' if 'ground_truth' in metadata['hallucination_definition'] else 'js'
    
    dir_name = f"rq2_analysis_{timestamp}_{exp_id_short}_{cld_short}_{run}_{metric_src}_{halluc_def_short}"
    
    base_dir = Path("parameter_tuning_experiments/rq2_analyses")
    output_dir = base_dir / dir_name
    
    # Create subdirectories
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    return output_dir


def add_metadata_to_csv(csv_path: Path, metadata: dict):
    """
    Add metadata columns to a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        metadata: Metadata dictionary
    """
    if not csv_path.exists():
        return
    
    df = pd.read_csv(csv_path)
    
    # Add metadata columns at the beginning
    meta_cols = {
        'analysis_timestamp': metadata['analysis_timestamp'],
        'experiment_type': metadata.get('experiment_type', 'unknown'),
        'experiment_id': metadata['experiment_id'],
        'cld_name': metadata['cld_name'],
        'run_number': metadata['run_number'],
        'prompt_type': metadata.get('prompt_type', 'unknown'),
        'metric_source': metadata['metric_source'],
        'hallucination_definition': metadata['hallucination_definition']
    }
    
    # Insert columns, but skip if they already exist
    for col, val in meta_cols.items():
        if col not in df.columns:
            df.insert(0, col, val)
        else:
            # Update the value if it already exists
            df[col] = val
    
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Run RQ2 analysis on a single Excel file (Enhanced Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze generator CI metrics vs ground truth hallucinations
  python run_rq2_single_file_v2.py results.xlsx
  
  # Analyze judge CI metrics
  python run_rq2_single_file_v2.py results.xlsx --metric-source judge
  
  # Use judge aggregate score as hallucination definition
  python run_rq2_single_file_v2.py results.xlsx --hallucination-def judge_aggregate_low
  
  # Fast testing with fewer permutations
  python run_rq2_single_file_v2.py results.xlsx --n-permutations 1000 --n-bootstrap 1000
        """
    )
    parser.add_argument('excel_file', type=str, help='Path to the Excel file')
    parser.add_argument(
        '--metric-source',
        type=str,
        choices=['generator', 'judge'],
        default='generator',
        help='Which CI metrics to analyze: generator (gen_*) or judge (judge_*) (default: generator)'
    )
    parser.add_argument(
        '--hallucination-def',
        type=str,
        choices=['ground_truth_FP_OR_FN', 'ground_truth_FP', 'is_corrupted', 'judge_aggregate_low'],
        default=None,
        help='Hallucination definition (default: auto-detect based on experiment type)'
    )
    parser.add_argument('--n-permutations', type=int, default=10000, help='Number of permutations (default: 10000)')
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Number of bootstrap samples (default: 10000)')
    
    args = parser.parse_args()
    
    excel_file = args.excel_file
    
    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ Error: File not found: {excel_file}")
        sys.exit(1)
    
    # Prepare data with specified configuration
    df, metadata = prepare_data_from_excel(
        excel_file,
        metric_source=args.metric_source,
        hallucination_def=args.hallucination_def
    )
    
    # Create dedicated output directory
    output_dir = create_output_directory(metadata)
    print(f"\n📁 Created output directory:")
    print(f"   {output_dir}")
    
    # Save metadata
    metadata_file = output_dir / "analysis_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Saved metadata to: {metadata_file.name}")
    
    # Save prepared data (save as both names)
    data_file = output_dir / "tables" / "rq2_prepared_data.csv"
    df.to_csv(data_file, index=False)
    print(f"✅ Saved prepared data to: tables/{data_file.name}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Also save as rq2_standalone_data.csv (required by RQ2 pipeline)
    standalone_file = output_dir / "tables" / "rq2_standalone_data.csv"
    df.to_csv(standalone_file, index=False)
    
    # Run the master pipeline with custom output directory
    print("\n" + "="*80)
    print("RUNNING RQ2 MASTER PIPELINE")
    print("="*80)
    
    from rq2_master_pipeline import RQ2MasterPipeline
    
    # Temporarily override output directories
    original_cwd = os.getcwd()
    
    # Run pipeline
    pipeline = RQ2MasterPipeline(output_base_dir=str(output_dir))
    pipeline.run_full_pipeline(n_permutations=args.n_permutations, n_bootstrap=args.n_bootstrap)
    
    # Add metadata to all generated CSV files
    print("\n📝 Adding metadata to CSV files...")
    tables_dir = output_dir / "tables"
    for csv_file in tables_dir.glob("rq2_*.csv"):
        if csv_file.name != "rq2_prepared_data.csv":  # Skip the one we already have metadata in
            add_metadata_to_csv(csv_file, metadata)
            print(f"   ✓ {csv_file.name}")
    
    # Rename figures with metadata
    print("\n🖼️  Renaming figures with metadata...")
    figures_dir = output_dir / "figures"
    timestamp = metadata['analysis_timestamp']
    exp_short = metadata['experiment_id'].replace('exp_', '')[:12]
    run = metadata['run_number']
    metric_src = metadata['metric_source'][:3]
    halluc_def_short = 'gt' if 'ground_truth' in metadata['hallucination_definition'] else 'js'
    
    # Create prefix for figure names
    fig_prefix = f"{timestamp}_{exp_short}_{run}_{metric_src}_{halluc_def_short}"
    
    for old_fig in figures_dir.glob("rq2_*.png"):
        # Extract the figure type (e.g., "distributions", "scatter_plots", etc.)
        fig_type = old_fig.stem.replace("rq2_", "")
        new_name = f"rq2_{fig_type}_{fig_prefix}.png"
        new_path = figures_dir / new_name
        old_fig.rename(new_path)
        print(f"   ✓ {old_fig.name} → {new_name}")
    
    print("\n" + "="*80)
    print("✅ RQ2 ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to:")
    print(f"   {output_dir}/")
    print(f"\n   📊 Tables:  {output_dir}/tables/")
    print(f"   📈 Figures: {output_dir}/figures/")
    print(f"   📋 Metadata: {metadata_file.name}")
    
    # Print summary of analysis
    print(f"\n" + "="*80)
    print("📊 ANALYSIS SUMMARY")
    print("="*80)
    print(f"   Experiment Type:      {metadata['experiment_type']}")
    print(f"   CLD:                  {metadata['cld_name']}")
    print(f"   Run:                  {metadata['run_number']}")
    print(f"   Prompt Type:          {metadata['prompt_type']}")
    print(f"   CI Metrics Source:    {metadata['metric_source'].upper()}")
    print(f"   Hallucination Def:    {metadata['hallucination_definition']}")
    print(f"   Total Edges:          {len(df)}")
    print(f"   Hallucinations:       {df['is_hallucination'].sum()} ({df['is_hallucination'].sum()/len(df)*100:.1f}%)")
    print(f"   Timestamp:            {metadata['analysis_timestamp']}")


if __name__ == "__main__":
    main()
