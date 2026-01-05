#!/usr/bin/env python3
"""
Run RQ2 analysis with BOTH hallucination definitions by default.

This runs two complete analyses:
1. Ground truth definition: classification == 'FP'
2. Judge-based definition: aggregate_score < threshold

This allows comparison to diagnose whether unexpected correlations are due to
ground truth issues or metric issues.

Usage:
    python run_rq2_dual_analysis.py <excel_file_path> [options]
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
        Dictionary with metadata
    """
    file_path = Path(excel_file)
    parts = file_path.parts
    
    metadata = {
        'excel_file_path': str(excel_file),
        'excel_file_name': file_path.name
    }
    
    # Try to extract experiment_id and cld_name from path
    try:
        # Path structure: .../exp_YYYYMMDD_HHMMSS_HASH/combo_X/prompts_*/CLD_NAME/run*/*.xlsx
        exp_idx = [i for i, p in enumerate(parts) if p.startswith('exp_')][0]
        metadata['experiment_id'] = parts[exp_idx]
        
        cld_idx = [i for i, p in enumerate(parts) if p.startswith('Depressive_') or p.startswith('Social_') or p.startswith('older_')][0]
        metadata['cld_name'] = parts[cld_idx]
        
        # Extract run number
        run_idx = [i for i, p in enumerate(parts) if p.startswith('run')][0]
        run_part = parts[run_idx]
        metadata['run_number'] = run_part.split('_')[0]  # e.g., 'run1' or 'run2'
        
        # Extract combo number
        combo_idx = [i for i, p in enumerate(parts) if p.startswith('combo_')][0]
        metadata['combo'] = parts[combo_idx]
        
        # Extract prompt set
        prompt_idx = [i for i, p in enumerate(parts) if 'prompts_' in p][0]
        metadata['prompt_set'] = parts[prompt_idx]
        
    except Exception as e:
        print(f"Warning: Could not fully extract metadata from path: {e}")
        metadata['experiment_id'] = 'exp_unknown'
        metadata['cld_name'] = 'unknown_cld'
        metadata['run_number'] = 'run_unknown'
        metadata['combo'] = 'combo_unknown'
        metadata['prompt_set'] = 'prompts_unknown'
    
    return metadata


def prepare_data_from_excel(
    excel_file: str,
    metric_source: str = 'generator',
    hallucination_def: str = 'ground_truth_FP'
):
    """
    Load and prepare data from a single Excel file for RQ2 analysis.
    
    Args:
        excel_file: Path to the Excel file
        metric_source: Which CI metrics to use ('generator' or 'judge')
        hallucination_def: How to define hallucinations:
            - 'ground_truth_FP': Classification == 'FP' (default)
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
    metadata['hallucination_definition'] = hallucination_def
    metadata['analysis_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📁 Loading data from: {Path(excel_file).name}")
    print(f"\n📋 Analysis Configuration:")
    print(f"   Experiment ID:        {metadata['experiment_id']}")
    print(f"   CLD Name:             {metadata['cld_name']}")
    print(f"   Run:                  {metadata['run_number']}")
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
    if hallucination_def == 'ground_truth_FP':
        # Ground truth: FP = False Positive = Hallucination
        df['is_hallucination'] = (df['classification'] == 'FP')
        print(f"\n🎯 Hallucination Definition: Ground Truth (Classification == 'FP')")
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
        'experiment_id': metadata['experiment_id'],
        'cld_name': metadata['cld_name'],
        'run_number': metadata['run_number'],
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


def run_single_analysis(excel_file, metric_source, hallucination_def, n_permutations, n_bootstrap):
    """Run a single RQ2 analysis with specified configuration."""
    
    # Prepare data with specified configuration
    df, metadata = prepare_data_from_excel(
        excel_file,
        metric_source=metric_source,
        hallucination_def=hallucination_def
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
    
    # Save prepared data
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
    
    # Run pipeline
    pipeline = RQ2MasterPipeline(output_base_dir=str(output_dir))
    pipeline.run_full_pipeline(n_permutations=n_permutations, n_bootstrap=n_bootstrap)
    
    # Add metadata to all generated CSV files
    print("\n📝 Adding metadata to CSV files...")
    tables_dir = output_dir / "tables"
    for csv_file in tables_dir.glob("rq2_*.csv"):
        if csv_file.name != "rq2_prepared_data.csv":
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
        # Extract the figure type
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
    
    # Print summary
    print(f"\n" + "="*80)
    print("📊 ANALYSIS SUMMARY")
    print("="*80)
    print(f"   Experiment:           {metadata['experiment_id']}")
    print(f"   CLD:                  {metadata['cld_name']}")
    print(f"   Run:                  {metadata['run_number']}")
    print(f"   CI Metrics Source:    {metadata['metric_source'].upper()}")
    print(f"   Hallucination Def:    {metadata['hallucination_definition']}")
    print(f"   Total Edges:          {len(df)}")
    print(f"   Hallucinations:       {df['is_hallucination'].sum()} ({df['is_hallucination'].sum()/len(df)*100:.1f}%)")
    print(f"   Timestamp:            {metadata['analysis_timestamp']}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run RQ2 analysis with BOTH hallucination definitions (Dual Analysis)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both analyses (default behavior)
  python run_rq2_dual_analysis.py results.xlsx
  
  # Specify metric source
  python run_rq2_dual_analysis.py results.xlsx --metric-source judge
  
  # Fast testing
  python run_rq2_dual_analysis.py results.xlsx --n-permutations 1000 --n-bootstrap 1000
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
    parser.add_argument('--n-permutations', type=int, default=10000, help='Number of permutations (default: 10000)')
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Number of bootstrap samples (default: 10000)')
    
    args = parser.parse_args()
    
    excel_file = args.excel_file
    
    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ Error: File not found: {excel_file}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🔬 RQ2 DUAL ANALYSIS - RUNNING BOTH HALLUCINATION DEFINITIONS")
    print("="*80)
    print("\nThis will run TWO complete analyses:")
    print("  1. Ground Truth Definition: classification == 'FP'")
    print("  2. Judge-Based Definition: aggregate_score < 0.5")
    print("\n" + "="*80)
    
    # Run Analysis 1: Ground truth FP
    print("\n\n")
    print("🔵 " + "="*76 + " 🔵")
    print("🔵 ANALYSIS 1: GROUND TRUTH HALLUCINATION DEFINITION" + " "*23 + "🔵")
    print("🔵 " + "="*76 + " 🔵")
    output_dir_1 = run_single_analysis(
        excel_file,
        metric_source=args.metric_source,
        hallucination_def='ground_truth_FP',
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap
    )
    
    # Run Analysis 2: Judge aggregate score
    print("\n\n")
    print("🟢 " + "="*76 + " 🟢")
    print("🟢 ANALYSIS 2: JUDGE AGGREGATE SCORE HALLUCINATION DEFINITION" + " "*14 + "🟢")
    print("🟢 " + "="*76 + " 🟢")
    output_dir_2 = run_single_analysis(
        excel_file,
        metric_source=args.metric_source,
        hallucination_def='judge_aggregate_low',
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap
    )
    
    # Final summary
    print("\n\n")
    print("="*80)
    print("🎉 DUAL ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📂 Output Directories:")
    print(f"\n  1️⃣  Ground Truth:  {output_dir_1}")
    print(f"  2️⃣  Judge-Based:   {output_dir_2}")
    print("\n💡 Compare the correlation results between the two analyses to diagnose:")
    print("   - If cosine/aggregate correlations flip between analyses")
    print("   - Whether unexpected patterns are due to ground truth or metric issues")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
