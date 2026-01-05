#!/usr/bin/env python3
"""
Run RQ2 analysis pipeline on a single Excel file.

Usage:
    python run_rq2_single_file.py <excel_file_path>
"""
import sys
import os
sys.path.insert(0, 'parameter_tuning_experiments/analysis')

import pandas as pd
from pathlib import Path
import argparse

def prepare_data_from_excel(excel_file: str) -> pd.DataFrame:
    """
    Load and prepare data from a single Excel file for RQ2 analysis.
    
    Args:
        excel_file: Path to the Excel file
        
    Returns:
        Prepared DataFrame
    """
    print("="*80)
    print("RQ2 DATA PREPARATION")
    print("="*80)
    print(f"\nLoading data from: {Path(excel_file).name}")
    
    # Load All Edges sheet
    df = pd.read_excel(excel_file, sheet_name="All Edges")
    print(f"✅ Loaded {len(df)} edges")
    
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
    
    # Create hallucination label based on ground truth Classification
    # FP = False Positive = Hallucination, TP = True Positive = Correct
    df['is_hallucination'] = (df['classification'] == 'FP')
    
    # Extract metadata from file path
    file_path = Path(excel_file)
    parts = file_path.parts
    
    # Try to extract experiment_id and cld_name from path
    try:
        # Path structure: .../exp_YYYYMMDD_HHMMSS_HASH/combo_X/prompts_*/CLD_NAME/run*/*.xlsx
        exp_idx = [i for i, p in enumerate(parts) if p.startswith('exp_')][0]
        cld_idx = [i for i, p in enumerate(parts) if p.startswith('Depressive_') or p.startswith('Social_') or p.startswith('older_')][0]
        
        experiment_id = parts[exp_idx]
        cld_name = parts[cld_idx]
    except:
        # Fallback: extract from filename
        experiment_id = 'exp_unknown'
        cld_name = file_path.stem.split('_run')[0].replace('results_', '').replace(experiment_id + '_', '')
    
    df['experiment_id'] = experiment_id
    df['cld_name'] = cld_name
    
    print(f"\n📊 Hallucination Distribution (Ground Truth):")
    print(f"   TP (Correct):        {(df['classification'] == 'TP').sum():3d} ({(df['classification'] == 'TP').sum()/len(df)*100:5.1f}%)")
    print(f"   FP (Hallucination):  {(df['classification'] == 'FP').sum():3d} ({(df['classification'] == 'FP').sum()/len(df)*100:5.1f}%)")
    print(f"   FN (Missed):         {(df['classification'] == 'FN').sum():3d} ({(df['classification'] == 'FN').sum()/len(df)*100:5.1f}%)")
    print(f"   TN (Correct None):   {(df['classification'] == 'TN').sum():3d} ({(df['classification'] == 'TN').sum()/len(df)*100:5.1f}%)")
    
    print(f"\n📋 Metadata:")
    print(f"   Experiment ID: {experiment_id}")
    print(f"   CLD Name:      {cld_name}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run RQ2 analysis on a single Excel file')
    parser.add_argument('excel_file', type=str, help='Path to the Excel file')
    parser.add_argument('--n-permutations', type=int, default=10000, help='Number of permutations (default: 10000)')
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Number of bootstrap samples (default: 10000)')
    
    args = parser.parse_args()
    
    excel_file = args.excel_file
    
    # Check if file exists
    if not Path(excel_file).exists():
        print(f"❌ Error: File not found: {excel_file}")
        sys.exit(1)
    
    # Prepare data
    df = prepare_data_from_excel(excel_file)
    
    # Save to expected location for RQ2 pipeline
    output_dir = Path("parameter_tuning_experiments/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "rq2_standalone_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved data to: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Now run the master pipeline
    print("\n" + "="*80)
    print("RUNNING RQ2 MASTER PIPELINE")
    print("="*80)
    
    from rq2_master_pipeline import RQ2MasterPipeline
    
    pipeline = RQ2MasterPipeline(output_base_dir="parameter_tuning_experiments")
    pipeline.run_full_pipeline(n_permutations=args.n_permutations, n_bootstrap=args.n_bootstrap)
    
    print("\n" + "="*80)
    print("✅ RQ2 ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Outputs saved to:")
    print(f"   • Tables: parameter_tuning_experiments/tables/")
    print(f"   • Figures: parameter_tuning_experiments/figures/")


if __name__ == "__main__":
    main()
