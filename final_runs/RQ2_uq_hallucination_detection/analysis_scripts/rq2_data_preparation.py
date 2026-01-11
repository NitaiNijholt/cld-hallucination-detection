#!/usr/bin/env python3
"""
RQ2 Shared Data Preparation Module

Provides reusable data loading functionality for all RQ2 analysis phases.
Loads deduplicated Excel files, applies hallucination definitions, and adds metadata.

Key concept: The independent replicate (block) is (CLD × run). Prompts are 
repeated conditions within each block, not independent samples.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, List, Tuple

from rq2_paths import rq2_dirs, repo_root

# CI metrics available in RQ2 data
CI_METRICS = [
    'Gen Perplexity',
    'Gen Min Prob',
    'Gen Max Window Entropy',
    'Gen Cosine Similarity'
]


def add_block_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add block identifier column combining CLD and run.
    
    The block (CLD × run) is the independent replicate for cross-validation.
    Prompts are repeated conditions within each block, not independent samples.
    
    Args:
        df: DataFrame with 'cld' and 'run' columns
        
    Returns:
        DataFrame with added 'block_id' column
    """
    if 'cld' not in df.columns or 'run' not in df.columns:
        raise ValueError("DataFrame must have 'cld' and 'run' columns")
    
    df = df.copy()
    df['block_id'] = df['cld'].astype(str) + '_' + df['run'].astype(str)
    return df


def get_block_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Get block group labels for GroupKFold cross-validation.
    
    Args:
        df: DataFrame with 'cld' and 'run' columns (or 'block_id')
        
    Returns:
        Array of block group labels aligned with df index
    """
    if 'block_id' in df.columns:
        return df['block_id'].values
    elif 'cld' in df.columns and 'run' in df.columns:
        return (df['cld'].astype(str) + '_' + df['run'].astype(str)).values
    else:
        raise ValueError("DataFrame must have 'block_id' or ('cld', 'run') columns")


def get_latest_inventory_file() -> Optional[Path]:
    """Find the latest deduplicated inventory file."""
    valid_files_path, _unused_output = rq2_dirs()
    
    # First try to find deduplicated files
    dedup_files = list(valid_files_path.glob("rq2_valid_files_deduplicated_*.json"))
    if dedup_files:
        return max(dedup_files, key=lambda p: p.stat().st_mtime)
    
    # Fall back to regular inventory files
    inventory_files = list(valid_files_path.glob("rq2_valid_files_*.json"))
    if inventory_files:
        return max(inventory_files, key=lambda p: p.stat().st_mtime)
    
    return None


def load_rq2_combined_data(verbose: bool = True) -> pd.DataFrame:
    """
    Load and combine all RQ2 data with CLD identifiers and metadata.
    
    Returns:
        DataFrame with columns:
        - CI_METRICS (Gen Perplexity, Gen Min Prob, Gen Max Window Entropy, Gen Cosine Similarity)
        - is_hallucination (bool): True for FP or FN
        - experiment_type (str): 'citation' or 'correctness'
        - cld (str): CLD identifier (depressive, social_norms, emergency_department)
        - run (str): Run identifier
        - prompt_type (str): Prompt variant used
        - All other columns from original Excel files
    """
    if verbose:
        print("="*80)
        print("LOADING RQ2 COMBINED DATA")
        print("="*80)
    
    inventory_file = get_latest_inventory_file()
    
    if inventory_file is None:
        analyses_dir, _unused_output = rq2_dirs()
        raise FileNotFoundError(f"No inventory file found in {analyses_dir}/")
    
    if verbose:
        print(f"\nUsing inventory: {inventory_file.name}")
    
    with open(inventory_file, 'r') as f:
        valid_files = json.load(f)
    
    all_data = []
    skipped = 0
    root = repo_root()
    
    for file_info in valid_files:
        filepath = file_info['filepath']
        exp_type = file_info['experiment_type']
        
        # Skip corruption experiments for natural hallucination analysis
        if exp_type not in ['citation', 'correctness']:
            skipped += 1
            continue
        
        try:
            fp = Path(filepath)
            if not fp.is_absolute():
                fp = root / fp

            df = pd.read_excel(fp, sheet_name='All Edges')
            
            # Check for required columns
            if not all(col in df.columns for col in CI_METRICS):
                if verbose:
                    print(f"  ⚠️  Skipped {fp.name}: Missing CI metrics")
                skipped += 1
                continue
            
            if 'Classification' not in df.columns:
                if verbose:
                    print(f"  ⚠️  Skipped {fp.name}: Missing Classification column")
                skipped += 1
                continue
            
            # Define hallucination based on experiment type
            # For citation and correctness: FP or FN are hallucinations
            df['is_hallucination'] = (df['Classification'] == 'FP') | (df['Classification'] == 'FN')
            
            # Add metadata
            df['experiment_type'] = exp_type
            df['cld'] = file_info['cld']
            df['run'] = file_info['run']
            df['prompt_type'] = file_info['prompt_type']
            df['source_file'] = fp.name
            
            all_data.append(df)
            
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Skipped {Path(filepath).name}: {e}")
            skipped += 1
            continue
    
    if not all_data:
        raise ValueError("No valid data files could be loaded")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print(f"\n✅ Loaded {len(combined)} edges from {len(all_data)} files")
        print(f"   Skipped: {skipped} files")
        print(f"\nHallucination distribution:")
        print(f"   Hallucinations (FP+FN): {combined['is_hallucination'].sum()} ({combined['is_hallucination'].sum()/len(combined)*100:.1f}%)")
        print(f"   Correct (TP+TN):        {(~combined['is_hallucination']).sum()} ({(~combined['is_hallucination']).sum()/len(combined)*100:.1f}%)")
        
        print(f"\nBy experiment type:")
        for exp_type in combined['experiment_type'].unique():
            exp_df = combined[combined['experiment_type'] == exp_type]
            print(f"   {exp_type:15s}: {len(exp_df):4d} edges ({exp_df['is_hallucination'].sum()} hallucinations, {exp_df['is_hallucination'].sum()/len(exp_df)*100:.1f}%)")
        
        print(f"\nBy CLD:")
        for cld in sorted(combined['cld'].unique()):
            cld_df = combined[combined['cld'] == cld]
            print(f"   {cld:25s}: {len(cld_df):4d} edges ({cld_df['is_hallucination'].sum()} hallucinations, {cld_df['is_hallucination'].sum()/len(cld_df)*100:.1f}%)")
    
    return combined


def prepare_clean_dataset(df: pd.DataFrame, 
                         features: Optional[List[str]] = None,
                         verbose: bool = True,
                         return_groups: bool = False) -> Tuple:
    """
    Clean dataset by removing missing/infinite values.
    
    Args:
        df: DataFrame from load_rq2_combined_data()
        features: List of feature columns to use (default: CI_METRICS)
        verbose: Print statistics
        return_groups: If True, also return block group labels for GroupKFold
    
    Returns:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        feature_names: List of feature names
        df_clean: Cleaned DataFrame with all columns
        groups: (only if return_groups=True) Block group labels for GroupKFold
    """
    if features is None:
        features = CI_METRICS
    
    if verbose:
        print(f"\n{'='*80}")
        print("PREPARING CLEAN DATASET")
        print(f"{'='*80}")
    
    # Select rows with all required columns
    required_cols = features + ['is_hallucination', 'experiment_type', 'cld']
    df_clean = df[required_cols + [col for col in df.columns if col not in required_cols]].copy()
    
    # Check for missing values BEFORE processing
    if verbose:
        print(f"\nMissing values before cleaning:")
        for col in features + ['is_hallucination']:
            n_missing = df_clean[col].isna().sum()
            if n_missing > 0:
                print(f"  {col:30s}: {n_missing} missing ({n_missing/len(df_clean)*100:.1f}%)")
    
    # Replace inf values with NaN in features only
    before_inf = len(df_clean)
    for feat in features:
        n_inf = np.isinf(df_clean[feat]).sum()
        if verbose and n_inf > 0:
            print(f"  {feat:30s}: {n_inf} infinite values")
    
    df_clean[features] = df_clean[features].replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN ONLY in the required features
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=features + ['is_hallucination'])
    removed = before - len(df_clean)
    
    if verbose and removed > 0:
        print(f"\n  Removed {removed} rows with missing/infinite values in CI metrics ({removed/before*100:.1f}%)")
        print(f"  These rows were missing one or more of: {', '.join(features)}")
    
    # Reproducibility: enforce deterministic ordering for downstream group splits.
    # This prevents incidental row-order changes (e.g., inventory order) from changing GroupKFold fold composition.
    if return_groups:
        # Ensure block_id exists for sorting
        if 'block_id' not in df_clean.columns:
            df_clean = add_block_id(df_clean)
        # Use stable keys when available; fall back gracefully if columns missing
        sort_cols = [c for c in ['block_id', 'source_file'] if c in df_clean.columns]
        if sort_cols:
            df_clean = df_clean.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

    X = df_clean[features].values
    y = df_clean['is_hallucination'].values
    
    if verbose:
        print(f"\nFinal dataset: {len(df_clean)} samples")
        print(f"  Hallucinations: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"  Correct:        {(~y).sum()} ({(~y).sum()/len(y)*100:.1f}%)")
        
        print(f"\nFeature ranges:")
        for i, col in enumerate(features):
            print(f"  {col:30s}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}] (mean={X[:, i].mean():.4f})")
    
    if return_groups:
        # Add block_id if not present (may already exist from deterministic sort)
        if 'block_id' not in df_clean.columns:
            df_clean = add_block_id(df_clean)
        groups = get_block_groups(df_clean)
        n_blocks = len(np.unique(groups))
        if verbose:
            print(f"\nBlock structure (for GroupKFold):")
            print(f"  N blocks (CLD × run): {n_blocks}")
            for block in sorted(np.unique(groups)):
                n_in_block = np.sum(groups == block)
                print(f"    {block}: {n_in_block} edges")
        return X, y, features, df_clean, groups
    
    return X, y, features, df_clean


def get_cld_mapping() -> dict:
    """Return mapping of CLD identifiers to full names."""
    return {
        'depressive': 'Depressive symptoms in response to a stressor',
        'social_norms': 'Social norms and obesity prevalence',
        'emergency_department': 'Older persons emergency department visits'
    }


def print_data_summary(df: pd.DataFrame):
    """Print detailed summary of loaded data."""
    print(f"\n{'='*80}")
    print("DATA SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTotal edges: {len(df)}")
    print(f"Total files: {df['source_file'].nunique()}")
    
    print(f"\nExperiment types:")
    for exp_type in sorted(df['experiment_type'].unique()):
        count = len(df[df['experiment_type'] == exp_type])
        print(f"  {exp_type:15s}: {count:4d} edges")
    
    print(f"\nCLDs:")
    cld_mapping = get_cld_mapping()
    for cld in sorted(df['cld'].unique()):
        count = len(df[df['cld'] == cld])
        full_name = cld_mapping.get(cld, cld)
        print(f"  {cld:25s}: {count:4d} edges ({full_name})")
    
    print(f"\nPrompt types:")
    for prompt in sorted(df['prompt_type'].unique()):
        count = len(df[df['prompt_type'] == prompt])
        print(f"  {prompt:20s}: {count:4d} edges")
    
    print(f"\nCI Metrics availability:")
    for metric in CI_METRICS:
        non_null = df[metric].notna().sum()
        non_inf = df[metric].replace([np.inf, -np.inf], np.nan).notna().sum()
        print(f"  {metric:30s}: {non_inf:4d}/{len(df)} valid ({non_inf/len(df)*100:.1f}%)")
    
    print()


if __name__ == "__main__":
    # Test the module
    print("Testing RQ2 Data Preparation Module")
    print("="*80)
    
    # Load data
    df = load_rq2_combined_data(verbose=True)
    
    # Print summary
    print_data_summary(df)
    
    # Prepare clean dataset with block groups
    X, y, features, df_clean, groups = prepare_clean_dataset(df, verbose=True, return_groups=True)
    
    print(f"\n✅ Module test complete!")
    print(f"   Loaded: {len(df)} edges")
    print(f"   Clean: {len(df_clean)} edges")
    print(f"   Features: {len(features)}")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Block groups: {len(np.unique(groups))} unique blocks")

