#!/usr/bin/env python3
"""
Time and Cost Scaling Aggregation Script

Extracts and aggregates time/cost metrics from RQ1 Excel files,
generating scaling curves distinguished by:
- Judge type (correctness vs citation)
- Prompt variant (baseline, cot, mechanistic)
- CLD size (depressive, social_norms, emergency_department)
- Judge model (gpt-4.1, gpt-5-mini)

Outputs:
- Excel file with raw data and aggregations
- Publication-quality figures for thesis
- LaTeX tables with mean (± 95% CI) per uncertainty reporting rule (N<30)

Usage:
    python aggregate_time_cost_scaling.py --base_dir /path/to/final_runs --output_dir /path/to/output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set matplotlib style for publication-quality figures
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_style("whitegrid")

# OpenAI Pricing (December 2025 - from platform.openai.com/docs/pricing)
# All prices in USD per token
PRICING = {
    # GPT-4.1 series (released April 2025)
    'gpt-4.1': {'input': 2.00 / 1e6, 'output': 8.00 / 1e6},
    'gpt-4.1-mini': {'input': 0.40 / 1e6, 'output': 1.60 / 1e6},
    'gpt-4.1-nano': {'input': 0.10 / 1e6, 'output': 0.40 / 1e6},
    
    # GPT-5-mini (released late 2025)
    'gpt-5-mini': {'input': 0.25 / 1e6, 'output': 2.00 / 1e6},
    
    # GPT-4o series
    'gpt-4o': {'input': 2.50 / 1e6, 'output': 10.00 / 1e6},
    'gpt-4o-2024-11-20': {'input': 2.50 / 1e6, 'output': 10.00 / 1e6},
    'gpt-4o-mini': {'input': 0.15 / 1e6, 'output': 0.60 / 1e6},
    
    # GPT-4 Turbo (legacy)
    'gpt-4-turbo': {'input': 10.00 / 1e6, 'output': 30.00 / 1e6},
    'gpt-4-turbo-2024-04-09': {'input': 10.00 / 1e6, 'output': 30.00 / 1e6},
    
    # Embeddings
    'text-embedding-3-small': {'input': 0.02 / 1e6, 'output': 0.0},
    'text-embedding-3-large': {'input': 0.13 / 1e6, 'output': 0.0},
}

# CLD name mappings for display
CLD_DISPLAY_NAMES = {
    'depressive': 'Depressive Symptoms',
    'social_norms': 'Social Norms',
    'emergency_department': 'Emergency Department',
}

# Experiment type mappings
EXPERIMENT_TYPES = {
    'RQ1a_gt_synth_correctness': {
        'judge_type': 'correctness',
        'ground_truth_type': 'corruption_detection',
    },
    'RQ1a_gt_synth_citation': {
        'judge_type': 'citation',
        'ground_truth_type': 'corruption_detection',
    },
    'RQ1a_gt_lit_correctness': {
        'judge_type': 'correctness',
        'ground_truth_type': 'ground_truth',
    },
    'RQ1a_gt_lit_citation': {
        'judge_type': 'citation',
        'ground_truth_type': 'ground_truth',
    },
}


def extract_llm_stats(excel_path: Path) -> Optional[dict]:
    """
    Extract LLM Usage Stats from a judged Excel file.
    
    Returns dict with stats for each role (Generator, Corruptor, Judge, Embeddings)
    or None if the sheet doesn't exist.
    """
    try:
        xl = pd.ExcelFile(excel_path)
        
        if 'LLM Usage Stats' not in xl.sheet_names:
            return None
        
        df_llm = pd.read_excel(excel_path, sheet_name='LLM Usage Stats')
        
        stats = {}
        for _, row in df_llm.iterrows():
            role = row['Role'].lower()
            
            # Parse token totals (stored as string representation of dict)
            token_str = str(row.get('Token Totals', '{}'))
            try:
                tokens = eval(token_str)
            except:
                tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            
            stats[role] = {
                'inference_time_s': float(row.get('Inference Time (s)', 0)),
                'inference_calls': int(row.get('Inference Calls', 0)),
                'prompt_tokens': tokens.get('prompt_tokens', tokens.get('input_tokens', 0)),
                'completion_tokens': tokens.get('completion_tokens', 0),
                'total_tokens': tokens.get('total_tokens', 0),
            }
        
        return stats
    
    except Exception as e:
        print(f"  Warning: Could not extract LLM stats from {excel_path.name}: {e}")
        return None


def extract_edge_and_node_counts(excel_path: Path) -> tuple:
    """
    Extract the number of edges and nodes from the 'All Edges' sheet.
    
    Returns (n_edges, n_nodes, edge_density)
    """
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        n_edges = len(df)
        
        # Count unique nodes (sources + targets)
        sources = set(df['Source'].dropna().unique())
        targets = set(df['Target'].dropna().unique())
        all_nodes = sources.union(targets)
        n_nodes = len(all_nodes)
        
        # Calculate edge density (for directed graph: max = n*(n-1))
        max_possible_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        edge_density = n_edges / max_possible_edges
        
        return n_edges, n_nodes, edge_density
    except Exception as e:
        print(f"  Warning: Could not extract counts from {excel_path.name}: {e}")
        return 0, 0, 0


def extract_judge_config(excel_path: Path) -> dict:
    """Extract judge configuration from Params sheet."""
    try:
        xl = pd.ExcelFile(excel_path)
        if 'Params' not in xl.sheet_names:
            return {}
        
        df_params = pd.read_excel(excel_path, sheet_name='Params')
        
        config = {}
        for _, row in df_params.iterrows():
            param = row.get('Parameter', '')
            value = row.get('Value', '')
            
            if param == 'judge_config':
                try:
                    judge_cfg = json.loads(str(value).replace("'", '"'))
                    config['judge_model'] = judge_cfg.get('model', 'unknown')
                    config['judge_provider'] = judge_cfg.get('provider', 'unknown')
                    config['judge_temperature'] = judge_cfg.get('temperature', 0.0)
                except:
                    pass
            elif param == 'yaml_path':
                config['yaml_path'] = str(value).split('/')[-1] if value else ''
        
        return config
    
    except Exception as e:
        print(f"  Warning: Could not extract config from {excel_path.name}: {e}")
        return {}


def parse_experiment_config(file_path: Path) -> dict:
    """
    Parse experiment configuration from file path and contents.
    
    Extracts:
    - experiment_folder: Full experiment folder name
    - judge_type: 'correctness' or 'citation'
    - prompt_variant: 'baseline', 'cot', or 'mechanistic'
    - ground_truth_type: 'corruption_detection' or 'ground_truth'
    - cld: CLD name
    - run: Run number
    """
    parts = file_path.parts
    filename = file_path.name
    
    config = {
        'file_path': str(file_path),
        'filename': filename,
    }
    
    # Extract experiment folder
    for part in parts:
        if part.startswith('RQ1a_'):
            config['experiment_folder'] = part
            if part in EXPERIMENT_TYPES:
                config['judge_type'] = EXPERIMENT_TYPES[part]['judge_type']
                config['ground_truth_type'] = EXPERIMENT_TYPES[part]['ground_truth_type']
            else:
                # Infer from name
                config['judge_type'] = 'citation' if 'citation' in part.lower() else 'correctness'
                config['ground_truth_type'] = 'ground_truth' if 'ground_truth' in part.lower() else 'corruption_detection'
            break
    
    # Extract CLD
    for cld in ['depressive', 'social_norms', 'emergency_department']:
        if cld in str(file_path):
            config['cld'] = cld
            break
    
    # Extract run number
    for part in parts:
        if part.startswith('run_'):
            config['run'] = part
            config['run_number'] = int(part.split('_')[1])
            break
    
    # Extract prompt variant from filename
    for variant in ['baseline', 'cot', 'mechanistic']:
        if f'_{variant}_' in filename:
            config['prompt_variant'] = variant
            break
    
    return config


def calculate_estimated_cost(stats: dict, model: str) -> float:
    """Calculate estimated cost based on token usage and model pricing."""
    if model not in PRICING:
        # Try to find a matching model
        for key in PRICING:
            if key in model or model in key:
                model = key
                break
        else:
            return 0.0
    
    pricing = PRICING[model]
    
    # Sum costs across all roles
    total_cost = 0.0
    for role, role_stats in stats.items():
        if role == 'embeddings':
            # Embeddings only have input cost
            total_cost += role_stats.get('prompt_tokens', 0) * PRICING.get('text-embedding-3-small', {}).get('input', 0)
        else:
            total_cost += role_stats.get('prompt_tokens', 0) * pricing['input']
            total_cost += role_stats.get('completion_tokens', 0) * pricing['output']
    
    return total_cost


def collect_all_data(base_dir: Path) -> pd.DataFrame:
    """
    Traverse final_runs directory and collect all data from judged Excel files.
    
    Returns DataFrame with all extracted metrics and configurations.
    """
    print("=" * 80)
    print("COLLECTING DATA FROM JUDGED EXCEL FILES")
    print("=" * 80)
    print(f"Base directory: {base_dir}\n")
    
    all_data = []
    files_processed = 0
    files_with_stats = 0
    
    # Iterate through experiment folders
    for exp_folder in sorted(base_dir.iterdir()):
        if not exp_folder.is_dir():
            continue
        if not exp_folder.name.startswith('RQ1a_'):
            continue
        if 'test' in exp_folder.name.lower() or 'deprecated' in exp_folder.name.lower():
            continue
        
        print(f"Processing: {exp_folder.name}")
        
        # Iterate through CLD folders
        for cld_folder in exp_folder.iterdir():
            if not cld_folder.is_dir():
                continue
            if cld_folder.name.startswith('enhanced_analysis') or cld_folder.name.startswith('sensitivity'):
                continue
            
            # Iterate through run folders
            for run_folder in cld_folder.iterdir():
                if not run_folder.is_dir():
                    continue
                if not run_folder.name.startswith('run_'):
                    continue
                
                # Find judged Excel files
                for file in run_folder.iterdir():
                    if not file.name.startswith('judged_'):
                        continue
                    if not file.name.endswith('.xlsx'):
                        continue
                    if file.name.endswith('.backup.xlsx'):
                        continue
                    
                    files_processed += 1
                    
                    # Extract all data
                    exp_config = parse_experiment_config(file)
                    llm_stats = extract_llm_stats(file)
                    
                    if llm_stats is None:
                        continue
                    
                    files_with_stats += 1
                    
                    edge_count, node_count, edge_density = extract_edge_and_node_counts(file)
                    judge_config = extract_judge_config(file)
                    
                    # Get judge-specific stats
                    generator_stats = llm_stats.get('generator', {})
                    judge_stats = llm_stats.get('judge', {})
                    embeddings_stats = llm_stats.get('embeddings', {})
                    
                    # Calculate estimated cost
                    judge_model = judge_config.get('judge_model', 'gpt-4.1')
                    estimated_cost = calculate_estimated_cost(llm_stats, judge_model)
                    
                    # Build data row
                    row = {
                        # Experiment config
                        'experiment_folder': exp_config.get('experiment_folder', ''),
                        'judge_type': exp_config.get('judge_type', ''),
                        'ground_truth_type': exp_config.get('ground_truth_type', ''),
                        'prompt_variant': exp_config.get('prompt_variant', ''),
                        'cld': exp_config.get('cld', ''),
                        'run': exp_config.get('run', ''),
                        'run_number': exp_config.get('run_number', 0),
                        
                        # Judge config
                        'judge_model': judge_model,
                        'judge_provider': judge_config.get('judge_provider', ''),
                        'yaml_path': judge_config.get('yaml_path', ''),
                        
                        # Edge and node counts
                        'n_edges': edge_count,
                        'n_nodes': node_count,
                        'edge_density': edge_density,
                        
                        # Generator stats (CLD generation step)
                        'generator_inference_time_s': generator_stats.get('inference_time_s', 0),
                        'generator_inference_calls': generator_stats.get('inference_calls', 0),
                        'generator_prompt_tokens': generator_stats.get('prompt_tokens', 0),
                        'generator_completion_tokens': generator_stats.get('completion_tokens', 0),
                        'generator_total_tokens': generator_stats.get('total_tokens', 0),
                        
                        # Judge stats
                        'judge_inference_time_s': judge_stats.get('inference_time_s', 0),
                        'judge_inference_calls': judge_stats.get('inference_calls', 0),
                        'judge_prompt_tokens': judge_stats.get('prompt_tokens', 0),
                        'judge_completion_tokens': judge_stats.get('completion_tokens', 0),
                        'judge_total_tokens': judge_stats.get('total_tokens', 0),
                        
                        # Embeddings stats
                        'embeddings_inference_time_s': embeddings_stats.get('inference_time_s', 0),
                        'embeddings_inference_calls': embeddings_stats.get('inference_calls', 0),
                        'embeddings_total_tokens': embeddings_stats.get('total_tokens', 0),
                        
                        # Total time (judge + embeddings)
                        'total_inference_time_s': judge_stats.get('inference_time_s', 0) + embeddings_stats.get('inference_time_s', 0),
                        
                        # Estimated cost
                        'estimated_cost_usd': estimated_cost,
                        
                        # File info
                        'file_path': str(file),
                        'filename': file.name,
                    }
                    
                    all_data.append(row)
    
    print(f"\n  Files processed: {files_processed}")
    print(f"  Files with LLM stats: {files_with_stats}")
    
    df = pd.DataFrame(all_data)
    return df


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived metrics: time_per_edge, tokens_per_edge, cost_per_edge."""
    df = df.copy()
    
    # Avoid division by zero
    df['time_per_edge'] = df['judge_inference_time_s'] / df['n_edges'].replace(0, np.nan)
    df['tokens_per_edge'] = df['judge_total_tokens'] / df['n_edges'].replace(0, np.nan)
    df['cost_per_edge'] = df['estimated_cost_usd'] / df['n_edges'].replace(0, np.nan)
    
    # Total time per edge (including embeddings)
    df['total_time_per_edge'] = df['total_inference_time_s'] / df['n_edges'].replace(0, np.nan)
    
    # Generator time per edge (generation step)
    if 'generator_inference_time_s' in df.columns:
        df['generator_time_per_edge'] = df['generator_inference_time_s'] / df['n_edges'].replace(0, np.nan)
    
    return df


def calculate_95_ci(values: list) -> tuple:
    """Calculate mean and 95% CI half-width for a list of values."""
    values = [v for v in values if pd.notna(v) and not np.isnan(v) and not np.isinf(v)]
    
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    mean = np.mean(values)
    
    if len(values) < 2:
        return mean, 0, mean, mean
    
    std = np.std(values, ddof=1)
    se = std / np.sqrt(len(values))
    
    # t-value for 95% CI
    t_val = stats.t.ppf(0.975, len(values) - 1)
    ci_halfwidth = t_val * se
    
    ci_low = mean - ci_halfwidth
    ci_high = mean + ci_halfwidth
    
    return mean, ci_halfwidth, ci_low, ci_high


def compute_comprehensive_stats(values: list) -> dict:
    """
    Compute comprehensive statistics for repeated measurements.
    
    Reports mean ± 95% CI (t-distribution) as primary per uncertainty reporting rule.
    Also includes SD, min, and max for reference.
    """
    values = [v for v in values if pd.notna(v) and not np.isnan(v) and not np.isinf(v)]
    n = len(values)
    
    if n == 0:
        return {
            'n': 0, 'values': [], 'mean': np.nan, 'std': np.nan,
            'min': np.nan, 'max': np.nan, 'ci_hw': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan
        }
    
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    
    # 95% CI using t-distribution
    if n > 1:
        se = std_val / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_hw = t_crit * se
        ci_lower = mean_val - ci_hw
        ci_upper = mean_val + ci_hw
    else:
        ci_hw = 0.0
        ci_lower = mean_val
        ci_upper = mean_val
    
    return {
        'n': n,
        'values': [float(v) for v in values],
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'ci_hw': float(ci_hw),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'latex_primary': f"{mean_val:.2f} ± {ci_hw:.2f}",
        'latex_std': f"{mean_val:.2f} ± {std_val:.2f}",
        'latex_ci': f"{mean_val:.2f} ± {ci_hw:.2f}"
    }


def aggregate_by_dimensions(df: pd.DataFrame) -> dict:
    """
    Aggregate data by multiple dimensions:
    - By Judge Type × Prompt Variant
    - By CLD (for scaling analysis)
    - By Model comparison
    
    Returns dict of DataFrames.
    """
    aggregates = {}
    
    # Metrics to aggregate
    metrics = [
        'judge_inference_time_s',
        'total_inference_time_s',
        'judge_total_tokens',
        'estimated_cost_usd',
        'time_per_edge',
        'tokens_per_edge',
        'cost_per_edge',
        'n_edges',
    ]
    
    # 1. Aggregate by Judge Type × Prompt Variant
    print("\nAggregating by Judge Type × Prompt Variant...")
    agg_judge_prompt = []
    
    for (judge_type, prompt_variant), group in df.groupby(['judge_type', 'prompt_variant']):
        row = {
            'judge_type': judge_type,
            'prompt_variant': prompt_variant,
            'n_runs': len(group),
            'n_clds': group['cld'].nunique(),
        }
        
        for metric in metrics:
            values = group[metric].dropna().tolist()
            mean, ci_hw, ci_low, ci_high = calculate_95_ci(values)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci'] = ci_hw
            row[f'{metric}_ci_low'] = ci_low
            row[f'{metric}_ci_high'] = ci_high
            # Add min/max for uncertainty rule (N<30)
            row[f'{metric}_min'] = np.min(values) if values else np.nan
            row[f'{metric}_max'] = np.max(values) if values else np.nan
        
        agg_judge_prompt.append(row)
    
    aggregates['by_judge_prompt'] = pd.DataFrame(agg_judge_prompt)
    
    # 2. Aggregate by CLD × Judge Type × Prompt (for scaling)
    print("Aggregating by CLD × Judge Type × Prompt...")
    agg_cld = []
    
    for (cld, judge_type, prompt_variant), group in df.groupby(['cld', 'judge_type', 'prompt_variant']):
        row = {
            'cld': cld,
            'cld_display': CLD_DISPLAY_NAMES.get(cld, cld),
            'judge_type': judge_type,
            'prompt_variant': prompt_variant,
            'n_runs': len(group),
            'n_edges_typical': int(group['n_edges'].median()),
        }
        
        for metric in metrics:
            values = group[metric].dropna().tolist()
            mean, ci_hw, ci_low, ci_high = calculate_95_ci(values)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci'] = ci_hw
            # Add min/max for uncertainty rule (N<30)
            row[f'{metric}_min'] = np.min(values) if values else np.nan
            row[f'{metric}_max'] = np.max(values) if values else np.nan
        
        agg_cld.append(row)
    
    aggregates['by_cld_scaling'] = pd.DataFrame(agg_cld)
    
    # 3. Aggregate by Model × Judge Type
    print("Aggregating by Model × Judge Type...")
    agg_model = []
    
    for (judge_model, judge_type), group in df.groupby(['judge_model', 'judge_type']):
        row = {
            'judge_model': judge_model,
            'judge_type': judge_type,
            'n_runs': len(group),
            'n_clds': group['cld'].nunique(),
            'n_prompts': group['prompt_variant'].nunique(),
        }
        
        for metric in metrics:
            values = group[metric].dropna().tolist()
            mean, ci_hw, ci_low, ci_high = calculate_95_ci(values)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci'] = ci_hw
            # Add min/max for uncertainty rule (N<30)
            row[f'{metric}_min'] = np.min(values) if values else np.nan
            row[f'{metric}_max'] = np.max(values) if values else np.nan
        
        agg_model.append(row)
    
    aggregates['by_model'] = pd.DataFrame(agg_model)
    
    # 4. Overall aggregate by Judge Type
    print("Aggregating overall by Judge Type...")
    agg_overall = []
    
    for judge_type, group in df.groupby('judge_type'):
        row = {
            'judge_type': judge_type,
            'n_runs': len(group),
            'n_clds': group['cld'].nunique(),
            'n_prompts': group['prompt_variant'].nunique(),
            'n_models': group['judge_model'].nunique(),
        }
        
        for metric in metrics:
            values = group[metric].dropna().tolist()
            mean, ci_hw, ci_low, ci_high = calculate_95_ci(values)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci'] = ci_hw
            # Add min/max for uncertainty rule (N<30)
            row[f'{metric}_min'] = np.min(values) if values else np.nan
            row[f'{metric}_max'] = np.max(values) if values else np.nan
        
        agg_overall.append(row)
    
    aggregates['by_judge_type'] = pd.DataFrame(agg_overall)
    
    return aggregates


def compute_scaling_analysis(df: pd.DataFrame) -> dict:
    """
    Compute scaling analysis: linear regression fits, R² values, and theoretical comparison.
    
    Theoretical expectation: O(n) linear scaling
    - Time ∝ n_edges (each edge judged independently)
    - Tokens ∝ n_edges
    - Cost ∝ n_edges
    """
    from scipy import stats as scipy_stats
    
    results = {}
    
    for judge_type in df['judge_type'].unique():
        subset = df[df['judge_type'] == judge_type]
        if len(subset) < 3:
            continue
        
        results[judge_type] = {}
        
        for metric in ['judge_inference_time_s', 'judge_total_tokens', 'estimated_cost_usd']:
            x = subset['n_edges'].values
            y = subset[metric].values
            
            # Remove NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            
            if len(x) < 3:
                continue
            
            # Linear regression (y = mx + b)
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Time/tokens per edge (should be constant for O(n) scaling)
            per_edge = y / x
            per_edge_mean = np.mean(per_edge)
            per_edge_std = np.std(per_edge)
            per_edge_cv = per_edge_std / per_edge_mean if per_edge_mean > 0 else 0  # Coefficient of variation
            
            results[judge_type][metric] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_err': std_err,
                'per_edge_mean': per_edge_mean,
                'per_edge_std': per_edge_std,
                'per_edge_cv': per_edge_cv,
                'n_points': len(x),
            }
    
    return results


def generate_scaling_figures(df: pd.DataFrame, aggregates: dict, output_dir: Path):
    """
    Generate 2 comprehensive, information-dense figures for thesis.
    
    Figure 1: Scaling Analysis - Time/Cost vs Edges AND Nodes with theoretical fits
    Figure 2: Configuration Efficiency - Comparison across all dimensions
    """
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    
    # Compute scaling analysis
    scaling_results = compute_scaling_analysis(df)
    
    # Color palettes
    prompt_colors = {'baseline': '#2ecc71', 'cot': '#3498db', 'mechanistic': '#9b59b6'}
    prompt_markers = {'baseline': 'o', 'cot': 's', 'mechanistic': '^'}
    judge_type_colors = {'correctness': '#e74c3c', 'citation': '#3498db'}
    cld_colors = {'depressive': '#f39c12', 'social_norms': '#27ae60', 'emergency_department': '#8e44ad'}
    
    # ==================================================================================
    # FIGURE 1: Scaling Analysis (2x2 grid) - Averaged per CLD with 95% CI
    # Row 1: Time vs Edges (LINEAR fit) for both judge types
    # Row 2: Time vs Nodes (QUADRATIC fit) for both judge types
    # ==================================================================================
    print("  Creating: figure1_scaling_analysis.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    from scipy import stats as scipy_stats
    
    # CLD display names and colors
    cld_short_names = {'depressive': 'Depressive', 'social_norms': 'Social Norms', 'emergency_department': 'Emergency Dept.'}
    cld_colors_fig = {'depressive': '#e74c3c', 'social_norms': '#27ae60', 'emergency_department': '#3498db'}
    cld_markers = {'depressive': 'o', 'social_norms': 's', 'emergency_department': '^'}
    
    # Row 1: Time vs Edges with LINEAR fit (averaged per CLD)
    for idx, judge_type in enumerate(['correctness', 'citation']):
        ax = axes[0, idx]
        subset = df[df['judge_type'] == judge_type]
        
        if len(subset) == 0:
            ax.set_title(f'{judge_type.title()} (No Data)')
            continue
        
        # Aggregate by CLD: mean and 95% CI
        cld_agg = subset.groupby('cld').agg({
            'n_edges': 'first',  # Same for all runs of a CLD
            'judge_inference_time_s': ['mean', 'std', 'count']
        }).reset_index()
        cld_agg.columns = ['cld', 'n_edges', 'time_mean', 'time_std', 'n']
        
        # Calculate 95% CI
        cld_agg['time_ci'] = cld_agg.apply(
            lambda row: scipy_stats.t.ppf(0.975, row['n']-1) * row['time_std'] / np.sqrt(row['n']) 
            if row['n'] > 1 else 0, axis=1)
        
        # Plot each CLD as a single point with error bars
        for _, row in cld_agg.iterrows():
            cld = row['cld']
            ax.errorbar(row['n_edges'], row['time_mean'], yerr=row['time_ci'],
                       fmt=cld_markers.get(cld, 'o'), markersize=12, capsize=6, capthick=2,
                       color=cld_colors_fig.get(cld, 'gray'), ecolor=cld_colors_fig.get(cld, 'gray'),
                       label=cld_short_names.get(cld, cld), markeredgecolor='black', markeredgewidth=1.5)
        
        # Linear fit using aggregated means
        x = cld_agg['n_edges'].values
        y = cld_agg['time_mean'].values
        if len(x) >= 2:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(x, y)
            r2 = r_value ** 2
            
            x_line = np.linspace(0, x.max() * 1.15, 100)
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, 'k-', lw=2.5, 
                   label=f'Linear: T = {slope:.2f}E + {intercept:.0f}\nR² = {r2:.3f}')
        
        ax.set_xlabel('Number of Edges (E)', fontsize=12)
        ax.set_ylabel('Inference Time (s) (± 95% CI)', fontsize=12)
        ax.set_title(f'{judge_type.title()} Judging: Time vs Edges\nExpected: O(E) linear scaling', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    # Row 2: Time vs Nodes with QUADRATIC fit (averaged per CLD)
    for idx, judge_type in enumerate(['correctness', 'citation']):
        ax = axes[1, idx]
        subset = df[df['judge_type'] == judge_type]
        
        if len(subset) == 0 or 'n_nodes' not in subset.columns:
            continue
        
        subset_clean = subset[subset['n_nodes'] > 0]
        
        # Aggregate by CLD
        cld_agg = subset_clean.groupby('cld').agg({
            'n_nodes': 'first',
            'judge_inference_time_s': ['mean', 'std', 'count']
        }).reset_index()
        cld_agg.columns = ['cld', 'n_nodes', 'time_mean', 'time_std', 'n']
        cld_agg['time_ci'] = cld_agg.apply(
            lambda row: scipy_stats.t.ppf(0.975, row['n']-1) * row['time_std'] / np.sqrt(row['n']) 
            if row['n'] > 1 else 0, axis=1)
        
        # Plot each CLD
        for _, row in cld_agg.iterrows():
            cld = row['cld']
            ax.errorbar(row['n_nodes'], row['time_mean'], yerr=row['time_ci'],
                       fmt=cld_markers.get(cld, 'o'), markersize=12, capsize=6, capthick=2,
                       color=cld_colors_fig.get(cld, 'gray'), ecolor=cld_colors_fig.get(cld, 'gray'),
                       label=cld_short_names.get(cld, cld), markeredgecolor='black', markeredgewidth=1.5)
        
        # Quadratic fit using aggregated means
        x = cld_agg['n_nodes'].values
        y = cld_agg['time_mean'].values
        if len(x) >= 3:
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs
            
            x_line = np.linspace(x.min() * 0.8, x.max() * 1.15, 100)
            y_quad = np.polyval(coeffs, x_line)
            
            # R² for quadratic
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_quad = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            ax.plot(x_line, y_quad, 'k-', lw=2.5, 
                   label=f'Quadratic: T = {a:.2f}N² + {b:.1f}N + {c:.0f}\nR² = {r2_quad:.3f}')
        
        ax.set_xlabel('Number of Nodes (N)', fontsize=12)
        ax.set_ylabel('Inference Time (s) (± 95% CI)', fontsize=12)
        ax.set_title(f'{judge_type.title()} Judging: Time vs Nodes\nExpected: O(N²) for dense CLDs (E = N×(N-1))', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure1_scaling_analysis.png', dpi=300)
    plt.close()
    
    # ==================================================================================
    # FIGURE 2: Configuration Efficiency Comparison (2x2 grid)
    # ==================================================================================
    print("  Creating: figure2_configuration_comparison.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Time per Edge by Judge Type × Prompt (grouped bar with 95% CI)
    ax = axes[0, 0]
    agg = aggregates['by_judge_prompt'].copy()
    if len(agg) > 0:
        # Grouped bars - explicit ordering
        prompts = ['baseline', 'cot', 'mechanistic']
        judge_types = ['correctness', 'citation']  # Explicit order
        x = np.arange(len(judge_types))
        width = 0.25
        
        for i, prompt in enumerate(prompts):
            values = []
            errors = []
            for jt in judge_types:
                row = agg[(agg['prompt_variant'] == prompt) & (agg['judge_type'] == jt)]
                if len(row) > 0:
                    values.append(row['time_per_edge_mean'].values[0])
                    errors.append(row['time_per_edge_ci'].values[0])
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax.bar(x + i * width, values, width, yerr=errors, 
                         label=prompt.title(), color=prompt_colors.get(prompt), 
                         alpha=0.8, edgecolor='black', capsize=4)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Judge Type', fontsize=11)
        ax.set_ylabel('Time per Edge (s) (± 95% CI)', fontsize=11)
        ax.set_title('(A) Time Efficiency by Configuration', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Correctness', 'Citation'])
        ax.legend(title='Prompt', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Cost per Edge by Judge Type × Prompt (in cents)
    ax = axes[0, 1]
    if len(agg) > 0:
        for i, prompt in enumerate(prompts):
            values = []
            errors = []
            for jt in judge_types:
                row = agg[(agg['prompt_variant'] == prompt) & (agg['judge_type'] == jt)]
                if len(row) > 0:
                    values.append(row['cost_per_edge_mean'].values[0] * 100)  # to cents
                    errors.append(row['cost_per_edge_ci'].values[0] * 100)
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax.bar(x + i * width, values, width, yerr=errors,
                         label=prompt.title(), color=prompt_colors.get(prompt), 
                         alpha=0.8, edgecolor='black', capsize=4)
        
        ax.set_xlabel('Judge Type', fontsize=11)
        ax.set_ylabel('Cost per Edge (¢) (± 95% CI)', fontsize=11)
        ax.set_title('(B) Cost per Edge (USD cents)', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Correctness', 'Citation'])
        ax.legend(title='Prompt', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Tokens per Edge by Judge Type × Prompt (bar chart matching Panel A/B)
    ax = axes[1, 0]
    if len(agg) > 0:
        for i, prompt in enumerate(prompts):
            values = []
            errors = []
            for jt in judge_types:
                row = agg[(agg['prompt_variant'] == prompt) & (agg['judge_type'] == jt)]
                if len(row) > 0:
                    values.append(row['tokens_per_edge_mean'].values[0])
                    errors.append(row['tokens_per_edge_ci'].values[0])
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax.bar(x + i * width, values, width, yerr=errors,
                         label=prompt.title(), color=prompt_colors.get(prompt), 
                         alpha=0.8, edgecolor='black', capsize=4)
        
        ax.set_xlabel('Judge Type', fontsize=11)
        ax.set_ylabel('Tokens per Edge (± 95% CI)', fontsize=11)
        ax.set_title('(C) Tokens per Edge', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Correctness', 'Citation'])
        ax.legend(title='Prompt', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel D: Model Comparison (gpt-4.1 vs gpt-5-mini for citation judging)
    ax = axes[1, 1]
    
    # Compare models on citation judging (where we have both)
    citation_data = df[df['judge_type'] == 'citation']
    models = citation_data['judge_model'].unique()
    
    if len(models) >= 2:
        model_colors = {'gpt-4.1': '#3498db', 'gpt-5-mini': '#2ecc71'}
        metrics = ['time_per_edge', 'cost_per_edge', 'tokens_per_edge']
        metric_labels = ['Time (s/edge)', 'Cost (¢/edge)', 'Tokens/edge']
        multipliers = [1, 100, 1]  # cost to cents
        
        x_pos = np.arange(len(metrics))
        bar_width = 0.35
        
        for i, model in enumerate(['gpt-4.1', 'gpt-5-mini']):
            model_data = citation_data[citation_data['judge_model'] == model]
            if len(model_data) == 0:
                continue
            
            values = []
            errors = []
            for metric, mult in zip(metrics, multipliers):
                mean, ci_hw, _, _ = calculate_95_ci(model_data[metric].tolist())
                values.append(mean * mult)
                errors.append(ci_hw * mult)
            
            # Normalize for display (tokens are much larger)
            # Show time and cost on primary axis, tokens normalized
            display_vals = [values[0], values[1], values[2] / 100]  # tokens / 100 for scale
            display_errs = [errors[0], errors[1], errors[2] / 100]
            
            bars = ax.bar(x_pos + i * bar_width, display_vals, bar_width, yerr=display_errs,
                         label=model, color=model_colors.get(model, 'gray'),
                         alpha=0.8, edgecolor='black', capsize=4)
            
            # Add value labels
            for j, (bar, val, orig_val) in enumerate(zip(bars, display_vals, values)):
                if j == 2:  # tokens
                    label = f'{orig_val:.0f}'
                elif j == 1:  # cost
                    label = f'{orig_val:.2f}¢'
                else:  # time
                    label = f'{orig_val:.1f}s'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + display_errs[j] + 0.5,
                       label, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Metric', fontsize=11)
        ax.set_ylabel('Value (tokens ÷100 for scale)', fontsize=10)
        ax.set_title('(D) Model Comparison: Citation Judging', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos + bar_width / 2)
        ax.set_xticklabels(['Time\n(s/edge)', 'Cost\n(¢/edge)', 'Tokens\n(÷100)'])
        ax.legend(title='Model', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Model comparison requires\nboth gpt-4.1 and gpt-5-mini data', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('(D) Model Comparison', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure2_configuration_comparison.png', dpi=300)
    plt.close()
    
    # ==================================================================================
    # Save scaling analysis summary to text file
    # ==================================================================================
    print("  Creating: scaling_analysis_summary.txt")
    with open(figures_dir.parent / 'scaling_analysis_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SCALING ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("THEORETICAL EXPECTATION\n")
        f.write("-" * 40 + "\n")
        f.write("Time complexity: O(E) where E = number of edges\n")
        f.write("  - Each edge is judged independently by the LLM\n")
        f.write("  - Total time = (constant time per edge) × E\n\n")
        f.write("For dense CLDs (edge density ≈ 100%):\n")
        f.write("  - E = N × (N-1) ≈ N² where N = nodes\n")
        f.write("  - Therefore: Time = O(E) = O(N²) w.r.t. nodes\n\n")
        f.write("Verification criteria:\n")
        f.write("  - R² close to 1.0 indicates good linear fit (Time vs E)\n")
        f.write("  - Constant time_per_edge across different E values\n")
        f.write("  - Low coefficient of variation (CV) for per-edge metrics\n\n")
        
        for judge_type, metrics in scaling_results.items():
            f.write(f"\n{'='*40}\n")
            f.write(f"Judge Type: {judge_type.upper()}\n")
            f.write(f"{'='*40}\n\n")
            
            for metric, data in metrics.items():
                metric_name = metric.replace('judge_', '').replace('_', ' ').title()
                f.write(f"{metric_name}:\n")
                f.write(f"  Linear fit: y = {data['slope']:.4f}×E + {data['intercept']:.4f}\n")
                f.write(f"  R² = {data['r_squared']:.4f} (p = {data['p_value']:.2e})\n")
                f.write(f"  Per-edge mean: {data['per_edge_mean']:.4f}\n")
                f.write(f"  Per-edge std:  {data['per_edge_std']:.4f}\n")
                f.write(f"  Coefficient of variation: {data['per_edge_cv']:.2%}\n")
                f.write(f"  Interpretation: {'Good O(E) scaling' if data['r_squared'] > 0.9 else 'Some deviation from O(E)'}\n")
                f.write("\n")
    
    print(f"\n  Figures saved to: {figures_dir}")


def generate_latex_tables(df: pd.DataFrame, aggregates: dict, output_dir: Path):
    """Generate LaTeX tables for thesis integration."""
    latex_dir = output_dir / 'latex'
    latex_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING LATEX TABLES")
    print("=" * 80)
    
    # Table 0: Scaling Analysis Summary
    print("  Creating: table_scaling_analysis.tex")
    scaling_results = compute_scaling_analysis(df)
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Scaling Analysis: Linear Fit to O(n) Theoretical Expectation}")
    lines.append("\\label{tab:scaling_analysis}")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Judge Type} & \\textbf{Metric} & \\textbf{Slope} & \\textbf{R²} & \\textbf{Per-Edge Mean} & \\textbf{CV (\\%)} & \\textbf{O(n)?} \\\\")
    lines.append("\\midrule")
    
    metric_display = {
        'judge_inference_time_s': 'Time (s)',
        'judge_total_tokens': 'Tokens',
        'estimated_cost_usd': 'Cost (\\$)',
    }
    
    for judge_type in ['correctness', 'citation']:
        if judge_type not in scaling_results:
            continue
        
        first = True
        for metric, data in scaling_results[judge_type].items():
            if metric not in metric_display:
                continue
            
            jt_display = judge_type.title() if first else ''
            metric_name = metric_display[metric]
            slope = f"{data['slope']:.4f}"
            r2 = f"{data['r_squared']:.3f}"
            per_edge = f"{data['per_edge_mean']:.3f}" if data['per_edge_mean'] < 100 else f"{data['per_edge_mean']:.0f}"
            cv = f"{data['per_edge_cv']*100:.1f}"
            on = "\\checkmark" if data['r_squared'] > 0.9 else "$\\sim$"
            
            lines.append(f"{jt_display} & {metric_name} & {slope} & {r2} & {per_edge} & {cv} & {on} \\\\")
            first = False
        
        lines.append("\\midrule")
    
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2mm}")
    lines.append("\\footnotesize{CV = Coefficient of Variation. O(n)? = \\checkmark if R² > 0.9, indicating good linear scaling.}")
    lines.append("\\end{table}")
    
    with open(latex_dir / 'table_scaling_analysis.tex', 'w') as f:
        f.write('\n'.join(lines))
    
    # Table 1: Time and Cost by Judge Type × Prompt Variant
    print("  Creating: table_time_cost_by_config.tex")
    
    agg = aggregates['by_judge_prompt'].copy()
    if len(agg) > 0:
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Inference Time and Cost by Judge Type and Prompt Variant (mean (± 95% CI))}")
        lines.append("\\label{tab:time_cost_by_config}")
        lines.append("\\begin{tabular}{llrrrrr}")
        lines.append("\\toprule")
        lines.append("\\textbf{Judge Type} & \\textbf{Prompt} & \\textbf{N} & \\textbf{Time (s)} & \\textbf{Time/Edge (s)} & \\textbf{Cost (\\$)} & \\textbf{Tokens/Edge} \\\\")
        lines.append("\\midrule")
        
        for judge_type in ['correctness', 'citation']:
            type_data = agg[agg['judge_type'] == judge_type].sort_values('prompt_variant')
            for i, row in type_data.iterrows():
                jt = row['judge_type'].title() if i == type_data.index[0] else ''
                prompt = row['prompt_variant'].title()
                n = int(row['n_runs'])
                
                time_str = f"${row['judge_inference_time_s_mean']:.1f} \\pm {row['judge_inference_time_s_ci']:.1f}$"
                tpe_str = f"${row['time_per_edge_mean']:.2f} \\pm {row['time_per_edge_ci']:.2f}$"
                cost_str = f"${row['estimated_cost_usd_mean']:.3f} \\pm {row['estimated_cost_usd_ci']:.3f}$"
                tokens_str = f"${row['tokens_per_edge_mean']:.0f} \\pm {row['tokens_per_edge_ci']:.0f}$"
                
                lines.append(f"{jt} & {prompt} & {n} & {time_str} & {tpe_str} & {cost_str} & {tokens_str} \\\\")
            
            if judge_type == 'correctness':
                lines.append("\\midrule")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(latex_dir / 'table_time_cost_by_config.tex', 'w') as f:
            f.write('\n'.join(lines))
    
    # Table 2: Scaling by CLD
    print("  Creating: table_scaling_by_cld.tex")
    
    agg_cld = aggregates['by_cld_scaling'].copy()
    if len(agg_cld) > 0:
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Time and Cost Scaling by CLD Size (mean (± 95% CI))}")
        lines.append("\\label{tab:scaling_by_cld}")
        lines.append("\\begin{tabular}{llrrrr}")
        lines.append("\\toprule")
        lines.append("\\textbf{CLD} & \\textbf{Edges} & \\textbf{Judge Type} & \\textbf{Time (s)} & \\textbf{Time/Edge (s)} & \\textbf{Cost (\\$)} \\\\")
        lines.append("\\midrule")
        
        # Group by CLD size
        for cld in ['social_norms', 'depressive', 'emergency_department']:
            cld_data = agg_cld[agg_cld['cld'] == cld]
            if len(cld_data) == 0:
                continue
            
            first_row = True
            for _, row in cld_data.iterrows():
                cld_name = CLD_DISPLAY_NAMES.get(cld, cld) if first_row else ''
                edges = row['n_edges_typical'] if first_row else ''
                jt = row['judge_type'].title()
                
                time_str = f"${row['judge_inference_time_s_mean']:.1f} \\pm {row['judge_inference_time_s_ci']:.1f}$"
                tpe_str = f"${row['time_per_edge_mean']:.2f} \\pm {row['time_per_edge_ci']:.2f}$"
                cost_str = f"${row['estimated_cost_usd_mean']:.3f} \\pm {row['estimated_cost_usd_ci']:.3f}$"
                
                lines.append(f"{cld_name} & {edges} & {jt} & {time_str} & {tpe_str} & {cost_str} \\\\")
                first_row = False
            
            lines.append("\\midrule")
        
        lines[-1] = "\\bottomrule"  # Replace last midrule
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(latex_dir / 'table_scaling_by_cld.tex', 'w') as f:
            f.write('\n'.join(lines))
    
    # Table 3: Model Comparison
    print("  Creating: table_model_comparison.tex")
    
    agg_model = aggregates['by_model'].copy()
    if len(agg_model) > 0:
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Model Efficiency Comparison (mean (± 95% CI))}")
        lines.append("\\label{tab:model_comparison}")
        lines.append("\\begin{tabular}{llrrrrr}")
        lines.append("\\toprule")
        lines.append("\\textbf{Model} & \\textbf{Judge Type} & \\textbf{N} & \\textbf{Time/Edge (s)} & \\textbf{Cost/Edge (\\$)} & \\textbf{Tokens/Edge} \\\\")
        lines.append("\\midrule")
        
        for _, row in agg_model.iterrows():
            model = row['judge_model']
            jt = row['judge_type'].title()
            n = int(row['n_runs'])
            
            tpe_str = f"${row['time_per_edge_mean']:.2f} \\pm {row['time_per_edge_ci']:.2f}$"
            cpe_str = f"${row['cost_per_edge_mean']:.5f} \\pm {row['cost_per_edge_ci']:.5f}$"
            tokens_str = f"${row['tokens_per_edge_mean']:.0f} \\pm {row['tokens_per_edge_ci']:.0f}$"
            
            lines.append(f"{model} & {jt} & {n} & {tpe_str} & {cpe_str} & {tokens_str} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(latex_dir / 'table_model_comparison.tex', 'w') as f:
            f.write('\n'.join(lines))
    
    print(f"\n  LaTeX tables saved to: {latex_dir}")


def save_excel_output(df: pd.DataFrame, aggregates: dict, output_path: Path):
    """Save all data to Excel file."""
    print("\n" + "=" * 80)
    print("SAVING EXCEL OUTPUT")
    print("=" * 80)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Raw data
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Aggregations
        for sheet_name, agg_df in aggregates.items():
            # Truncate sheet name if needed
            sheet_name = sheet_name[:31]
            agg_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"  Excel saved to: {output_path}")


def save_analysis_metadata(base_dir: Path, output_dir: Path, df: pd.DataFrame, timestamp: str):
    """Save metadata file documenting data sources and analysis script."""
    import shutil
    
    print("\n" + "=" * 80)
    print("SAVING ANALYSIS METADATA")
    print("=" * 80)
    
    # 1. Copy the analysis script to output directory (skip if same file)
    script_path = Path(__file__).resolve()
    script_copy_path = output_dir / script_path.name
    if script_path != script_copy_path:
        shutil.copy2(script_path, script_copy_path)
        print(f"  Script copied to: {script_copy_path}")
    else:
        print(f"  Script already in output directory: {script_copy_path}")
    
    # 2. Create metadata file
    metadata_path = output_dir / 'ANALYSIS_METADATA.txt'
    
    # Collect unique file paths from the data
    unique_files = df['file_path'].unique().tolist() if 'file_path' in df.columns else []
    unique_experiments = df['experiment_folder'].unique().tolist() if 'experiment_folder' in df.columns else []
    
    metadata_lines = [
        "=" * 80,
        "TIME AND COST SCALING ANALYSIS - METADATA",
        "=" * 80,
        "",
        f"Analysis Timestamp: {timestamp}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 80,
        "INPUT DATA SOURCES",
        "=" * 80,
        "",
        f"Base Directory: {base_dir.resolve()}",
        "",
        "Experiment Folders Processed:",
    ]
    
    for exp in sorted(unique_experiments):
        metadata_lines.append(f"  - {exp}")
    
    metadata_lines.extend([
        "",
        f"Total Files Processed: {len(unique_files)}",
        "",
        "=" * 80,
        "DATA SUMMARY",
        "=" * 80,
        "",
        f"Total Data Points: {len(df)}",
        f"Unique CLDs: {df['cld'].nunique() if 'cld' in df.columns else 'N/A'}",
        f"Unique Prompt Variants: {df['prompt_variant'].nunique() if 'prompt_variant' in df.columns else 'N/A'}",
        f"Unique Judge Types: {df['judge_type'].nunique() if 'judge_type' in df.columns else 'N/A'}",
        f"Unique Judge Models: {df['judge_model'].nunique() if 'judge_model' in df.columns else 'N/A'}",
        "",
        "=" * 80,
        "OUTPUT FILES",
        "=" * 80,
        "",
        f"Output Directory: {output_dir.resolve()}",
        "",
        "Generated Files:",
        f"  - {script_path.name} (analysis script copy)",
        "  - ANALYSIS_METADATA.txt (this file)",
        "  - time_cost_scaling_*.xlsx (data and aggregations)",
        "  - figures/ (scaling curves and comparisons)",
        "  - latex/ (thesis-ready tables)",
        "",
        "=" * 80,
        "INDIVIDUAL INPUT FILES",
        "=" * 80,
        "",
    ])
    
    for f in sorted(unique_files):
        metadata_lines.append(f"  {f}")
    
    metadata_lines.extend([
        "",
        "=" * 80,
        "END OF METADATA",
        "=" * 80,
    ])
    
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata_lines))
    
    print(f"  Metadata saved to: {metadata_path}")


def collect_generation_data(gen_base_dir: Path) -> pd.DataFrame:
    """
    Collect CLD generation timing data from parameter_tuning_experiments.
    
    Returns DataFrame with generation metrics.
    """
    print("\n" + "=" * 80)
    print("COLLECTING GENERATION DATA")
    print("=" * 80)
    
    gen_data = []
    
    for exp_dir in gen_base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if 'generation' not in exp_dir.name.lower():
            continue
        
        for xlsx_file in exp_dir.glob("*.xlsx"):
            try:
                xl = pd.ExcelFile(xlsx_file)
                if 'LLM Usage Stats' not in xl.sheet_names:
                    continue
                
                df_llm = pd.read_excel(xlsx_file, sheet_name='LLM Usage Stats')
                df_edges = pd.read_excel(xlsx_file, sheet_name='All Edges')
                
                # Get generator stats
                gen_row = df_llm[df_llm['Role'] == 'Generator']
                if len(gen_row) == 0:
                    continue
                gen_row = gen_row.iloc[0]
                
                if gen_row['Inference Time (s)'] <= 0:
                    continue
                
                tokens = eval(str(gen_row['Token Totals']))
                
                # Count unique nodes
                sources = set(df_edges['Source'].dropna().unique())
                targets = set(df_edges['Target'].dropna().unique())
                n_nodes = len(sources.union(targets))
                
                gen_data.append({
                    'experiment': exp_dir.name,
                    'file': xlsx_file.name,
                    'n_edges': len(df_edges),
                    'n_nodes': n_nodes,
                    'gen_inference_time_s': gen_row['Inference Time (s)'],
                    'gen_inference_calls': gen_row['Inference Calls'],
                    'gen_prompt_tokens': tokens.get('prompt_tokens', 0),
                    'gen_completion_tokens': tokens.get('completion_tokens', 0),
                    'gen_total_tokens': tokens.get('total_tokens', 0),
                })
            except Exception as e:
                pass
    
    df = pd.DataFrame(gen_data)
    
    if len(df) > 0:
        # Calculate derived metrics
        df['gen_time_per_edge'] = df['gen_inference_time_s'] / df['n_edges']
        df['gen_tokens_per_edge'] = df['gen_total_tokens'] / df['n_edges']
        
        # Estimate cost (using gpt-4.1 pricing)
        df['gen_estimated_cost_usd'] = (
            df['gen_prompt_tokens'] * PRICING['gpt-4.1']['input'] +
            df['gen_completion_tokens'] * PRICING['gpt-4.1']['output']
        )
        df['gen_cost_per_edge'] = df['gen_estimated_cost_usd'] / df['n_edges']
        
        print(f"  Found {len(df)} generation files")
        print(f"  Edge counts: {sorted(df['n_edges'].unique())}")
    else:
        print("  No generation data found")
    
    return df


def collect_deep_research_timing_data(data_science_dir: Path) -> list:
    """
    Collect Deep Research timing data from result JSON files.
    
    Uses multiple sources:
    1. The big_run file with all 3 CLDs (107 edges with timing)
    2. The Older Persons ALL_EDGES file (184 edges with timing)
    
    Returns list of dicts with: cld, n_edges, total_time_s, time_per_edge_s
    """
    print("\n" + "=" * 80)
    print("COLLECTING DEEP RESEARCH TIMING DATA")
    print("=" * 80)
    
    # CLD name mappings for display
    CLD_NAME_MAP = {
        'Social_norms_and_obesity_prevalence': 'Social Norms',
        'Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults': 'Depressive',
        'older_persons_emergency_department_visits_and_interactionstitled_spreadsheet': 'Older Persons',
    }
    
    # Files with timing data
    DR_FILES = [
        "deep_research_results_20251012_080426_all_3_CLDs_107edges_big_run.json",  # Has all 3 CLDs
        "deep_research_results_older_persons_ALL_EDGES_184edges.json",  # Full Older Persons run
    ]
    
    # Collect timing data per CLD
    cld_data = {}  # cld_name -> {'edge_times': [], 'sources': set()}
    
    for filename in DR_FILES:
        filepath = data_science_dir / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found, skipping")
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
            elif isinstance(data, list):
                results = data
            else:
                results = data.get('results', [])
            
            edges_found = 0
            for result in results:
                duration_str = result.get('duration', '')
                if duration_str:
                    try:
                        duration_s = float(duration_str.rstrip('s'))
                        cld_raw = result.get('CLD', 'unknown')
                        cld_name = CLD_NAME_MAP.get(cld_raw, cld_raw)
                        
                        if cld_name not in cld_data:
                            cld_data[cld_name] = {'edge_times': [], 'sources': set()}
                        cld_data[cld_name]['edge_times'].append(duration_s)
                        cld_data[cld_name]['sources'].add(filename)
                        edges_found += 1
                    except (ValueError, AttributeError):
                        pass
            
            print(f"  Loaded {edges_found} edges with timing from {filename}")
        
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")
    
    # Build output list
    dr_timing_data = []
    for cld_name, data in cld_data.items():
        edge_times = data['edge_times']
        if edge_times:
            n_edges = len(edge_times)
            total_time = sum(edge_times)
            time_per_edge = total_time / n_edges
            
            dr_timing_data.append({
                'cld': cld_name,
                'n_edges': n_edges,
                'total_time_s': total_time,
                'time_per_edge_s': time_per_edge,
                'edge_times': edge_times,
            })
            
            print(f"  {cld_name}: {n_edges} edges, total={total_time:.0f}s, avg={time_per_edge:.1f}s/edge")
    
    if dr_timing_data:
        # Compute aggregate stats
        total_edges = sum(d['n_edges'] for d in dr_timing_data)
        total_time = sum(d['total_time_s'] for d in dr_timing_data)
        all_edge_times = []
        for d in dr_timing_data:
            all_edge_times.extend(d['edge_times'])
        avg_time_per_edge = np.mean(all_edge_times) if all_edge_times else 0
        std_time_per_edge = np.std(all_edge_times) if all_edge_times else 0
        
        print(f"\n  AGGREGATE: {total_edges} edges, total={total_time:.0f}s")
        print(f"  Per-edge: {avg_time_per_edge:.1f} +/- {std_time_per_edge:.1f}s")
    else:
        print("  No Deep Research timing data found")
    
    return dr_timing_data


def generate_combined_scaling_figure(df_judge: pd.DataFrame, df_gen: pd.DataFrame, output_dir: Path, 
                                     dr_timing_data: list = None):
    """Generate combined scaling figure showing generation, judging, correction, and Deep Research."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("  Creating: figure3_generation_vs_judging.png")
    
    from scipy import stats as scipy_stats
    
    # Load corrector timing data if available
    corrector_timing_file = output_dir / 'corrector_timing_data.json'
    corrector_data = []
    if corrector_timing_file.exists():
        with open(corrector_timing_file, 'r') as f:
            corrector_json = json.load(f)
            corrector_data = corrector_json.get('results', [])
        print(f"    Loaded {len(corrector_data)} corrector timing records")
    
    # Use provided DR timing data or empty list
    if dr_timing_data is None:
        dr_timing_data = []
    if dr_timing_data:
        print(f"    Using {len(dr_timing_data)} Deep Research timing records")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Time vs Edges - Generation + Judging comparison
    ax = axes[0]
    
    # Generation data (aggregate by n_edges)
    if len(df_gen) > 0:
        gen_agg = df_gen.groupby('n_edges').agg({
            'gen_inference_time_s': ['mean', 'std', 'count']
        }).reset_index()
        gen_agg.columns = ['n_edges', 'time_mean', 'time_std', 'n']
        gen_agg['time_ci'] = gen_agg.apply(
            lambda row: scipy_stats.t.ppf(0.975, max(row['n']-1, 1)) * row['time_std'] / np.sqrt(row['n']) 
            if row['n'] > 1 else 0, axis=1)
        
        ax.errorbar(gen_agg['n_edges'], gen_agg['time_mean'], yerr=gen_agg['time_ci'],
                   fmt='o', markersize=10, capsize=5, capthick=2,
                   color='#9b59b6', ecolor='#9b59b6', label='Generation',
                   markeredgecolor='black', markeredgewidth=1)
        
        # Linear fit for generation
        if len(gen_agg) >= 2:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(gen_agg['n_edges'], gen_agg['time_mean'])
            x_line = np.linspace(0, gen_agg['n_edges'].max() * 1.1, 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color='#9b59b6', lw=2,
                   label=f'Gen fit: {slope:.2f}E (R²={r_value**2:.3f})')
    
    # Judging data (correctness)
    judge_corr = df_judge[df_judge['judge_type'] == 'correctness']
    if len(judge_corr) > 0:
        judge_agg = judge_corr.groupby('n_edges').agg({
            'judge_inference_time_s': ['mean', 'std', 'count']
        }).reset_index()
        judge_agg.columns = ['n_edges', 'time_mean', 'time_std', 'n']
        judge_agg['time_ci'] = judge_agg.apply(
            lambda row: scipy_stats.t.ppf(0.975, max(row['n']-1, 1)) * row['time_std'] / np.sqrt(row['n']) 
            if row['n'] > 1 else 0, axis=1)
        
        ax.errorbar(judge_agg['n_edges'], judge_agg['time_mean'], yerr=judge_agg['time_ci'],
                   fmt='s', markersize=10, capsize=5, capthick=2,
                   color='#e74c3c', ecolor='#e74c3c', label='Judging (Correctness)',
                   markeredgecolor='black', markeredgewidth=1)
        
        # Linear fit for correctness judging
        if len(judge_agg) >= 2:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(judge_agg['n_edges'], judge_agg['time_mean'])
            x_line = np.linspace(0, judge_agg['n_edges'].max() * 1.1, 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color='#e74c3c', lw=2,
                   label=f'Corr fit: {slope:.2f}E (R²={r_value**2:.3f})')
    
    # Judging data (citation)
    judge_cit = df_judge[df_judge['judge_type'] == 'citation']
    if len(judge_cit) > 0:
        judge_cit_agg = judge_cit.groupby('n_edges').agg({
            'judge_inference_time_s': ['mean', 'std', 'count']
        }).reset_index()
        judge_cit_agg.columns = ['n_edges', 'time_mean', 'time_std', 'n']
        judge_cit_agg['time_ci'] = judge_cit_agg.apply(
            lambda row: scipy_stats.t.ppf(0.975, max(row['n']-1, 1)) * row['time_std'] / np.sqrt(row['n']) 
            if row['n'] > 1 else 0, axis=1)
        
        ax.errorbar(judge_cit_agg['n_edges'], judge_cit_agg['time_mean'], yerr=judge_cit_agg['time_ci'],
                   fmt='^', markersize=10, capsize=5, capthick=2,
                   color='#3498db', ecolor='#3498db', label='Judging (Citation)',
                   markeredgecolor='black', markeredgewidth=1)
        
        # Linear fit for citation judging
        if len(judge_cit_agg) >= 2:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(judge_cit_agg['n_edges'], judge_cit_agg['time_mean'])
            x_line = np.linspace(0, judge_cit_agg['n_edges'].max() * 1.1, 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color='#3498db', lw=2,
                   label=f'Cit fit: {slope:.2f}E (R²={r_value**2:.3f})')
    
    # Corrector data (from JSON)
    if len(corrector_data) > 0:
        corr_edges = [r['n_edges'] for r in corrector_data]
        corr_times = [r['corrector_time_s'] for r in corrector_data]
        
        ax.scatter(corr_edges, corr_times,
                   marker='D', s=100, 
                   color='#27ae60', edgecolor='black', linewidth=1,
                   label='Corrector', zorder=5)
        
        # Linear fit for corrector
        if len(corrector_data) >= 2:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(corr_edges, corr_times)
            x_line = np.linspace(0, max(corr_edges) * 1.1, 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color='#27ae60', lw=2,
                   label=f'Corr fit: {slope:.2f}E (R²={r_value**2:.3f})')
    
    # Deep Research timing data
    if len(dr_timing_data) > 0:
        dr_edges = [r['n_edges'] for r in dr_timing_data]
        dr_times = [r['total_time_s'] for r in dr_timing_data]
        
        ax.scatter(dr_edges, dr_times,
                   marker='p', s=150,  # Pentagon marker, larger for visibility
                   color='#e67e22', edgecolor='black', linewidth=1.5,
                   label='Deep Research', zorder=6)
        
        # Linear fit for DR
        if len(dr_timing_data) >= 2:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(dr_edges, dr_times)
            x_line = np.linspace(0, max(dr_edges) * 1.1, 100)
            ax.plot(x_line, slope * x_line + intercept, '--', color='#e67e22', lw=2,
                   label=f'DR fit: {slope:.1f}E (R²={r_value**2:.3f})')
    
    ax.set_xlabel('Number of Edges (E)', fontsize=12)
    ax.set_ylabel('Inference Time (s) (± 95% CI) [log scale]', fontsize=12)
    ax.set_title('Pipeline Stage Time Scaling\nAll stages show O(E) linear scaling', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1)
    
    # Panel 2: Time per Edge comparison (bar chart)
    ax = axes[1]
    
    categories = []
    means = []
    cis = []
    colors = []
    
    if len(df_gen) > 0:
        gen_tpe = df_gen['gen_time_per_edge']
        mean, ci_hw, _, _ = calculate_95_ci(gen_tpe.tolist())
        categories.append('Generation')
        means.append(mean)
        cis.append(ci_hw)
        colors.append('#9b59b6')
    
    for jt in ['correctness', 'citation']:
        subset = df_judge[df_judge['judge_type'] == jt]
        if len(subset) > 0:
            tpe = subset['time_per_edge']
            mean, ci_hw, _, _ = calculate_95_ci(tpe.tolist())
            categories.append(f'Judging ({jt.title()})')
            means.append(mean)
            cis.append(ci_hw)
            colors.append('#e74c3c' if jt == 'correctness' else '#3498db')
    
    # Add corrector time per edge
    if len(corrector_data) > 0:
        corr_tpe = [r['corrector_time_s'] / r['n_edges'] for r in corrector_data if r['n_edges'] > 0]
        if corr_tpe:
            mean, ci_hw, _, _ = calculate_95_ci(corr_tpe)
            categories.append('Corrector')
            means.append(mean)
            cis.append(ci_hw)
            colors.append('#27ae60')
    
    # Add Deep Research time per edge
    if len(dr_timing_data) > 0:
        # Collect all individual edge times for proper stats
        all_dr_edge_times = []
        for d in dr_timing_data:
            all_dr_edge_times.extend(d.get('edge_times', []))
        if all_dr_edge_times:
            mean, ci_hw, _, _ = calculate_95_ci(all_dr_edge_times)
            categories.append('Deep Research')
            means.append(mean)
            cis.append(ci_hw)
            colors.append('#e67e22')
    
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=cis, capsize=8, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels - format appropriately for scale
    for bar, mean, ci in zip(bars, means, cis):
        # For very large values (DR), format differently
        if mean > 100:
            label_text = f'{mean:.0f}'
        else:
            label_text = f'{mean:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + max(means)*0.02,
               label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Time per Edge (s) (± 95% CI)', fontsize=12)
    ax.set_title('Per-Edge Time Comparison\n(Constant for O(E) scaling)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Use log scale due to large DR times
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure3_generation_vs_judging.png', dpi=300)
    plt.close()


def generate_expert_vs_llm_generation_figure(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    minutes_per_pair: float = 5.0,
    n_experts: int = 3,
    directed_pairs: bool = True,
    # Kept for backward-compatibility with the CLI; not used in the expert-time-only figure.
    llm_seconds_per_pair: float = 2.22,
    llm_valid_fraction: float = 0.5,
    minutes_per_pair_light_validation: float = 1.0,
):
    """
    Create a thesis-ready figure comparing expert elicitation time vs LLM CLD generation time.
    
    Assumptions (expert time):
      - All variable-pairs are discussed
      - Time per pair per expert = minutes_per_pair
      - Total expert-minutes = pairs × minutes_per_pair × n_experts
      - pairs = N*(N-1) if directed_pairs else N*(N-1)/2
    LLM time is measured from observed generator inference-time in the Excel files.
    """
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    required_cols = {'n_nodes', 'generator_inference_time_s'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Warning: Cannot generate expert-vs-LLM figure (missing columns: {sorted(required_cols - set(df.columns))})")
        return
    
    df_use = df[(df['n_nodes'] > 0) & (df['generator_inference_time_s'] > 0)].copy()
    
    # Aggregate LLM generation time by number of nodes (if observed generator timings exist)
    from scipy import stats as scipy_stats
    agg = pd.DataFrame()
    if len(df_use) > 0:
        agg = df_use.groupby('n_nodes').agg({'generator_inference_time_s': ['mean', 'std', 'count']}).reset_index()
        agg.columns = ['n_nodes', 'time_mean_s', 'time_std_s', 'n']
        agg['time_ci_s'] = agg.apply(
            lambda row: scipy_stats.t.ppf(0.975, max(row['n'] - 1, 1)) * row['time_std_s'] / np.sqrt(row['n'])
            if row['n'] > 1 else 0,
            axis=1
        )
    
    # Expert-time curve across a wider N range for readability
    if len(agg) > 0:
        n_min = max(3, int(min(agg['n_nodes'].min(), 10)))
        n_max = int(max(agg['n_nodes'].max(), 50))
    else:
        n_min, n_max = 3, 50
    N = np.arange(n_min, n_max + 1)
    if directed_pairs:
        pairs = N * (N - 1)
        pairs_label = r"Pairs $= N(N-1)$ (directed)"
    else:
        pairs = N * (N - 1) / 2
        pairs_label = r"Pairs $= N(N-1)/2$ (undirected)"
    
    # -----------------------------
    # Expert-time-only model
    # -----------------------------
    # Baseline: full expert discussion per candidate pair.
    # LLM-assisted: assume a fraction of candidate relationships are correctly generated by the LLM,
    # so they only require light expert validation (minutes_per_pair_light_validation).
    llm_valid_fraction = float(llm_valid_fraction)
    llm_valid_fraction = 0.0 if llm_valid_fraction < 0 else (1.0 if llm_valid_fraction > 1 else llm_valid_fraction)
    minutes_per_pair_light_validation = float(minutes_per_pair_light_validation)
    if minutes_per_pair_light_validation < 0:
        minutes_per_pair_light_validation = 0.0
    
    expert_hours_baseline = (pairs * minutes_per_pair * n_experts) / 60.0
    effective_minutes_per_pair_assisted = (
        (1.0 - llm_valid_fraction) * minutes_per_pair
        + llm_valid_fraction * minutes_per_pair_light_validation
    )
    expert_hours_llm_assisted = (pairs * effective_minutes_per_pair_assisted * n_experts) / 60.0
    
    # Plot (single y-axis; linear scale; hours)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    
    ax.plot(
        N,
        expert_hours_baseline,
        color='#e67e22',
        lw=3,
        label='Baseline expert review'
    )
    
    ax.plot(
        N,
        expert_hours_llm_assisted,
        color='#27ae60',
        lw=3.0,
        linestyle='--',
        label='LLM-assisted expert review'
    )
    
    # Shade savings
    ax.fill_between(
        N,
        expert_hours_llm_assisted,
        expert_hours_baseline,
        color='#27ae60',
        alpha=0.15,
        label='Saved expert time'
    )
    
    # Mark the observed N values (from judged Excel files) for visual anchoring
    try:
        observed_N = sorted({int(x) for x in df['n_nodes'].dropna().unique().tolist() if int(x) > 0})
        # Keep only a handful (usually 3 CLDs)
        observed_N = [n for n in observed_N if n_min <= n <= n_max][:8]
        if observed_N:
            obs_pairs = (np.array(observed_N) * (np.array(observed_N) - 1)) if directed_pairs else (np.array(observed_N) * (np.array(observed_N) - 1) / 2)
            obs_baseline_h = (obs_pairs * minutes_per_pair * n_experts) / 60.0
            obs_assisted_h = (obs_pairs * effective_minutes_per_pair_assisted * n_experts) / 60.0
            obs_saved_h = obs_baseline_h - obs_assisted_h
            
            ax.scatter(observed_N, obs_baseline_h, s=65, color='#e67e22', edgecolor='black', linewidth=0.8, zorder=5)
            ax.scatter(observed_N, obs_assisted_h, s=65, color='#27ae60', edgecolor='black', linewidth=0.8, zorder=6)
            
            # Hour labels removed for cleaner presentation
            pass
    except Exception:
        pass
    
    ax.set_xlabel('Number of variables (nodes) in CLD, N', fontweight='bold')
    ax.set_ylabel('Time (hours)', fontweight='bold')
    ax.set_title(
        'Theoretical Expert Review Burden vs CLD Size',
        fontweight='bold'
    )
    ax.grid(True, alpha=0.25)
    
    # Keep bounds tight and consistent
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(0, float(expert_hours_baseline.max()) * 1.05)
    
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    out_png = figures_dir / 'figure4_expert_time_vs_llm_generation.png'
    out_pdf = figures_dir / 'figure4_expert_time_vs_llm_generation.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate time and cost metrics from RQ1 Excel files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python aggregate_time_cost_scaling.py \\
        --base_dir /path/to/RQ1_excel_snapshot_20251215/final_runs
    
    python aggregate_time_cost_scaling.py \\
        --base_dir ./final_runs \\
        --output_dir ./analysis_output \\
        --latex
        """
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing final_runs experiment folders'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: creates timestamped dir in base_dir)'
    )
    
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Generate LaTeX tables for thesis'
    )
    
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Skip figure generation'
    )
    
    parser.add_argument(
        '--gen-dir',
        type=str,
        default=None,
        help='Directory containing generation experiments (default: auto-detect from workspace)'
    )
    
    parser.add_argument(
        '--expert-figure',
        action='store_true',
        help='Only generate expert-vs-LLM generation time figure (no Excel/metadata/other figures).'
    )
    parser.add_argument('--expert-minutes-per-pair', type=float, default=5.0, help='Expert minutes per pair per expert (default: 5).')
    parser.add_argument('--expert-n-experts', type=int, default=3, help='Number of experts (default: 3).')
    parser.add_argument('--expert-undirected', action='store_true', help='Use undirected pair count N(N-1)/2 instead of directed N(N-1).')
    parser.add_argument('--llm-seconds-per-pair', type=float, default=2.22, help='LLM seconds per pair (default: 2.22; from prior analysis).')
    parser.add_argument('--llm-valid-fraction', type=float, default=0.5, help='Assumed fraction of candidate relationships validly generated by LLM (default: 0.5).')
    parser.add_argument('--expert-minutes-per-pair-light-validation', type=float, default=1.0, help='Minutes per pair per expert for light validation of LLM-generated relationships (default: 1.0).')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        sys.exit(1)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Default output directory: keep expert-figure runs local to this script folder to avoid cluttering snapshots
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.expert_figure:
        output_dir = Path(__file__).parent
    else:
        output_dir = base_dir / f'time_cost_analysis_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Header
    print("=" * 80)
    print("TIME AND COST SCALING AGGREGATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: Collect all data
    df = collect_all_data(base_dir)
    
    if len(df) == 0:
        print("\nError: No data collected! Check that the base_dir contains RQ1a experiment folders.")
        sys.exit(1)
    
    print(f"\n  Total data points collected: {len(df)}")
    print(f"  Unique experiments: {df['experiment_folder'].nunique()}")
    print(f"  Unique CLDs: {df['cld'].nunique()}")
    print(f"  Unique prompts: {df['prompt_variant'].nunique()}")
    print(f"  Unique judge types: {df['judge_type'].nunique()}")
    
    # Step 2: Calculate derived metrics
    df = calculate_derived_metrics(df)
    
    # Expert-only figure mode (minimal outputs)
    if args.expert_figure:
        print("\nGenerating expert-vs-LLM generation time figure...")
        generate_expert_vs_llm_generation_figure(
            df,
            output_dir,
            minutes_per_pair=args.expert_minutes_per_pair,
            n_experts=args.expert_n_experts,
            directed_pairs=not args.expert_undirected,
            llm_seconds_per_pair=args.llm_seconds_per_pair,
            llm_valid_fraction=args.llm_valid_fraction,
            minutes_per_pair_light_validation=args.expert_minutes_per_pair_light_validation,
        )
        print("\nDone.")
        return
    
    # Step 3: Aggregate by dimensions
    print("\n" + "=" * 80)
    print("AGGREGATING DATA")
    print("=" * 80)
    aggregates = aggregate_by_dimensions(df)
    
    # Step 4: Save Excel output
    excel_path = output_dir / f'time_cost_scaling_{timestamp}.xlsx'
    save_excel_output(df, aggregates, excel_path)
    
    # Step 5: Generate figures
    if not args.no_figures:
        generate_scaling_figures(df, aggregates, output_dir)
    
    # Step 5b: Collect generation data and generate combined figure
    df_gen = pd.DataFrame()
    if args.gen_dir:
        gen_dir = Path(args.gen_dir)
    else:
        # Auto-detect from workspace
        workspace_root = base_dir.parent if 'final_runs' in str(base_dir) else base_dir.parent.parent
        gen_dir = workspace_root / 'data_science' / 'parameter_tuning_experiments' / 'results'
    
    # Step 5c: Collect Deep Research timing data
    dr_timing_data = []
    data_science_dir = workspace_root / 'data_science'
    if data_science_dir.exists():
        dr_timing_data = collect_deep_research_timing_data(data_science_dir)
    
    if gen_dir.exists():
        df_gen = collect_generation_data(gen_dir)
        if len(df_gen) > 0 and not args.no_figures:
            generate_combined_scaling_figure(df, df_gen, output_dir, dr_timing_data=dr_timing_data)
    
    # Step 6: Generate LaTeX tables
    if args.latex:
        generate_latex_tables(df, aggregates, output_dir)
    
    # Step 7: Save analysis metadata and script copy
    save_analysis_metadata(base_dir, output_dir, df, timestamp)
    
    # Step 8: Save comprehensive statistics JSON (per uncertainty reporting rule)
    comprehensive_stats = {
        'timestamp': timestamp,
        'by_judge_type': {},
        'by_cld': {},
        'by_judge_prompt': {}
    }
    
    # Aggregate comprehensive stats by judge type
    for jt in df['judge_type'].unique():
        jt_data = df[df['judge_type'] == jt]
        time_values = jt_data['judge_inference_time_s'].dropna().tolist()
        stats_entry = {'time': compute_comprehensive_stats(time_values)}
        if 'total_cost_usd' in df.columns:
            cost_values = jt_data['total_cost_usd'].dropna().tolist()
            stats_entry['cost'] = compute_comprehensive_stats(cost_values)
        elif 'estimated_cost_usd' in df.columns:
            cost_values = jt_data['estimated_cost_usd'].dropna().tolist()
            stats_entry['cost'] = compute_comprehensive_stats(cost_values)
        comprehensive_stats['by_judge_type'][jt] = stats_entry
    
    # Aggregate by CLD
    for cld in df['cld'].unique():
        cld_data = df[df['cld'] == cld]
        time_values = cld_data['judge_inference_time_s'].dropna().tolist()
        comprehensive_stats['by_cld'][cld] = {
            'time': compute_comprehensive_stats(time_values)
        }
    
    # Save JSON
    json_path = output_dir / f'time_cost_comprehensive_stats_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(comprehensive_stats, f, indent=2, default=str)
    print(f"✓ Comprehensive stats saved to: {json_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - {excel_path.name}")
    print(f"  - aggregate_time_cost_scaling.py (script copy)")
    print(f"  - ANALYSIS_METADATA.txt")
    if not args.no_figures:
        print("  - figures/")
    if args.latex:
        print("  - latex/")
    print()
    
    # Print key summary stats
    print("KEY FINDINGS:")
    if len(aggregates['by_judge_type']) > 0:
        for _, row in aggregates['by_judge_type'].iterrows():
            jt = row['judge_type']
            time_mean = row['time_per_edge_mean']
            time_ci = row['time_per_edge_ci']
            cost_mean = row['cost_per_edge_mean']
            print(f"  {jt.title()} judging: {time_mean:.2f} ± {time_ci:.2f} s/edge, ${cost_mean:.5f}/edge")


if __name__ == "__main__":
    main()
