#!/usr/bin/env python3
"""
Batch Sensitivity Analysis for all RQ1a Sub-experiments

Runs sensitivity analysis for:
1. RQ1a_gt_synth_citation
2. RQ1a_gt_synth_correctness
3. RQ1a_gt_lit_citation
4. RQ1a_gt_lit_correctness

Creates aggregate comparison figure.

Reference for MPNet:
    Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020).
    "MPNet: Masked and Permuted Pre-training for Language Understanding."
    NeurIPS 2020. arXiv:2004.09297
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_science.logit_metrics import get_embedding_local


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS = {
    'corruption_citation': {
        'name': 'RQ1a_gt_synth_citation',
        'enhanced_analysis': 'enhanced_analysis_20251204_155831',
        'excel_name': 'rq1a_enhanced_results.xlsx',
        'sheet_name': 'Main (CLD x Prompt)',
        'prompt_col': 'Prompt',
        'f1_mean_col': 'F1 Mean',
        'f1_std_col': 'F1 Std',
        'prompt_type': 'citation',
        # Data prompts -> YAML prompt mapping
        'data_prompts': ['Baseline', 'CoT', 'Mechanistic'],
        'yaml_prompts': {
            'Baseline': ('prompts_citation_baseline.yaml', 'judgeCitation'),
            'CoT': ('prompts_citation_cot.yaml', 'judgeCitation'),
            'Mechanistic': ('prompts_citation_mechanistic.yaml', 'judgeCitation')
        }
    },
    'corruption_correctness': {
        'name': 'RQ1a_gt_synth_correctness',
        'enhanced_analysis': 'enhanced_analysis_20251117_024026',
        'excel_name': 'rq1a_enhanced_results.xlsx',
        'sheet_name': 'Main (CLD x Prompt)',
        'prompt_col': 'Prompt',
        'f1_mean_col': 'F1 Mean',
        'f1_std_col': 'F1 Std',
        'prompt_type': 'correctness',
        'data_prompts': ['Baseline', 'CoT', 'Mechanistic'],
        'yaml_prompts': {
            'Baseline': ('prompts_correctness_baseline.yaml', 'judgeCorrectness'),
            'CoT': ('prompts_correctness_cot.yaml', 'judgeCorrectness'),
            'Mechanistic': ('prompts_correctness_mechanistic.yaml', 'judgeCorrectness')
        }
    },
    'ground_truth_citation': {
        'name': 'RQ1a_gt_lit_citation',
        'enhanced_analysis': 'enhanced_analysis_20251125_013535',
        'excel_name': 'rq1a_ground_truth_enhanced_results.xlsx',
        'sheet_name': 0,  # First sheet
        'prompt_col': 'prompt',
        'f1_mean_col': 'f1_mean',
        'f1_std_col': 'f1_std',
        'prompt_type': 'citation',
        # Note: ground truth citation has 4 prompts including mechanistic_lit and mechanistic_original
        'data_prompts': ['baseline', 'cot', 'mechanistic_lit', 'mechanistic_original'],
        'yaml_prompts': {
            'baseline': ('prompts_citation_baseline.yaml', 'judgeCitation'),
            'cot': ('prompts_citation_cot.yaml', 'judgeCitation'),
            # mechanistic_lit and mechanistic_original both use the same mechanistic prompt
            'mechanistic_lit': ('prompts_citation_mechanistic.yaml', 'judgeCitation'),
            'mechanistic_original': ('prompts_citation_mechanistic.yaml', 'judgeCitation')
        }
    },
    'ground_truth_correctness': {
        'name': 'RQ1a_gt_lit_correctness',
        'enhanced_analysis': 'enhanced_analysis_20251120_174738',
        'excel_name': 'rq1a_ground_truth_enhanced_results.xlsx',
        'sheet_name': 0,
        'prompt_col': 'prompt',
        'f1_mean_col': 'f1_mean',
        'f1_std_col': 'f1_std',
        'prompt_type': 'correctness',
        'data_prompts': ['baseline', 'cot', 'mechanistic'],
        'yaml_prompts': {
            'baseline': ('prompts_correctness_baseline.yaml', 'judgeCorrectness'),
            'cot': ('prompts_correctness_cot.yaml', 'judgeCorrectness'),
            'mechanistic': ('prompts_correctness_mechanistic.yaml', 'judgeCorrectness')
        }
    }
}


# ============================================================================
# PROMPT EXTRACTION
# ============================================================================

def extract_prompt_from_yaml(yaml_path: Path, prompt_id: str = "judgeCitation") -> str:
    """Extract sys_prompt + usr_prompt from a YAML file (returns LAST matching prompt)."""
    if not yaml_path.exists():
        print(f"    WARNING: YAML file not found: {yaml_path}")
        return ""
    
    try:
        # Parse the YAML content - find all prompts sections
        with open(yaml_path, 'r') as f:
            full_yaml = yaml.safe_load(f)
        
        # Collect ALL matching prompts
        all_prompts = []
        if full_yaml and 'prompts' in full_yaml:
            for prompt in full_yaml['prompts']:
                if prompt.get('prompt_id') == prompt_id:
                    sys_prompt = prompt.get('prompts', {}).get('sys_prompt', '')
                    usr_prompt = prompt.get('prompts', {}).get('usr_prompt', '')
                    all_prompts.append((sys_prompt, usr_prompt))
        
        if all_prompts:
            # Return the LAST matching prompt (most recent version)
            sys_prompt, usr_prompt = all_prompts[-1]
            combined = f"SYSTEM: {sys_prompt.strip()}\n\nUSER: {usr_prompt.strip()}"
            return combined
        
    except yaml.YAMLError as e:
        print(f"    WARNING: YAML parsing error in {yaml_path.name}: {e}")
        # Try regex-based extraction as fallback
        return extract_prompt_regex(yaml_path, prompt_id)
    
    print(f"    WARNING: prompt_id '{prompt_id}' not found in {yaml_path.name}")
    return ""


def extract_prompt_regex(yaml_path: Path, prompt_id: str) -> str:
    """Fallback: Extract prompts using regex when YAML parsing fails."""
    import re
    
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Find the section for this prompt_id
    pattern = rf'prompt_id:\s*{prompt_id}\s*\n.*?sys_prompt:\s*\|([^|]+?)usr_prompt:\s*\|([^|]+?)(?:usr_prompt_extension|version)'
    
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if matches:
        # Get the last match
        match = matches[-1]
        sys_prompt = match.group(1).strip()
        usr_prompt = match.group(2).strip()
        
        # Clean up indentation
        sys_lines = [line.strip() for line in sys_prompt.split('\n') if line.strip()]
        usr_lines = [line.strip() for line in usr_prompt.split('\n') if line.strip()]
        
        sys_prompt = '\n'.join(sys_lines)
        usr_prompt = '\n'.join(usr_lines)
        
        return f"SYSTEM: {sys_prompt}\n\nUSER: {usr_prompt}"
    
    return ""


def load_prompts_for_config(config: dict) -> Dict[str, str]:
    """Load prompts based on configuration with explicit YAML mappings."""
    prompts_dir = Path(__file__).parent / "alternative_prompts"
    
    prompts = {}
    yaml_prompts = config.get('yaml_prompts', {})
    
    for prompt_name, (yaml_file, prompt_id) in yaml_prompts.items():
        yaml_path = prompts_dir / yaml_file
        prompt_text = extract_prompt_from_yaml(yaml_path, prompt_id)
        if prompt_text:
            prompts[prompt_name] = prompt_text
        else:
            print(f"    WARNING: Could not load prompt '{prompt_name}' from {yaml_file}")
    
    return prompts


# ============================================================================
# EMBEDDING COMPUTATION
# ============================================================================

def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_prompt_distances(prompts: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cosine distances between all prompt pairs."""
    prompt_names = list(prompts.keys())
    prompt_texts = list(prompts.values())
    
    # Get embeddings
    embeddings = get_embedding_local(prompt_texts, device="cpu")
    embeddings_np = [np.array(e) for e in embeddings]
    
    # Compute pairwise similarities
    n = len(prompt_names)
    similarity_matrix = np.zeros((n, n))
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim = compute_cosine_similarity(embeddings_np[i], embeddings_np[j])
            similarity_matrix[i, j] = sim
            distance_matrix[i, j] = 1 - sim
    
    similarity_df = pd.DataFrame(similarity_matrix, index=prompt_names, columns=prompt_names)
    distance_df = pd.DataFrame(distance_matrix, index=prompt_names, columns=prompt_names)
    
    return distance_df, similarity_df


# ============================================================================
# OUTPUT VARIANCE LOADING
# ============================================================================

def load_output_variance(config: dict, base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load output variance from enhanced analysis Excel."""
    enhanced_dir = base_dir / config['name'] / config['enhanced_analysis']
    excel_path = enhanced_dir / config['excel_name']
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path, sheet_name=config['sheet_name'])
    
    # Normalize column names
    prompt_col = config['prompt_col']
    f1_mean_col = config['f1_mean_col']
    f1_std_col = config['f1_std_col']
    
    # Aggregate by prompt
    agg_df = df.groupby(prompt_col).agg({
        f1_mean_col: 'mean',
        f1_std_col: 'mean'
    }).reset_index()
    
    # Normalize prompt names to title case for consistency
    agg_df[prompt_col] = agg_df[prompt_col].apply(lambda x: x.title() if isinstance(x, str) else x)
    
    return df, agg_df


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def compute_sensitivity_for_experiment(
    config: dict,
    base_dir: Path,
    output_dir: Path
) -> Dict:
    """Run sensitivity analysis for a single experiment."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {config['name']}")
    print('='*60)
    
    # Load prompts using explicit YAML mappings
    prompts = load_prompts_for_config(config)
    
    if not any(prompts.values()):
        print(f"  WARNING: No prompts found for {config['name']}")
        return {}
    
    print(f"  Loaded {len(prompts)} prompts")
    
    # Compute embeddings and distances
    print("  Computing embeddings...")
    distance_df, similarity_df = compute_prompt_distances(prompts)
    
    # Load output variance
    print("  Loading output variance...")
    try:
        variance_df, agg_variance_df = load_output_variance(config, base_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return {}
    
    # Create output directory
    exp_output_dir = output_dir / config['name']
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the reference prompt (first one, usually 'Baseline' or 'baseline')
    prompt_names = list(prompts.keys())
    reference = prompt_names[0]  # Use first prompt as reference
    
    prompt_col = config['prompt_col']
    f1_std_col = config['f1_std_col']
    
    # Build per-prompt analysis
    per_prompt_results = []
    
    for prompt in prompt_names:
        distance_from_ref = distance_df.loc[reference, prompt]
        
        # Get variance - try exact match first, then case-insensitive
        mask = agg_variance_df[prompt_col] == prompt
        if not mask.any():
            mask = agg_variance_df[prompt_col].str.lower() == prompt.lower()
        
        if mask.any():
            f1_std = agg_variance_df.loc[mask, f1_std_col].values[0]
        else:
            print(f"    WARNING: Could not find variance for prompt '{prompt}'")
            f1_std = np.nan
        
        per_prompt_results.append({
            'Prompt': prompt,
            'Distance from Reference': distance_from_ref,
            'F1 Std (Output Variance)': f1_std
        })
    
    per_prompt_df = pd.DataFrame(per_prompt_results)
    
    # Compute pairwise sensitivity - use all unique pairs
    pairwise_results = []
    
    for i, prompt_a in enumerate(prompt_names):
        for prompt_b in prompt_names[i+1:]:
            if prompt_a not in distance_df.index or prompt_b not in distance_df.columns:
                continue
                
            dist = distance_df.loc[prompt_a, prompt_b]
            
            std_a = per_prompt_df[per_prompt_df['Prompt'] == prompt_a]['F1 Std (Output Variance)'].values
            std_b = per_prompt_df[per_prompt_df['Prompt'] == prompt_b]['F1 Std (Output Variance)'].values
            
            if len(std_a) > 0 and len(std_b) > 0:
                std_a = std_a[0]
                std_b = std_b[0]
                if not np.isnan(std_a) and not np.isnan(std_b):
                    variance_diff = abs(std_b - std_a)
                    sensitivity = variance_diff / dist if dist > 0 else 0
                else:
                    variance_diff = sensitivity = np.nan
            else:
                std_a = std_b = variance_diff = sensitivity = np.nan
            
            pairwise_results.append({
                'Comparison': f"{prompt_a} → {prompt_b}",
                'Cosine Distance': dist,
                'F1 Std (A)': std_a,
                'F1 Std (B)': std_b,
                'ΔVariance': variance_diff,
                'Sensitivity': sensitivity
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    # Print summary
    print(f"\n  Per-Prompt Analysis:")
    print(per_prompt_df.to_string(index=False))
    print(f"\n  Pairwise Sensitivity:")
    print(pairwise_df.to_string(index=False))
    
    # Save results
    excel_path = exp_output_dir / "sensitivity_analysis_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        per_prompt_df.to_excel(writer, sheet_name='Per-Prompt Analysis', index=False)
        pairwise_df.to_excel(writer, sheet_name='Pairwise Analysis', index=False)
        distance_df.to_excel(writer, sheet_name='Cosine Distance Matrix')
        similarity_df.to_excel(writer, sheet_name='Cosine Similarity Matrix')
    
    print(f"\n  ✓ Results saved: {excel_path}")
    
    # Create visualization
    create_experiment_visualization(per_prompt_df, pairwise_df, distance_df, 
                                   config['name'], exp_output_dir)
    
    # Generate LaTeX table
    generate_latex_table(per_prompt_df, config['name'], exp_output_dir)
    
    return {
        'name': config['name'],
        'per_prompt_df': per_prompt_df,
        'pairwise_df': pairwise_df,
        'distance_df': distance_df
    }


def create_experiment_visualization(per_prompt_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                                   distance_df: pd.DataFrame, exp_name: str, output_dir: Path):
    """Create visualization for a single experiment."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Distance vs Variance scatter
    ax1 = axes[0]
    x = per_prompt_df['Distance from Reference'].values
    y = per_prompt_df['F1 Std (Output Variance)'].values * 1000
    prompts = per_prompt_df['Prompt'].tolist()
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(prompts)]
    
    for i, (xi, yi, prompt) in enumerate(zip(x, y, prompts)):
        if not np.isnan(yi):
            ax1.scatter(xi, yi, s=150, c=colors[i % len(colors)], edgecolor='black', 
                       linewidth=2, label=prompt, zorder=5)
            ax1.annotate(prompt, (xi, yi), textcoords="offset points", 
                        xytext=(5, 5), ha='left', fontsize=9)
    
    ax1.set_xlabel('Cosine Distance from Reference', fontsize=10)
    ax1.set_ylabel('F1 Std × 1000', fontsize=10)
    ax1.set_title('Distance vs Variance', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of variance
    ax2 = axes[1]
    valid_mask = ~per_prompt_df['F1 Std (Output Variance)'].isna()
    prompts_valid = per_prompt_df.loc[valid_mask, 'Prompt'].tolist()
    f1_stds = per_prompt_df.loc[valid_mask, 'F1 Std (Output Variance)'].values * 1000
    
    bars = ax2.bar(prompts_valid, f1_stds, color=colors[:len(prompts_valid)], edgecolor='black')
    ax2.set_xlabel('Prompt', fontsize=10)
    ax2.set_ylabel('F1 Std × 1000', fontsize=10)
    ax2.set_title('Output Variance by Prompt', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Distance heatmap
    ax3 = axes[2]
    sns.heatmap(distance_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=ax3, cbar_kws={'label': 'Cosine Distance'})
    ax3.set_title('Distance Matrix', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Sensitivity Analysis: {exp_name.replace("_", " ")}', 
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    viz_path = output_dir / "sensitivity_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {viz_path}")
    plt.close()


def generate_latex_table(per_prompt_df: pd.DataFrame, exp_name: str, output_dir: Path):
    """Generate LaTeX table for the experiment."""
    
    short_name = exp_name.replace('RQ1a_', '').replace('_', ' ').title()
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{Sensitivity Analysis: {short_name}}}")
    latex_lines.append(f"\\label{{tab:sens_{exp_name.lower()}}}")
    latex_lines.append("\\begin{tabular}{lcc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Prompt} & \\textbf{Cosine Distance} & \\textbf{F1 Std} \\\\")
    latex_lines.append(" & (from Reference) & (Output Var.) \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in per_prompt_df.iterrows():
        dist = f"{row['Distance from Reference']:.4f}"
        f1_std = f"{row['F1 Std (Output Variance)']:.6f}" if not np.isnan(row['F1 Std (Output Variance)']) else "N/A"
        latex_lines.append(f"{row['Prompt']} & {dist} & {f1_std} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_path = output_dir / "table_sensitivity.tex"
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"  ✓ LaTeX table saved: {latex_path}")


# ============================================================================
# AGGREGATE VISUALIZATION
# ============================================================================

def create_aggregate_visualization(all_results: List[Dict], output_dir: Path):
    """Create aggregate comparison figure for all experiments."""
    
    print(f"\n{'='*60}")
    print("CREATING AGGREGATE VISUALIZATION")
    print('='*60)
    
    # Prepare data for aggregate plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for common prompt names
    color_map = {
        'baseline': '#3498db', 'Baseline': '#3498db',
        'cot': '#2ecc71', 'CoT': '#2ecc71',
        'mechanistic': '#e74c3c', 'Mechanistic': '#e74c3c',
        'mechanistic_lit': '#e74c3c', 'mechanistic_original': '#9b59b6'
    }
    
    for idx, result in enumerate(all_results):
        if not result:
            continue
            
        ax = axes[idx]
        per_prompt_df = result['per_prompt_df']
        exp_name = result['name'].replace('RQ1a_', '').replace('_', '\n')
        
        x = per_prompt_df['Distance from Reference'].values
        y = per_prompt_df['F1 Std (Output Variance)'].values * 1000
        prompts = per_prompt_df['Prompt'].tolist()
        
        for xi, yi, prompt in zip(x, y, prompts):
            if not np.isnan(yi):
                color = color_map.get(prompt, '#9b59b6')
                ax.scatter(xi, yi, s=200, c=color, edgecolor='black', 
                          linewidth=2, label=prompt, zorder=5)
                ax.annotate(prompt, (xi, yi), textcoords="offset points", 
                           xytext=(5, 5), ha='left', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Cosine Distance from Reference', fontsize=10)
        ax.set_ylabel('F1 Std × 1000', fontsize=10)
        ax.set_title(exp_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, None)
    
    plt.tight_layout()
    
    agg_path = output_dir / "aggregate_sensitivity_comparison.png"
    plt.savefig(agg_path, dpi=150, bbox_inches='tight')
    print(f"✓ Aggregate visualization saved: {agg_path}")
    plt.close()
    
    # Create summary table
    create_aggregate_summary(all_results, output_dir)


def create_aggregate_summary(all_results: List[Dict], output_dir: Path):
    """Create aggregate summary table."""
    
    summary_data = []
    
    for result in all_results:
        if not result:
            continue
        
        per_prompt_df = result['per_prompt_df']
        
        # Get variance for each prompt
        for _, row in per_prompt_df.iterrows():
            summary_data.append({
                'Experiment': result['name'].replace('RQ1a_', ''),
                'Prompt': row['Prompt'],
                'Cosine Distance': row['Distance from Reference'],
                'F1 Std': row['F1 Std (Output Variance)']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to Excel
    summary_path = output_dir / "aggregate_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"✓ Aggregate summary saved: {summary_path}")
    
    # Generate aggregate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Sensitivity Analysis Summary: All RQ1a Experiments}")
    latex_lines.append("\\label{tab:sensitivity_summary}")
    latex_lines.append("\\begin{tabular}{llcc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Experiment} & \\textbf{Prompt} & \\textbf{Cosine Dist.} & \\textbf{F1 Std} \\\\")
    latex_lines.append("\\midrule")
    
    current_exp = None
    for _, row in summary_df.iterrows():
        exp = row['Experiment'].replace('_', ' ')[:20]
        if exp != current_exp:
            if current_exp is not None:
                latex_lines.append("\\midrule")
            current_exp = exp
            exp_label = exp
        else:
            exp_label = ""
        
        dist = f"{row['Cosine Distance']:.4f}"
        f1_std = f"{row['F1 Std']:.6f}" if not np.isnan(row['F1 Std']) else "N/A"
        latex_lines.append(f"{exp_label} & {row['Prompt']} & {dist} & {f1_std} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\begin{tablenotes}")
    latex_lines.append("\\small")
    latex_lines.append("\\item Cosine distance: all-mpnet-base-v2 embeddings (Song et al., 2020).")
    latex_lines.append("\\end{tablenotes}")
    latex_lines.append("\\end{table}")
    
    latex_path = output_dir / "table_sensitivity_summary.tex"
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"✓ Aggregate LaTeX table saved: {latex_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("BATCH SENSITIVITY ANALYSIS FOR ALL RQ1a EXPERIMENTS")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nReference:")
    print("  Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020).")
    print("  'MPNet: Masked and Permuted Pre-training for Language Understanding.'")
    print("  NeurIPS 2020. arXiv:2004.09297")
    
    base_dir = Path(__file__).parent.parent.parent / "final_runs"
    output_dir = base_dir / "Sensitivity_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nBase directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    all_results = []
    
    for exp_key, config in EXPERIMENTS.items():
        try:
            result = compute_sensitivity_for_experiment(config, base_dir, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {config['name']}: {e}")
            all_results.append({})
    
    # Create aggregate visualization
    create_aggregate_visualization(all_results, output_dir)
    
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print('='*60)
    print(f"Output directory: {output_dir}")
    print("\nGenerated folders:")
    for config in EXPERIMENTS.values():
        print(f"  - {config['name']}/")
    print("\nAggregate files:")
    print("  - aggregate_sensitivity_comparison.png")
    print("  - aggregate_summary.xlsx")
    print("  - table_sensitivity_summary.tex")


if __name__ == "__main__":
    main()

