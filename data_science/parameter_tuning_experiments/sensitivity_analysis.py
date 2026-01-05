#!/usr/bin/env python3
"""
Prompt Sensitivity Analysis for RQ1a

Analyzes the relationship between prompt changes (input) and output variance:
- ΔInput: Cosine distance between prompt embeddings (all-mpnet-base-v2)
- Output Variance: Standard deviation of F1/Precision/Recall across runs

This follows proper sensitivity analysis methodology where we examine
how output variance relates to input perturbations.

Reference for MPNet:
    Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020).
    "MPNet: Masked and Permuted Pre-training for Language Understanding."
    NeurIPS 2020. arXiv:2004.09297

Usage:
    python sensitivity_analysis.py [--enhanced_analysis_dir PATH]
"""

import sys
import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_science.logit_metrics import get_embedding_local


def extract_prompts_from_yaml(yaml_path: Path, prompt_id: str = "judgeCitation") -> str:
    """
    Extract sys_prompt + usr_prompt from a YAML file for a given prompt_id.
    
    Returns the last occurrence of the prompt (as files may have multiple versions).
    """
    with open(yaml_path, 'r') as f:
        content = yaml.safe_load_all(f)
        
        # YAML files have multiple documents separated by ---
        # We need to find the last judgeCitation prompt
        all_prompts = []
        
        # Re-read as raw text and parse sections
        f.seek(0)
        raw_content = f.read()
    
    # Parse the YAML content - find all prompts sections
    # The files have multiple "prompts:" sections, we want the last judgeCitation
    with open(yaml_path, 'r') as f:
        full_yaml = yaml.safe_load(f)
        
    # Try to find judgeCitation in the loaded structure
    if full_yaml and 'prompts' in full_yaml:
        for prompt in full_yaml['prompts']:
            if prompt.get('prompt_id') == prompt_id:
                sys_prompt = prompt.get('prompts', {}).get('sys_prompt', '')
                usr_prompt = prompt.get('prompts', {}).get('usr_prompt', '')
                all_prompts.append((sys_prompt, usr_prompt))
    
    # If found, return the last one (most recent version)
    if all_prompts:
        sys_prompt, usr_prompt = all_prompts[-1]
        # Combine sys_prompt and usr_prompt for embedding
        combined = f"SYSTEM: {sys_prompt.strip()}\n\nUSER: {usr_prompt.strip()}"
        return combined
    
    # Fallback: manual parsing for multi-document YAML
    return extract_prompts_manual(yaml_path, prompt_id)


def extract_prompts_manual(yaml_path: Path, prompt_id: str = "judgeCitation") -> str:
    """
    Manual extraction for YAML files with multiple documents.
    Parses the raw text to find the last judgeCitation prompt.
    """
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Split by document markers and parse each
    documents = content.split('\nprompts:')
    
    last_prompt = None
    for doc in documents[1:]:  # Skip first part before any "prompts:"
        # Find judgeCitation section
        if 'prompt_id: judgeCitation' in doc:
            # Extract the section
            lines = doc.split('\n')
            in_judge = False
            sys_prompt_lines = []
            usr_prompt_lines = []
            current_section = None
            
            for i, line in enumerate(lines):
                if 'prompt_id: judgeCitation' in line:
                    in_judge = True
                    continue
                
                if in_judge:
                    if 'prompt_id:' in line and 'judgeCitation' not in line:
                        # Hit next prompt, stop
                        break
                    
                    if 'sys_prompt:' in line:
                        current_section = 'sys'
                        # Check if value is on same line
                        if '|' in line:
                            continue  # Multi-line follows
                    elif 'usr_prompt:' in line:
                        current_section = 'usr'
                        if '|' in line:
                            continue
                    elif 'usr_prompt_extension:' in line:
                        current_section = None
                    elif 'version:' in line:
                        current_section = None
                    elif current_section == 'sys':
                        if line.strip() and not line.strip().startswith('-'):
                            sys_prompt_lines.append(line)
                    elif current_section == 'usr':
                        if line.strip() and not line.strip().startswith('-'):
                            usr_prompt_lines.append(line)
            
            if sys_prompt_lines or usr_prompt_lines:
                # Clean up indentation
                sys_text = '\n'.join(l.strip() for l in sys_prompt_lines if l.strip())
                usr_text = '\n'.join(l.strip() for l in usr_prompt_lines if l.strip())
                last_prompt = f"SYSTEM: {sys_text}\n\nUSER: {usr_text}"
    
    return last_prompt or ""


def load_prompts() -> Dict[str, str]:
    """
    Load all three prompts from YAML files.
    
    Returns dict with keys: 'Baseline', 'CoT', 'Mechanistic'
    """
    prompts_dir = Path(__file__).parent / "alternative_prompts"
    
    prompts = {}
    
    # Baseline prompt
    baseline_path = prompts_dir / "prompts_citation_baseline.yaml"
    prompts['Baseline'] = extract_prompts_from_yaml(baseline_path)
    
    # CoT prompt
    cot_path = prompts_dir / "prompts_citation_cot.yaml"
    prompts['CoT'] = extract_prompts_from_yaml(cot_path)
    
    # Mechanistic prompt
    mech_path = prompts_dir / "prompts_citation_mechanistic.yaml"
    prompts['Mechanistic'] = extract_prompts_from_yaml(mech_path)
    
    return prompts


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_prompt_distances(prompts: Dict[str, str]) -> pd.DataFrame:
    """
    Compute SBERT cosine distances between all prompt pairs.
    
    Returns DataFrame with distance matrix.
    """
    print("\n" + "="*60)
    print("COMPUTING EMBEDDINGS")
    print("="*60)
    print("Model: all-mpnet-base-v2 (768-dim)")
    print("Reference: Song et al. (2020), NeurIPS, arXiv:2004.09297")
    
    prompt_names = list(prompts.keys())
    prompt_texts = list(prompts.values())
    
    # Print prompt lengths for reference
    print("\nPrompt lengths (characters):")
    for name, text in prompts.items():
        print(f"  {name}: {len(text)} chars")
    
    # Get embeddings using existing function
    print("\nComputing embeddings...")
    embeddings = get_embedding_local(prompt_texts, device="cpu")
    
    # Convert to numpy arrays
    embeddings_np = [np.array(e) for e in embeddings]
    
    print(f"Embedding dimension: {len(embeddings_np[0])}")
    
    # Compute pairwise cosine similarities
    n = len(prompt_names)
    similarity_matrix = np.zeros((n, n))
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim = compute_cosine_similarity(embeddings_np[i], embeddings_np[j])
            similarity_matrix[i, j] = sim
            distance_matrix[i, j] = 1 - sim  # Distance = 1 - similarity
    
    # Create DataFrames
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=prompt_names, 
        columns=prompt_names
    )
    distance_df = pd.DataFrame(
        distance_matrix, 
        index=prompt_names, 
        columns=prompt_names
    )
    
    print("\n" + "-"*60)
    print("COSINE SIMILARITY MATRIX (all-mpnet-base-v2)")
    print("-"*60)
    print(similarity_df.round(4).to_string())
    
    print("\n" + "-"*60)
    print("COSINE DISTANCE MATRIX (1 - similarity)")
    print("-"*60)
    print(distance_df.round(4).to_string())
    
    return distance_df, similarity_df


def load_output_variance(enhanced_analysis_dir: Path) -> pd.DataFrame:
    """Load output variance (F1 Std, etc.) from existing analysis."""
    excel_path = enhanced_analysis_dir / "rq1a_enhanced_results.xlsx"
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Enhanced results file not found: {excel_path}")
    
    df = pd.read_excel(excel_path, sheet_name='Main (CLD x Prompt)')
    
    print("\n" + "="*60)
    print("OUTPUT VARIANCE DATA (from enhanced analysis)")
    print("="*60)
    print(df[['CLD', 'Prompt', 'F1 Mean', 'F1 Std', 'Precision Std', 'Recall Std']].to_string(index=False))
    
    # Aggregate by prompt (mean across CLDs)
    agg_df = df.groupby('Prompt').agg({
        'F1 Mean': 'mean',
        'F1 Std': 'mean',
        'Precision Std': 'mean',
        'Recall Std': 'mean',
        'Accuracy Std': 'mean',
        'AUC-ROC Std': 'mean'
    }).reset_index()
    
    print("\n" + "-"*60)
    print("AGGREGATED BY PROMPT (mean across CLDs)")
    print("-"*60)
    print(agg_df.to_string(index=False))
    
    return df, agg_df


def compute_sensitivity(distance_df: pd.DataFrame, variance_df: pd.DataFrame, 
                        agg_variance_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute sensitivity analysis examining relationship between 
    prompt distance (ΔInput) and output variance.
    
    Returns:
    - Per-prompt analysis: prompt distance from baseline vs output variance
    - Pairwise analysis: distance between prompts vs variance difference
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    print("Methodology: Examining relationship between input perturbation")
    print("             (cosine distance) and output variance (F1 Std)")
    
    # Use Baseline as reference point
    reference = 'Baseline'
    
    # === Per-Prompt Analysis ===
    # How does distance from baseline relate to output variance?
    per_prompt_results = []
    
    for prompt in ['Baseline', 'CoT', 'Mechanistic']:
        distance_from_baseline = distance_df.loc[reference, prompt]
        
        # Get variance for this prompt (mean across CLDs)
        prompt_data = agg_variance_df[agg_variance_df['Prompt'] == prompt].iloc[0]
        f1_std = prompt_data['F1 Std']
        precision_std = prompt_data['Precision Std']
        recall_std = prompt_data['Recall Std']
        
        per_prompt_results.append({
            'Prompt': prompt,
            'Distance from Baseline': distance_from_baseline,
            'F1 Std (Output Variance)': f1_std,
            'Precision Std': precision_std,
            'Recall Std': recall_std
        })
    
    per_prompt_df = pd.DataFrame(per_prompt_results)
    
    print("\n" + "-"*60)
    print("PER-PROMPT ANALYSIS (Distance from Baseline vs Output Variance)")
    print("-"*60)
    print(per_prompt_df.to_string(index=False))
    
    # === Pairwise Analysis ===
    # How does pairwise distance relate to variance differences?
    pairs = [('Baseline', 'CoT'), ('Baseline', 'Mechanistic'), ('CoT', 'Mechanistic')]
    pairwise_results = []
    
    for prompt_a, prompt_b in pairs:
        sbert_distance = distance_df.loc[prompt_a, prompt_b]
        
        # Get F1 Std for each prompt
        std_a = agg_variance_df[agg_variance_df['Prompt'] == prompt_a]['F1 Std'].values[0]
        std_b = agg_variance_df[agg_variance_df['Prompt'] == prompt_b]['F1 Std'].values[0]
        
        # Variance difference (how much does variance change?)
        variance_diff = abs(std_b - std_a)
        
        # Sensitivity: variance change per unit of input change
        sensitivity = variance_diff / sbert_distance if sbert_distance > 0 else 0
        
        pairwise_results.append({
            'Comparison': f"{prompt_a} → {prompt_b}",
            'SBERT Distance (ΔInput)': sbert_distance,
            f'F1 Std ({prompt_a})': std_a,
            f'F1 Std ({prompt_b})': std_b,
            'ΔVariance': variance_diff,
            'Sensitivity (ΔVar/ΔInput)': sensitivity
        })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    print("\n" + "-"*60)
    print("PAIRWISE ANALYSIS (Distance vs Variance Change)")
    print("-"*60)
    print(pairwise_df.to_string(index=False))
    
    # === Correlation Analysis ===
    print("\n" + "-"*60)
    print("CORRELATION ANALYSIS")
    print("-"*60)
    
    # Correlation between distance from baseline and output variance
    distances = per_prompt_df['Distance from Baseline'].values
    variances = per_prompt_df['F1 Std (Output Variance)'].values
    
    if len(distances) > 2:
        from scipy import stats
        corr, p_value = stats.pearsonr(distances, variances)
        print(f"Pearson correlation (distance vs F1 variance): r = {corr:.4f}, p = {p_value:.4f}")
        
        # Spearman for robustness
        spearman_corr, spearman_p = stats.spearmanr(distances, variances)
        print(f"Spearman correlation (distance vs F1 variance): ρ = {spearman_corr:.4f}, p = {spearman_p:.4f}")
    else:
        corr, p_value = np.nan, np.nan
        print("Not enough data points for correlation analysis")
    
    # === Interpretation ===
    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    baseline_var = per_prompt_df[per_prompt_df['Prompt'] == 'Baseline']['F1 Std (Output Variance)'].values[0]
    
    for _, row in per_prompt_df.iterrows():
        prompt = row['Prompt']
        dist = row['Distance from Baseline']
        var = row['F1 Std (Output Variance)']
        
        if prompt == 'Baseline':
            print(f"  {prompt}: Reference (variance = {var:.6f})")
        else:
            var_change = ((var - baseline_var) / baseline_var) * 100 if baseline_var > 0 else 0
            if var_change > 0:
                change_dir = "INCREASES"
            else:
                change_dir = "DECREASES"
            print(f"  {prompt}: Distance={dist:.4f}, Variance {change_dir} by {abs(var_change):.1f}%")
    
    return per_prompt_df, pairwise_df


def generate_latex_table(per_prompt_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                        distance_df: pd.DataFrame, output_dir: Path) -> str:
    """Generate LaTeX tables for the thesis."""
    
    # Table 1: Per-Prompt Sensitivity Analysis
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Prompt Sensitivity Analysis: Relationship between semantic distance and output variance}")
    latex_lines.append("\\label{tab:sensitivity_analysis}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Prompt} & \\textbf{Cosine Distance} & \\textbf{F1 Std} & \\textbf{Precision Std} & \\textbf{Recall Std} \\\\")
    latex_lines.append(" & (from Baseline) & (Output Var.) & & \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in per_prompt_df.iterrows():
        prompt = row['Prompt']
        dist = f"{row['Distance from Baseline']:.4f}"
        f1_std = f"{row['F1 Std (Output Variance)']:.6f}"
        prec_std = f"{row['Precision Std']:.6f}"
        rec_std = f"{row['Recall Std']:.6f}"
        
        latex_lines.append(f"{prompt} & {dist} & {f1_std} & {prec_std} & {rec_std} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\begin{tablenotes}")
    latex_lines.append("\\small")
    latex_lines.append("\\item Cosine distance computed using all-mpnet-base-v2 embeddings (Song et al., 2020).")
    latex_lines.append("\\item Output variance: Standard deviation of metric across experimental runs.")
    latex_lines.append("\\end{tablenotes}")
    latex_lines.append("\\end{table}")
    
    latex_content = "\n".join(latex_lines)
    
    # Save to file
    latex_path = output_dir / "table_sensitivity_analysis.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"\n✓ LaTeX table saved: {latex_path}")
    
    # Table 2: Pairwise Sensitivity
    latex_pair = []
    latex_pair.append("\\begin{table}[htbp]")
    latex_pair.append("\\centering")
    latex_pair.append("\\caption{Pairwise Prompt Sensitivity: Variance change per unit of semantic distance}")
    latex_pair.append("\\label{tab:pairwise_sensitivity}")
    latex_pair.append("\\begin{tabular}{lcccc}")
    latex_pair.append("\\toprule")
    latex_pair.append("\\textbf{Comparison} & \\textbf{$\\Delta$Input} & \\textbf{$\\Delta$Variance} & \\textbf{Sensitivity} \\\\")
    latex_pair.append(" & (Cosine dist.) & (F1 Std diff.) & ($\\Delta$Var/$\\Delta$Input) \\\\")
    latex_pair.append("\\midrule")
    
    for _, row in pairwise_df.iterrows():
        comp = row['Comparison'].replace('→', '$\\rightarrow$')
        delta_in = f"{row['SBERT Distance (ΔInput)']:.4f}"
        delta_var = f"{row['ΔVariance']:.6f}"
        sens = f"{row['Sensitivity (ΔVar/ΔInput)']:.6f}"
        
        latex_pair.append(f"{comp} & {delta_in} & {delta_var} & {sens} \\\\")
    
    latex_pair.append("\\bottomrule")
    latex_pair.append("\\end{tabular}")
    latex_pair.append("\\begin{tablenotes}")
    latex_pair.append("\\small")
    latex_pair.append("\\item Sensitivity = $|\\Delta\\text{Variance}| / \\Delta\\text{Input}$. Lower values indicate robustness.")
    latex_pair.append("\\end{tablenotes}")
    latex_pair.append("\\end{table}")
    
    pair_content = "\n".join(latex_pair)
    pair_path = output_dir / "table_pairwise_sensitivity.tex"
    with open(pair_path, 'w') as f:
        f.write(pair_content)
    print(f"✓ Pairwise sensitivity table saved: {pair_path}")
    
    # Table 3: Distance matrix
    latex_dist = []
    latex_dist.append("\\begin{table}[htbp]")
    latex_dist.append("\\centering")
    latex_dist.append("\\caption{Cosine Distance Matrix between Prompt Variants}")
    latex_dist.append("\\label{tab:prompt_distance_matrix}")
    latex_dist.append("\\begin{tabular}{lccc}")
    latex_dist.append("\\toprule")
    latex_dist.append(" & \\textbf{Baseline} & \\textbf{CoT} & \\textbf{Mechanistic} \\\\")
    latex_dist.append("\\midrule")
    
    for idx in distance_df.index:
        row_vals = [f"{distance_df.loc[idx, col]:.4f}" for col in distance_df.columns]
        latex_dist.append(f"\\textbf{{{idx}}} & {' & '.join(row_vals)} \\\\")
    
    latex_dist.append("\\bottomrule")
    latex_dist.append("\\end{tabular}")
    latex_dist.append("\\begin{tablenotes}")
    latex_dist.append("\\small")
    latex_dist.append("\\item Distance = 1 - cosine\\_similarity. Model: all-mpnet-base-v2 (768-dim).")
    latex_dist.append("\\end{tablenotes}")
    latex_dist.append("\\end{table}")
    
    dist_content = "\n".join(latex_dist)
    dist_path = output_dir / "table_prompt_distance_matrix.tex"
    with open(dist_path, 'w') as f:
        f.write(dist_content)
    print(f"✓ Distance matrix table saved: {dist_path}")
    
    return latex_content


def create_visualization(per_prompt_df: pd.DataFrame, pairwise_df: pd.DataFrame,
                        distance_df: pd.DataFrame, output_dir: Path):
    """Create sensitivity analysis visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distance vs Output Variance (scatter)
    ax1 = axes[0, 0]
    x = per_prompt_df['Distance from Baseline'].values
    y = per_prompt_df['F1 Std (Output Variance)'].values * 1000  # Scale for visibility
    prompts = per_prompt_df['Prompt'].tolist()
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (xi, yi, prompt) in enumerate(zip(x, y, prompts)):
        ax1.scatter(xi, yi, s=200, c=colors[i], edgecolor='black', linewidth=2, 
                   label=prompt, zorder=5)
        ax1.annotate(prompt, (xi, yi), textcoords="offset points", 
                    xytext=(8, 8), ha='left', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Cosine Distance from Baseline', fontsize=11)
    ax1.set_ylabel('F1 Std × 1000 (Output Variance)', fontsize=11)
    ax1.set_title('Prompt Distance vs Output Variance', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, max(x) * 1.15)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add trend line if we have enough points
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, max(x) * 1.1, 100)
        ax1.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, label='Trend')
    
    # Plot 2: Bar chart of output variance by prompt
    ax2 = axes[0, 1]
    prompts = per_prompt_df['Prompt'].tolist()
    f1_stds = per_prompt_df['F1 Std (Output Variance)'].values * 1000
    
    bars = ax2.bar(prompts, f1_stds, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Prompt Variant', fontsize=11)
    ax2.set_ylabel('F1 Std × 1000', fontsize=11)
    ax2.set_title('Output Variance by Prompt', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, f1_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Pairwise sensitivity
    ax3 = axes[1, 0]
    comparisons = pairwise_df['Comparison'].tolist()
    sensitivities = pairwise_df['Sensitivity (ΔVar/ΔInput)'].values * 1000  # Scale
    
    bars = ax3.barh(comparisons, sensitivities, color='#9b59b6', edgecolor='black')
    ax3.set_xlabel('Sensitivity × 1000 (ΔVariance / ΔInput)', fontsize=11)
    ax3.set_title('Pairwise Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{sens:.3f}', va='center', fontsize=10)
    
    # Plot 4: Distance heatmap
    ax4 = axes[1, 1]
    sns.heatmap(distance_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                ax=ax4, cbar_kws={'label': 'Cosine Distance'},
                linewidths=0.5, linecolor='white')
    ax4.set_title('Cosine Distance Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    viz_path = output_dir / "sensitivity_analysis_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {viz_path}")
    plt.close()
    
    # Also create a standalone distance heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(distance_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Cosine Distance'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Cosine Distance Matrix\n(all-mpnet-base-v2, Song et al. 2020)', 
                fontsize=13, fontweight='bold')
    
    heatmap_path = output_dir / "prompt_distance_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Distance heatmap saved: {heatmap_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Prompt Sensitivity Analysis')
    parser.add_argument('--enhanced_analysis_dir', type=str,
                       default='/home/nitai/code/causalix.ai/final_runs/RQ1a_gt_synth_citation/enhanced_analysis_20251204_155831',
                       help='Directory containing existing enhanced analysis')
    args = parser.parse_args()
    
    enhanced_dir = Path(args.enhanced_analysis_dir)
    
    print("="*60)
    print("PROMPT SENSITIVITY ANALYSIS FOR RQ1a")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMethodology:")
    print("  - Input perturbation: Cosine distance (all-mpnet-base-v2 embeddings)")
    print("  - Output variance: F1 Std across experimental runs")
    print(f"\nReference:")
    print("  Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020).")
    print("  'MPNet: Masked and Permuted Pre-training for Language Understanding.'")
    print("  NeurIPS 2020. arXiv:2004.09297")
    
    # Step 1: Load prompts
    print("\n" + "="*60)
    print("LOADING PROMPTS")
    print("="*60)
    prompts = load_prompts()
    for name, text in prompts.items():
        print(f"\n--- {name} ---")
        print(text[:300] + "..." if len(text) > 300 else text)
    
    # Step 2: Compute SBERT distances
    distance_df, similarity_df = compute_prompt_distances(prompts)
    
    # Step 3: Load existing output variance data
    variance_df, agg_variance_df = load_output_variance(enhanced_dir)
    
    # Step 4: Compute sensitivity
    per_prompt_df, pairwise_df = compute_sensitivity(distance_df, variance_df, agg_variance_df)
    
    # Step 5: Create output directory
    output_dir = enhanced_dir / "sensitivity_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Step 6: Generate LaTeX tables
    generate_latex_table(per_prompt_df, pairwise_df, distance_df, output_dir)
    
    # Step 7: Create visualizations
    create_visualization(per_prompt_df, pairwise_df, distance_df, output_dir)
    
    # Step 8: Save results to Excel
    excel_path = output_dir / "sensitivity_analysis_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        per_prompt_df.to_excel(writer, sheet_name='Per-Prompt Analysis', index=False)
        pairwise_df.to_excel(writer, sheet_name='Pairwise Analysis', index=False)
        distance_df.to_excel(writer, sheet_name='SBERT Distance Matrix')
        similarity_df.to_excel(writer, sheet_name='SBERT Similarity Matrix')
        agg_variance_df.to_excel(writer, sheet_name='Output Variance by Prompt', index=False)
        variance_df.to_excel(writer, sheet_name='Full Variance Data', index=False)
    print(f"✓ Excel results saved: {excel_path}")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    print("  - sensitivity_analysis_results.xlsx")
    print("  - table_sensitivity_analysis.tex")
    print("  - table_pairwise_sensitivity.tex")
    print("  - table_prompt_distance_matrix.tex")
    print("  - sensitivity_analysis_visualization.png")
    print("  - prompt_distance_heatmap.png")
    
    # Final summary statistics
    print("\n" + "-"*60)
    print("SUMMARY STATISTICS")
    print("-"*60)
    
    mean_distance = per_prompt_df['Distance from Baseline'].mean()
    mean_variance = per_prompt_df['F1 Std (Output Variance)'].mean()
    
    print(f"Mean Cosine Distance from Baseline: {mean_distance:.4f}")
    print(f"Mean F1 Std (Output Variance): {mean_variance:.6f}")
    
    # Compute overall sensitivity
    pairwise_sens = pairwise_df['Sensitivity (ΔVar/ΔInput)'].values
    print(f"Mean Pairwise Sensitivity: {pairwise_sens.mean():.6f} ± {pairwise_sens.std():.6f}")


if __name__ == "__main__":
    main()

