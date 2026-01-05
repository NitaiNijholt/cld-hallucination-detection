"""
Simplified Prompt Sensitivity Analysis

Creates a single dense 2-panel figure showing:
- Panel A: Semantic distance vs. ΔF1 scatter plot across all experiments
- Panel B: Compact summary table

This replaces the verbose Sobol/ANOVA approach with a simple quantitative
input-output sensitivity visualization following standard NLP ablation conventions.

References:
- Song, K. et al. (2020). MPNet: Masked and Permuted Pre-training. NeurIPS.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import yaml
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_science.logit_metrics import get_embedding_local


def bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction to a family of p-values.
    
    Args:
        p_values: Dict of {test_name: p_value}
        alpha: Significance level (default 0.05)
    
    Returns:
        Dict with correction results including adjusted p-values and significance
    """
    m = len(p_values)
    if m == 0:
        return {'method': 'Bonferroni', 'n_tests': 0, 'results': {}}
    
    alpha_adj = alpha / m
    
    results = {}
    for name, p in p_values.items():
        if p is None or np.isnan(p):
            results[name] = {
                'p_original': p,
                'p_adjusted': np.nan,
                'significant_original': False,
                'significant_adjusted': False,
                'changed': False
            }
        else:
            p_adj = min(float(p) * m, 1.0)
            results[name] = {
                'p_original': float(p),
                'p_adjusted': float(p_adj),
                'significant_original': bool(p < alpha),
                'significant_adjusted': bool(p < alpha_adj),
                'changed': bool((p < alpha) != (p < alpha_adj))
            }
    
    return {
        'method': 'Bonferroni',
        'n_tests': m,
        'alpha_original': alpha,
        'alpha_adjusted': alpha_adj,
        'results': results,
        'n_significant_original': sum(1 for r in results.values() if r.get('significant_original', False)),
        'n_significant_adjusted': sum(1 for r in results.values() if r.get('significant_adjusted', False))
    }


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    'Corr. Citation': {
        'name': 'RQ1a_gt_synth_citation',
        'judge_type': 'citation',
        'is_ground_truth': False,
        'has_mech_variants': False,  # Only 'Mechanistic' in filename
        'prompts': {
            'Baseline': ('prompts_citation_baseline.yaml', 'judgeCitation'),
            'CoT': ('prompts_citation_cot.yaml', 'judgeCitation'),
            'Mechanistic': ('prompts_citation_mechanistic.yaml', 'judgeCitation')
        }
    },
    'Corr. Correctness': {
        'name': 'RQ1a_gt_synth_correctness',
        'judge_type': 'correctness',
        'is_ground_truth': False,
        'has_mech_variants': False,
        'prompts': {
            'Baseline': ('prompts_correctness_baseline.yaml', 'judgeCorrectness'),
            'CoT': ('prompts_correctness_cot.yaml', 'judgeCorrectness'),
            'Mechanistic': ('prompts_correctness_mechanistic.yaml', 'judgeCorrectness')
        }
    },
    'GT Citation': {
        'name': 'RQ1a_gt_lit_citation',
        'judge_type': 'citation',
        'is_ground_truth': True,
        'has_mech_variants': True,  # Has mechanistic_lit and mechanistic_original
        'prompts': {
            'Baseline': ('prompts_citation_baseline.yaml', 'judgeCitation'),
            'CoT': ('prompts_citation_cot.yaml', 'judgeCitation'),
            'Mechanistic_Lit': ('prompts_citation_mechanistic.yaml', 'judgeCitation'),
            'Mechanistic_Original': ('prompts_citation_mechanistic.yaml', 'judgeCitation')
        }
    },
    'GT Correctness': {
        'name': 'RQ1a_gt_lit_correctness',
        'judge_type': 'correctness',
        'is_ground_truth': True,
        'has_mech_variants': True,
        'prompts': {
            'Baseline': ('prompts_correctness_baseline.yaml', 'judgeCorrectness'),
            'CoT': ('prompts_correctness_cot.yaml', 'judgeCorrectness'),
            'Mechanistic_Lit': ('prompts_correctness_mechanistic.yaml', 'judgeCorrectness'),
            'Mechanistic_Original': ('prompts_correctness_mechanistic.yaml', 'judgeCorrectness')
        }
    }
}

# Colors for experiments
EXP_COLORS = {
    'Corr. Citation': '#3498db',      # Blue
    'Corr. Correctness': '#2ecc71',   # Green
    'GT Citation': '#e74c3c',         # Red
    'GT Correctness': '#9b59b6'       # Purple
}

# Markers for prompt types
PROMPT_MARKERS = {
    'Baseline': 'o',
    'CoT': 's',
    'Mechanistic': '^',
    'Mechanistic_Lit': 'v',
    'Mechanistic_Original': 'D'
}

# Colors for prompts in bar charts
PROMPT_COLORS = {
    'Baseline': '#3498db',        # Blue
    'CoT': '#2ecc71',             # Green
    'Mechanistic': '#e74c3c',     # Red
    'Mechanistic_Lit': '#9b59b6', # Purple
    'Mechanistic_Original': '#f39c12'  # Orange
}


# =============================================================================
# PROMPT EXTRACTION AND EMBEDDING
# =============================================================================

def extract_prompt_from_yaml(yaml_path: Path, prompt_id: str) -> str:
    """Extract sys_prompt + usr_prompt from YAML (returns LAST matching prompt)."""
    if not yaml_path.exists():
        return ""
    
    try:
        with open(yaml_path, 'r') as f:
            full_yaml = yaml.safe_load(f)
        
        matching_prompts = []
        if full_yaml and 'prompts' in full_yaml:
            for prompt in full_yaml['prompts']:
                if prompt.get('prompt_id') == prompt_id:
                    sys_prompt = prompt.get('prompts', {}).get('sys_prompt', '')
                    usr_prompt = prompt.get('prompts', {}).get('usr_prompt', '')
                    matching_prompts.append((sys_prompt, usr_prompt))
        
        if matching_prompts:
            sys_prompt, usr_prompt = matching_prompts[-1]
            return f"SYSTEM: {sys_prompt.strip()}\n\nUSER: {usr_prompt.strip()}"
            
    except yaml.YAMLError:
        pass
    
    return ""


def compute_embeddings(prompts_dir: Path, config: dict) -> Dict[str, np.ndarray]:
    """Load prompts and compute embeddings."""
    prompts = {}
    for prompt_name, (yaml_file, prompt_id) in config['prompts'].items():
        yaml_path = prompts_dir / yaml_file
        prompt_text = extract_prompt_from_yaml(yaml_path, prompt_id)
        if prompt_text:
            prompts[prompt_name] = prompt_text
    
    if not prompts:
        return {}
    
    prompt_names = list(prompts.keys())
    prompt_texts = [prompts[name] for name in prompt_names]
    
    embeddings_list = get_embedding_local(prompt_texts, show_progress=False)
    embeddings = {name: emb for name, emb in zip(prompt_names, embeddings_list) if emb is not None}
    
    return embeddings


def compute_distance_from_baseline(embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute cosine distance from baseline for each prompt."""
    if 'Baseline' not in embeddings:
        return {}
    
    baseline_emb = embeddings['Baseline']
    distances = {}
    for name, emb in embeddings.items():
        if name == 'Baseline':
            distances[name] = 0.0
        else:
            sim = cosine_similarity([baseline_emb], [emb])[0][0]
            distances[name] = 1 - sim
    
    return distances


# =============================================================================
# F1 EXTRACTION
# =============================================================================

def compute_f1_from_file(excel_path: Path, judge_type: str, is_ground_truth: bool = False) -> float:
    """Compute F1 score from a judged Excel file."""
    try:
        df = pd.read_excel(excel_path)
        
        if is_ground_truth:
            # Ground truth: Classification = TP/TN/FP/FN based on graph structure
            # FP/FN are hallucinations (edge errors)
            # Judge's job is to identify these - low score should predict hallucination
            if 'Classification' not in df.columns or 'Judge Verdict' not in df.columns:
                return np.nan
            
            classification = df['Classification'].str.upper().str.strip()
            verdict = df['Judge Verdict'].str.upper().str.strip()
            
            # y_true: Is this edge a hallucination? (FP or FN in ground truth)
            y_true = classification.isin(['FP', 'FN']).astype(int)
            
            # y_pred: Does judge predict hallucination? (not CORRECT)
            if judge_type == 'citation':
                y_pred = verdict.isin(['NOT SUPPORTED', 'NOT_SUPPORTED',
                                       'PARTIALLY SUPPORTED', 'PARTIALLY_SUPPORTED']).astype(int)
            else:
                y_pred = verdict.isin(['INCORRECT', 'PARTIALLY_CORRECT', 'PARTIALLY CORRECT']).astype(int)
            
            if y_true.sum() == 0 or len(y_true) == 0:
                return np.nan
            
            return f1_score(y_true, y_pred, zero_division=0)
        else:
            # Corruption detection: Use Is Corrupted column
            if 'Is Corrupted' not in df.columns or 'Judge Verdict' not in df.columns:
                return np.nan
            
            # Drop rows with NaN in Is Corrupted
            df_valid = df.dropna(subset=['Is Corrupted'])
            if len(df_valid) == 0:
                return np.nan
            
            y_true = df_valid['Is Corrupted'].astype(int)
            verdict = df_valid['Judge Verdict'].str.lower().str.strip()
            
            if judge_type == 'citation':
                y_pred = verdict.isin(['not supported', 'not_supported', 
                                       'partially supported', 'partially_supported']).astype(int)
            else:
                y_pred = verdict.isin(['incorrect', 'partially_correct', 'partially correct']).astype(int)
            
            if y_true.sum() == 0 or len(y_true) == 0:
                return np.nan
                
            return f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        return np.nan


def get_prompt_type_from_filename(fname_lower: str) -> str:
    """Determine prompt type from filename, distinguishing mechanistic variants."""
    if 'baseline' in fname_lower:
        return 'Baseline'
    elif 'mechanistic_lit' in fname_lower:
        return 'Mechanistic_Lit'
    elif 'mechanistic_original' in fname_lower:
        return 'Mechanistic_Original'
    elif 'mechanistic' in fname_lower:
        return 'Mechanistic'  # fallback for experiments without variants
    elif 'cot' in fname_lower:
        return 'CoT'
    return None


def get_latest_file_per_prompt(files: List[Path]) -> Dict[str, Path]:
    """Group files by prompt type and return only the latest file per type."""
    files_by_prompt = {}
    
    for f in files:
        fname_lower = f.name.lower()
        prompt = get_prompt_type_from_filename(fname_lower)
        if prompt is None:
            continue
        
        if prompt not in files_by_prompt:
            files_by_prompt[prompt] = []
        files_by_prompt[prompt].append(f)
    
    # Select latest file per prompt type (by modification time)
    latest_per_prompt = {}
    for prompt, prompt_files in files_by_prompt.items():
        latest_per_prompt[prompt] = max(prompt_files, key=lambda x: x.stat().st_mtime)
    
    return latest_per_prompt


def extract_f1_by_subject(exp_dir: Path, judge_type: str, is_ground_truth: bool = False) -> Dict[str, Dict[str, float]]:
    """Extract F1 scores organized by subject (CLD × run) for repeated-measures ANOVA.
    
    Returns:
        Dict of {subject_id: {prompt: f1_value}}
        e.g., {'Depression_run_1': {'Baseline': 0.4, 'CoT': 0.5, 'Mechanistic': 0.45}, ...}
    """
    data_by_subject = {}
    
    cld_dirs = [d for d in exp_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('enhanced') and not d.name.startswith('.')]
    
    for cld_dir in cld_dirs:
        cld_name = cld_dir.name
        run_dirs = [d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        for run_dir in run_dirs:
            run_name = run_dir.name
            subject_id = f"{cld_name}_{run_name}"
            
            # Get all judged files, excluding backups and deprecated
            excel_files = [f for f in run_dir.glob('judged_*.xlsx') 
                          if not f.name.endswith('.backup.xlsx')
                          and 'deprecated' not in str(f).lower()]
            
            # Get only the latest file per prompt type
            latest_files = get_latest_file_per_prompt(excel_files)
            
            if not latest_files:
                continue
            
            subject_data = {}
            for prompt, excel_file in latest_files.items():
                f1 = compute_f1_from_file(excel_file, judge_type, is_ground_truth)
                if not np.isnan(f1):
                    subject_data[prompt] = f1
            
            if subject_data:
                data_by_subject[subject_id] = subject_data
    
    return data_by_subject


def extract_f1_per_prompt(exp_dir: Path, judge_type: str, is_ground_truth: bool = False) -> Dict[str, List[float]]:
    """Extract F1 scores for each prompt across all CLDs and runs.
    
    - Excludes files in /deprecated/ paths
    - Uses only the LATEST file per prompt type per run
    - Distinguishes Mechanistic_Lit vs Mechanistic_Original as separate types
    """
    f1_by_prompt = {}
    
    cld_dirs = [d for d in exp_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('enhanced') and not d.name.startswith('.')]
    
    for cld_dir in cld_dirs:
        run_dirs = [d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        for run_dir in run_dirs:
            # Get all judged files, excluding backups and deprecated
            excel_files = [f for f in run_dir.glob('judged_*.xlsx') 
                          if not f.name.endswith('.backup.xlsx')
                          and 'deprecated' not in str(f).lower()]
            
            # Get only the latest file per prompt type
            latest_files = get_latest_file_per_prompt(excel_files)
            
            for prompt, excel_file in latest_files.items():
                f1 = compute_f1_from_file(excel_file, judge_type, is_ground_truth)
                if not np.isnan(f1):
                    if prompt not in f1_by_prompt:
                        f1_by_prompt[prompt] = []
                    f1_by_prompt[prompt].append(f1)
    
    return f1_by_prompt


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def collect_all_data(base_dir: Path, prompts_dir: Path) -> pd.DataFrame:
    """Collect sensitivity data from all experiments."""
    
    all_data = []
    
    for exp_label, config in EXPERIMENTS.items():
        exp_name = config['name']
        exp_path = base_dir / exp_name
        is_ground_truth = config.get('is_ground_truth', False)
        
        print(f"Processing: {exp_label}...")
        
        if not exp_path.exists():
            print(f"  WARNING: Directory not found: {exp_path}")
            continue
        
        # Compute embeddings and distances
        embeddings = compute_embeddings(prompts_dir, config)
        if not embeddings:
            print(f"  WARNING: No embeddings computed")
            continue
        
        distances = compute_distance_from_baseline(embeddings)
        
        # Extract F1 scores
        f1_by_prompt = extract_f1_per_prompt(exp_path, config['judge_type'], is_ground_truth)
        if not f1_by_prompt:
            print(f"  WARNING: No F1 data found")
            continue
        
        # Get baseline F1 for delta calculation
        baseline_f1 = np.mean(f1_by_prompt.get('Baseline', [np.nan]))
        
        # Compile data for each prompt found in the data
        # Order: Baseline, CoT, then any Mechanistic variants
        all_possible_prompts = ['Baseline', 'CoT', 'Mechanistic', 'Mechanistic_Lit', 'Mechanistic_Original']
        available_prompts = [p for p in all_possible_prompts if p in f1_by_prompt]
        
        for prompt in available_prompts:
            f1_values = f1_by_prompt[prompt]
            f1_mean = np.mean(f1_values)
            f1_std = np.std(f1_values)
            delta_f1 = f1_mean - baseline_f1
            
            # Get distance if available, else use 0 (for variants not in config)
            dist = distances.get(prompt, 0.0)
            
            all_data.append({
                'Experiment': exp_label,
                'Prompt': prompt,
                'Distance': dist,
                'F1': f1_mean,
                'F1 Std': f1_std,
                'ΔF1': delta_f1,
                'N': len(f1_values)
            })
    
    return pd.DataFrame(all_data)


def compute_repeated_measures_anova(data_by_subject: Dict[str, Dict[str, float]]) -> dict:
    """Compute repeated-measures ANOVA with partial eta-squared.
    
    For sensitivity analysis, this properly accounts for the blocking structure
    where each subject (CLD × run) is measured under multiple conditions (prompts).
    
    Args:
        data_by_subject: Dict of {subject_id: {prompt: f1_value}}
            e.g., {'CLD1_run1': {'Baseline': 0.4, 'CoT': 0.5, 'Mechanistic': 0.45}, ...}
    
    Returns:
        dict with partial_eta_squared, f_statistic, p_value, df_effect, df_error, n_subjects, k_levels
    """
    if not data_by_subject:
        return {'partial_eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': 0, 'df_error': 0, 'n_subjects': 0, 'k_levels': 0}
    
    # Get all prompts (conditions) that appear in ANY subject
    all_prompts = set()
    for subj_data in data_by_subject.values():
        all_prompts.update(subj_data.keys())
    prompts = sorted(all_prompts)
    k = len(prompts)
    
    if k < 2:
        return {'partial_eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': 0, 'df_error': 0, 'n_subjects': 0, 'k_levels': k}
    
    # Filter to subjects with complete data (all prompts measured)
    complete_subjects = {subj: data for subj, data in data_by_subject.items() 
                         if all(p in data for p in prompts)}
    n = len(complete_subjects)
    
    if n < 2:
        return {'partial_eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': k-1, 'df_error': 0, 'n_subjects': n, 'k_levels': k}
    
    # Build data matrix: rows = subjects, cols = prompts
    Y = np.zeros((n, k))
    for i, (subj, data) in enumerate(complete_subjects.items()):
        for j, prompt in enumerate(prompts):
            Y[i, j] = data[prompt]
    
    # Compute sums of squares
    grand_mean = np.mean(Y)
    
    # SS_total
    ss_total = np.sum((Y - grand_mean) ** 2)
    
    # SS_subjects (between-subject variance)
    subject_means = np.mean(Y, axis=1)  # Mean across prompts for each subject
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    
    # SS_effect (prompt effect)
    prompt_means = np.mean(Y, axis=0)  # Mean across subjects for each prompt
    ss_effect = n * np.sum((prompt_means - grand_mean) ** 2)
    
    # SS_error (residual = SS_total - SS_subjects - SS_effect)
    ss_error = ss_total - ss_subjects - ss_effect
    
    # Degrees of freedom
    df_effect = k - 1
    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)
    
    if df_error <= 0 or ss_error <= 0:
        return {'partial_eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': df_effect, 'df_error': df_error, 'n_subjects': n, 'k_levels': k}
    
    # Mean squares
    ms_effect = ss_effect / df_effect
    ms_error = ss_error / df_error
    
    # F-statistic and p-value
    f_stat = ms_effect / ms_error
    p_value = 1 - stats.f.cdf(f_stat, df_effect, df_error)
    
    # Partial eta-squared: SS_effect / (SS_effect + SS_error)
    # This removes subject variance, giving the proportion of within-subject variance explained by prompt
    partial_eta_sq = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else 0.0
    
    # Generalized eta-squared (alternative): SS_effect / (SS_effect + SS_subjects + SS_error)
    # = SS_effect / SS_total (same as one-way ANOVA eta-squared, for comparison)
    gen_eta_sq = ss_effect / ss_total if ss_total > 0 else 0.0
    
    return {
        'partial_eta_squared': partial_eta_sq,  # Recommended for repeated measures
        'generalized_eta_squared': gen_eta_sq,  # For comparison with one-way ANOVA
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_effect': df_effect,
        'df_error': df_error,
        'n_subjects': n,
        'k_levels': k,
        'ss_effect': ss_effect,
        'ss_subjects': ss_subjects,
        'ss_error': ss_error,
        'ss_total': ss_total
    }


def get_effect_size_interpretation(eta_sq: float) -> str:
    """Interpret eta-squared effect size (Cohen, 1988)."""
    if np.isnan(eta_sq):
        return "—"
    elif eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def compute_all_sobol_metrics(base_dir: Path) -> Dict[str, dict]:
    """Compute repeated-measures ANOVA metrics for all experiments.
    
    Uses repeated-measures ANOVA with partial η² to properly account for
    the blocking structure (CLD × run as subjects, prompt as within-subject factor).
    """
    all_metrics = {}
    
    for exp_label, config in EXPERIMENTS.items():
        exp_name = config['name']
        exp_path = base_dir / exp_name
        is_ground_truth = config.get('is_ground_truth', False)
        
        if not exp_path.exists():
            continue
        
        # Extract data by subject for repeated-measures ANOVA
        data_by_subject = extract_f1_by_subject(exp_path, config['judge_type'], is_ground_truth)
        
        if not data_by_subject:
            continue
        
        # Compute repeated-measures ANOVA
        metrics = compute_repeated_measures_anova(data_by_subject)
        
        # Use partial eta-squared for effect size interpretation
        eta_sq = metrics.get('partial_eta_squared', np.nan)
        metrics['eta_squared'] = eta_sq  # For compatibility with visualization code
        metrics['effect_size'] = get_effect_size_interpretation(eta_sq)
        
        # Also store subject count and prompt count for reporting
        metrics['N'] = metrics['n_subjects'] * metrics['k_levels']  # Total observations
        metrics['k'] = metrics['k_levels']  # Number of prompt levels
        
        all_metrics[exp_label] = metrics
    
    return all_metrics


def compute_anova_p_values(base_dir: Path) -> Dict[str, float]:
    """Compute ANOVA p-value for each experiment (legacy function for compatibility)."""
    all_metrics = compute_all_sobol_metrics(base_dir)
    return {exp: m['p_value'] for exp, m in all_metrics.items()}


def create_visualization(data: pd.DataFrame, p_values: Dict[str, float], sobol_metrics: Dict[str, dict], output_dir: Path, bonferroni_results: dict = None):
    """Create a grouped bar chart showing F1 by prompt for each experiment."""
    
    # Set up the figure with 2x2 subplots (one per experiment)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    experiments = ['Corr. Citation', 'Corr. Correctness', 'GT Citation', 'GT Correctness']
    # More descriptive titles for the figure
    exp_titles = {
        'Corr. Citation': 'Citation Judge (Corruption Detection)',
        'Corr. Correctness': 'Correctness Judge (Corruption Detection)',
        'GT Citation': 'Citation Judge (Ground Truth)',
        'GT Correctness': 'Correctness Judge (Ground Truth)'
    }
    # Prompts vary by experiment type
    prompts_3 = ['Baseline', 'CoT', 'Mechanistic']
    prompts_4 = ['Baseline', 'CoT', 'Mechanistic_Lit', 'Mechanistic_Original']
    
    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        exp_data = data[data['Experiment'] == exp]
        
        if exp_data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp_titles.get(exp, exp), fontweight='bold')
            continue
        
        # Determine which prompts are available for this experiment
        available_prompts = exp_data['Prompt'].unique().tolist()
        # Order them consistently
        if any('Mechanistic_' in p for p in available_prompts):
            prompts = [p for p in prompts_4 if p in available_prompts]
        else:
            prompts = [p for p in prompts_3 if p in available_prompts]
        
        # Get data for each prompt
        x_pos = np.arange(len(prompts))
        bar_width = 0.6
        
        f1_means = []
        f1_stds = []
        colors = []
        
        for prompt in prompts:
            prompt_data = exp_data[exp_data['Prompt'] == prompt]
            if len(prompt_data) > 0:
                f1_means.append(prompt_data['F1'].values[0])
                f1_stds.append(prompt_data['F1 Std'].values[0])
                colors.append(PROMPT_COLORS.get(prompt, '#cccccc'))
            else:
                f1_means.append(0)
                f1_stds.append(0)
                colors.append('#cccccc')
        
        # Create bars
        bars = ax.bar(x_pos, f1_means, bar_width, yerr=f1_stds, 
                      color=colors, edgecolor='black', linewidth=1.5,
                      capsize=5, error_kw={'linewidth': 1.5})
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, f1_means, f1_stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.02),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add RM-ANOVA effect size + significance annotation (with Bonferroni-adjusted p-value)
        p_val = p_values.get(exp, np.nan)
        metrics = sobol_metrics.get(exp, {})
        eta_sq = metrics.get('eta_squared', np.nan)
        
        # Get adjusted p-value from bonferroni_results if available
        p_adj = np.nan
        sig_adj = False
        if bonferroni_results and exp in bonferroni_results.get('results', {}):
            p_adj = bonferroni_results['results'][exp].get('p_adjusted', np.nan)
            sig_adj = bonferroni_results['results'][exp].get('significant_adjusted', False)
        
        if not np.isnan(p_val):
            sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            adj_marker = '†' if sig_adj else ''
            # Include partial eta-squared (η²_p), p, and p_adj in annotation
            if not np.isnan(eta_sq) and not np.isnan(p_adj):
                ann_text = f'η²_p={eta_sq:.2f}, p={p_val:.3f}{sig_str}\np_adj={p_adj:.3f}{adj_marker}'
            elif not np.isnan(eta_sq):
                ann_text = f'η²_p={eta_sq:.2f}, p={p_val:.3f} ({sig_str})'
            else:
                ann_text = f'p={p_val:.3f} ({sig_str})'
            ax.text(0.98, 0.95, ann_text, 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting - use shorter labels for mechanistic variants
        short_labels = {
            'Baseline': 'Base', 'CoT': 'CoT', 'Mechanistic': 'Mech',
            'Mechanistic_Lit': 'Mech-Lit', 'Mechanistic_Original': 'Mech-Orig'
        }
        ax.set_xticks(x_pos)
        ax.set_xticklabels([short_labels.get(p, p) for p in prompts], fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title(exp_titles.get(exp, exp), fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(f1_means) * 1.4 if max(f1_means) > 0 else 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add baseline reference line
        if f1_means[0] > 0:
            ax.axhline(y=f1_means[0], color='#3498db', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add legend with all possible prompt types
    all_prompts_in_data = data['Prompt'].unique()
    legend_elements = [mpatches.Patch(facecolor=PROMPT_COLORS.get(p, '#cccccc'), edgecolor='black', label=p)
                       for p in ['Baseline', 'CoT', 'Mechanistic', 'Mechanistic_Lit', 'Mechanistic_Original']
                       if p in all_prompts_in_data]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), 
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Prompt Sensitivity: F1 Performance by Prompt Type\n(η²_p = partial eta-squared; RM-ANOVA α=0.05: * p<.05, ** p<.01, *** p<.001; † significant after Bonferroni)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prompt_sensitivity_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'prompt_sensitivity_figure.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Figure saved to: {output_dir / 'prompt_sensitivity_figure.png'}")


def create_sensitivity_bar_chart(sobol_metrics: Dict[str, dict], output_dir: Path):
    """Create a bar chart showing η²_p (Sobol sensitivity index) for each experiment."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Experiment labels and order
    experiments = ['Corr. Citation', 'Corr. Correctness', 'GT Citation', 'GT Correctness']
    exp_display = {
        'Corr. Citation': 'Citation Judge\n(Corruption)',
        'Corr. Correctness': 'Correctness Judge\n(Corruption)',
        'GT Citation': 'Citation Judge\n(Ground Truth)',
        'GT Correctness': 'Correctness Judge\n(Ground Truth)'
    }
    
    # Extract η²_p values and p-values
    eta_values = []
    p_values = []
    for exp in experiments:
        metrics = sobol_metrics.get(exp, {})
        eta_values.append(metrics.get('eta_squared', 0))
        p_values.append(metrics.get('p_value', 1.0))
    
    x_pos = np.arange(len(experiments))
    
    # Color by effect size magnitude
    colors = []
    for eta in eta_values:
        if eta >= 0.14:
            colors.append('#e74c3c')  # Large - red
        elif eta >= 0.06:
            colors.append('#f39c12')  # Medium - orange
        elif eta >= 0.01:
            colors.append('#3498db')  # Small - blue
        else:
            colors.append('#95a5a6')  # Negligible - gray
    
    # Create bars
    bars = ax.bar(x_pos, eta_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels and significance markers
    for i, (bar, eta, p) in enumerate(zip(bars, eta_values, p_values)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.annotate(f'{eta:.2f}{sig}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add effect size threshold lines
    ax.axhline(y=0.14, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1, label='Large (≥0.14)')
    ax.axhline(y=0.06, color='#f39c12', linestyle='--', alpha=0.5, linewidth=1, label='Medium (≥0.06)')
    ax.axhline(y=0.01, color='#3498db', linestyle='--', alpha=0.5, linewidth=1, label='Small (≥0.01)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp_display[e] for e in experiments], fontsize=10)
    ax.set_ylabel('Partial Eta-Squared (η²_p)', fontsize=12)
    ax.set_title('Prompt Sensitivity Index: How Much Does Prompt Choice Affect Performance?\n(η²_p = fraction of within-subject variance explained by prompt)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(eta_values) * 1.3 if max(eta_values) > 0 else 0.2)
    ax.legend(loc='upper right', fontsize=9, title='Effect Size Thresholds')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_index_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'sensitivity_index_figure.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Sensitivity index figure saved to: {output_dir / 'sensitivity_index_figure.png'}")


def generate_latex_table(data: pd.DataFrame, p_values: Dict[str, float], sobol_metrics: Dict[str, dict], output_dir: Path, bonferroni_results: dict = None):
    """Generate a proper LaTeX table for the thesis with Sobol S_i and Bonferroni-adjusted p-values."""
    
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\caption{Judge Prompt Sensitivity Analysis}")
    lines.append("\\label{tab:prompt_sensitivity}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\begin{tabular}{@{}llcccccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Experiment} & \\textbf{Prompt} & \\textbf{Dist.} & \\textbf{F1} & \\textbf{Std} & \\textbf{$\\Delta$F1} & \\textbf{$\\eta^2_p$} & \\textbf{p} \\\\")
    lines.append("\\midrule")
    
    experiments = ['Corr. Citation', 'Corr. Correctness', 'GT Citation', 'GT Correctness']
    exp_latex_names = {
        'Corr. Citation': 'Corr. Cit.',
        'Corr. Correctness': 'Corr. Corr.', 
        'GT Citation': 'GT Cit.',
        'GT Correctness': 'GT Corr.'
    }
    prompt_latex_names = {
        'Baseline': 'Baseline',
        'CoT': 'CoT',
        'Mechanistic': 'Mechanistic',
        'Mechanistic_Lit': 'Mech-Lit',
        'Mechanistic_Original': 'Mech-Orig'
    }
    
    # Order of prompts to show
    all_prompts = ['Baseline', 'CoT', 'Mechanistic', 'Mechanistic_Lit', 'Mechanistic_Original']
    
    for exp in experiments:
        exp_data = data[data['Experiment'] == exp]
        metrics = sobol_metrics.get(exp, {})
        p_val = metrics.get('p_value', np.nan)
        eta_sq = metrics.get('eta_squared', np.nan)
        effect = metrics.get('effect_size', '—')
        n_prompts = int(metrics.get('k', 0))
        
        # Get prompts available for this experiment
        available_prompts = [p for p in all_prompts if p in exp_data['Prompt'].values]
        
        first_row = True
        for prompt in available_prompts:
            row = exp_data[exp_data['Prompt'] == prompt]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            exp_name = exp_latex_names[exp] if first_row else ''
            prompt_name = prompt_latex_names.get(prompt, prompt)
            distance = f"{row['Distance']:.2f}"
            f1 = f"{row['F1']:.3f}"
            f1_std = f"{row['F1 Std']:.3f}"
            delta_f1 = f"{row['ΔF1']:+.3f}"
            
            if first_row:
                # Use Bonferroni-adjusted p-value if available
                p_adj = p_val
                if bonferroni_results and exp in bonferroni_results.get('results', {}):
                    p_adj = bonferroni_results['results'][exp].get('p_adjusted', p_val)
                
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
                p_str = f"$<$.001{sig}" if p_adj < 0.001 else f"{p_adj:.3f}{sig}"
                eta_str = f"{eta_sq:.2f}" if not np.isnan(eta_sq) else "—"
                # Use multirow for eta and p columns
                eta_cell = f"\\multirow{{{n_prompts}}}{{*}}{{{eta_str}}}"
                p_cell = f"\\multirow{{{n_prompts}}}{{*}}{{{p_str}}}"
            else:
                eta_cell = ""
                p_cell = ""
            
            lines.append(f"{exp_name} & {prompt_name} & {distance} & {f1} & {f1_std} & {delta_f1} & {eta_cell} & {p_cell} \\\\")
            first_row = False
        
        lines.append("\\midrule")
    
    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\scriptsize")
    lines.append("\\item \\textit{Note.} Dist. = MPNet cosine distance from baseline. $\\eta^2_p$ = partial eta-squared (RM-ANOVA). N=9 per prompt.")
    lines.append("\\item p-values are Bonferroni-corrected (4 tests, $\\alpha_{adj}$=0.0125). * $p<.05$, ** $p<.01$, *** $p<.001$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    
    latex_content = '\n'.join(lines)
    
    with open(output_dir / 'prompt_sensitivity_table.tex', 'w') as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX table saved to: {output_dir / 'prompt_sensitivity_table.tex'}")
    
    return latex_content


def main():
    """Run simplified sensitivity analysis."""
    
    print("="*70)
    print("SIMPLIFIED PROMPT SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_dir = Path(__file__).parent.parent.parent / "final_runs"
    prompts_dir = Path(__file__).parent / "alternative_prompts"
    output_dir = base_dir / "Sensitivity_analysis_simple"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Collect all data
    print("Collecting data from all experiments...")
    data = collect_all_data(base_dir, prompts_dir)
    
    if data.empty:
        print("ERROR: No data collected!")
        return
    
    print(f"\nData collected: {len(data)} rows")
    print(data.to_string(index=False))
    print()
    
    # Compute Sobol metrics (including ANOVA)
    print("Computing repeated-measures ANOVA (partial η²)...")
    sobol_metrics = compute_all_sobol_metrics(base_dir)
    p_values = {exp: m['p_value'] for exp, m in sobol_metrics.items()}
    
    for exp, metrics in sobol_metrics.items():
        eta = metrics.get('partial_eta_squared', metrics.get('eta_squared', np.nan))
        p = metrics['p_value']
        effect = metrics['effect_size']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        n_subj = metrics.get('n_subjects', 0)
        k = metrics.get('k_levels', metrics.get('k', 0))
        df_eff = metrics.get('df_effect', k-1)
        df_err = metrics.get('df_error', 0)
        print(f"  {exp}: partial η²={eta:.3f} ({effect}), F({df_eff},{df_err})={metrics.get('f_statistic',0):.2f}, p={p:.4f} ({sig}), n_subj={n_subj}, k={k}")
    print()
    
    # Apply Bonferroni correction (4 tests within RQ1a family)
    print("Applying Bonferroni correction (4 tests within RQ1a family)...")
    bonferroni_results = bonferroni_correction(p_values, alpha=0.05)
    print(f"  α_adjusted = {bonferroni_results['alpha_adjusted']:.4f}")
    for exp, res in bonferroni_results['results'].items():
        status = "SIG†" if res['significant_adjusted'] else "ns"
        print(f"  {exp}: p={res['p_original']:.4f} → p_adj={res['p_adjusted']:.4f} ({status})")
    print()
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(data, p_values, sobol_metrics, output_dir, bonferroni_results)
    
    # Create sensitivity index bar chart
    print("Creating sensitivity index visualization...")
    create_sensitivity_bar_chart(sobol_metrics, output_dir)
    
    # Generate LaTeX table
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(data, p_values, sobol_metrics, output_dir, bonferroni_results)
    
    # Save data to Excel
    data.to_excel(output_dir / 'sensitivity_data.xlsx', index=False)
    print(f"✓ Data saved to: {output_dir / 'sensitivity_data.xlsx'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

