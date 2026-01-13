"""
Corrector Prompt Sensitivity Analysis

Creates sensitivity analysis for RQ1b corrector experiments:
- Panel: F1 Δ by prompt for synthetic + ground truth experiments
- LaTeX table with semantic distances and ANOVA

References:
- Günther et al. (2023). Jina Embeddings 2: 8192-token general-purpose text embeddings. arXiv:2310.19923
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import yaml
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

def calc_95ci(values):
    """Calculate 95% CI using t-distribution (appropriate for N=9 blocks)."""
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)  # df=8 for N=9
    return mean - t_crit * sem, mean + t_crit * sem
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# Use Jina embeddings v2 for long-context support (8192 tokens vs MPNet's 384)
# Made optional to allow running without sentence_transformers installed
_JINA_MODEL = None
_JINA_AVAILABLE = True

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _JINA_AVAILABLE = False
    print("WARNING: sentence_transformers not installed, using default distances")

def bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """Apply Bonferroni correction to a family of p-values."""
    m = len(p_values)
    if m == 0:
        return {'method': 'Bonferroni', 'n_tests': 0, 'results': {}}
    
    alpha_adj = alpha / m
    results = {}
    for exp, p in p_values.items():
        p_adj = min(p * m, 1.0)  # Adjusted p-value (capped at 1.0)
        results[exp] = {
            'p_original': p,
            'p_adjusted': p_adj,
            'significant_original': p < alpha,
            'significant_adjusted': p_adj < alpha
        }
    
    return {
        'method': 'Bonferroni',
        'n_tests': m,
        'alpha_original': alpha,
        'alpha_adjusted': alpha_adj,
        'results': results
    }


def get_jina_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Get embeddings using Jina v2 (8192 token context)."""
    global _JINA_MODEL
    if not _JINA_AVAILABLE:
        return [None] * len(texts)
    if _JINA_MODEL is None:
        print("  Loading Jina embeddings v2 (8192 tokens)...")
        _JINA_MODEL = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    return [_JINA_MODEL.encode(t) for t in texts]


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

CORRECTOR_EXPERIMENTS = {
    'Synthetic': {
        'dirs': {
            'Baseline': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline',
            'CoT': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_cot',
            'Mechanistic': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic'
        }
    },
    'Ground Truth': {
        'dirs': {
            'Baseline': 'RQ1b_corrector_experiment_ground_truth_correctness_baseline',
            'CoT': 'RQ1b_corrector_experiment_ground_truth_correctness_cot',
            'Mechanistic': 'RQ1b_corrector_experiment_ground_truth_correctness_mechanistic'
        }
    }
}

PROMPT_COLORS = {
    'Baseline': '#3498db',
    'CoT': '#2ecc71',
    'Mechanistic': '#e74c3c'
}


# =============================================================================
# PROMPT EMBEDDING
# =============================================================================

def load_corrector_prompts() -> Dict[str, str]:
    """Load corrector prompts from YAML file."""
    # Script location: final_runs/Sensitivity_analysis_simple/scripts/
    project_root = Path(__file__).parent.parent.parent.parent
    yaml_path = project_root / "data_science" / "parameter_tuning_experiments" / "corrector_prompts.yaml"
    
    if not yaml_path.exists():
        print(f"WARNING: {yaml_path} not found")
        return {}
    
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Parse the YAML-like structure
    prompts = {}
    
    # Baseline: ends before '# VARIANT 2:'
    baseline_start = content.find('baseline_prompt: |') + len('baseline_prompt: |')
    baseline_end = content.find('# VARIANT 2:', baseline_start)
    if baseline_end != -1:
        prompts['Baseline'] = content[baseline_start:baseline_end].strip()
        print(f"  Baseline prompt length: {len(prompts['Baseline'])} chars")
    
    # CoT: between 'cot_prompt: |' and '# VARIANT 3:'
    cot_start = content.find('cot_prompt: |') + len('cot_prompt: |')
    cot_end = content.find('# VARIANT 3:', cot_start)
    if cot_end == -1:
        cot_end = content.find('mechanistic_prompt:', cot_start)
    if cot_end != -1:
        prompts['CoT'] = content[cot_start:cot_end].strip()
        print(f"  CoT prompt length: {len(prompts['CoT'])} chars")
    
    # Mechanistic: use the SECOND occurrence (with full Bradford Hill)
    first_mech = content.find('mechanistic_prompt: |')
    second_mech = content.find('mechanistic_prompt: |', first_mech + 1)
    mech_start = second_mech + len('mechanistic_prompt: |') if second_mech != -1 else first_mech + len('mechanistic_prompt: |')
    prompts['Mechanistic'] = content[mech_start:].strip()
    print(f"  Mechanistic prompt length: {len(prompts['Mechanistic'])} chars")
    
    return prompts


def compute_prompt_distances() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute cosine distances from baseline for each corrector prompt.
    
    Uses Jina embeddings v2 (8192 token context) to embed FULL prompts without truncation.
    This avoids the MPNet 384-token limitation that caused identical embeddings for
    cumulative prompts sharing the same prefix.
    
    Returns:
        Tuple of (distances dict, length_ratios dict)
    """
    prompts = load_corrector_prompts()
    
    if not prompts or 'Baseline' not in prompts:
        print("WARNING: Could not load corrector prompts")
        return {'Baseline': 0.0, 'CoT': 0.0, 'Mechanistic': 0.0}, {'Baseline': 1.0, 'CoT': 1.0, 'Mechanistic': 1.0}
    
    baseline = prompts['Baseline']
    cot = prompts['CoT']
    mech = prompts['Mechanistic']
    
    # Get embeddings of FULL prompts using Jina (8192 tokens - no truncation)
    prompt_texts = [baseline, cot, mech]
    embeddings = get_jina_embeddings(prompt_texts)
    
    if embeddings[0] is None:
        print("WARNING: Could not compute embeddings")
        return {'Baseline': 0.0, 'CoT': 0.0, 'Mechanistic': 0.0}, {'Baseline': 1.0, 'CoT': 1.0, 'Mechanistic': 1.0}
    
    # Compute distances from baseline
    baseline_emb = embeddings[0]
    distances = {}
    for i, name in enumerate(['Baseline', 'CoT', 'Mechanistic']):
        sim = cosine_similarity([baseline_emb], [embeddings[i]])[0][0]
        distances[name] = 1 - sim
    
    print(f"\n  Cosine distances (Jina v2, full prompts):")
    for name, dist in distances.items():
        print(f"    {name}: {dist:.4f}")
    
    # Also compute relative length increase as alternative measure
    baseline_len = len(baseline)
    length_ratios = {}
    print(f"\n  Prompt length ratios (vs baseline):")
    for name in ['Baseline', 'CoT', 'Mechanistic']:
        prompt_len = len(prompts.get(name, ''))
        ratio = prompt_len / baseline_len if baseline_len > 0 else 1.0
        length_ratios[name] = ratio
        print(f"    {name}: {ratio:.2f}x ({prompt_len} chars)")
    
    return distances, length_ratios


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def count_actions_from_json(json_path: Path) -> dict:
    """Count correction actions from outcomes JSON file."""
    import json
    
    try:
        with open(json_path, 'r') as f:
            outcomes = json.load(f)
        
        counts = {'remove': 0, 'flip_polarity': 0, 'change_type': 0, 'revise': 0, 'none': 0}
        for o in outcomes:
            action = o.get('action', 'none')
            if action in counts:
                counts[action] += 1
            else:
                counts['none'] += 1
        
        return {
            'Actions_Remove': counts['remove'],
            'Actions_Flip': counts['flip_polarity'],
            'Actions_Change': counts['change_type'],
            'Actions_Revise': counts['revise'],
            'Actions_Total': sum(counts.values())
        }
    except Exception as e:
        return {}


def extract_action_counts_from_jsons(exp_path: Path) -> Dict[str, Dict[int, dict]]:
    """Extract action counts from JSON files in run folders.
    
    Returns: Dict[cld_name, Dict[run_num, action_counts]]
    """
    action_data = {}
    
    for cld_dir in exp_path.iterdir():
        if not cld_dir.is_dir() or cld_dir.name.startswith('.'):
            continue
        cld_name = cld_dir.name
        action_data[cld_name] = {}
        
        for run_dir in cld_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                continue
            
            try:
                run_num = int(run_dir.name.split('_')[1])
            except:
                continue
            
            # Find the most recent correction outcomes JSON
            json_files = list(run_dir.glob('*correction_outcomes*.json'))
            if json_files:
                # Sort by modification time, newest first
                json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                counts = count_actions_from_json(json_files[0])
                if counts:
                    action_data[cld_name][run_num] = counts
    
    return action_data


def load_corrector_results(base_dir: Path, exp_dir: str) -> pd.DataFrame:
    """Load corrector results from Excel file, enriching with action counts from JSON if needed.
    
    Priority order:
    1. rq1b_all_results_combined_final.xlsx (most complete F1 data)
    2. rq1b_all_results.xlsx
    3. rq1b_synthetic_complete.xlsx
    
    If action counts are missing, extract from JSON files.
    """
    exp_path = base_dir / exp_dir
    
    # Try different Excel file patterns - prefer complete data first
    df = None
    for pattern in ['rq1b_all_results_combined_final.xlsx', 'rq1b_all_results.xlsx', 
                    'rq1b_synthetic_complete.xlsx']:
        xlsx_path = exp_path / pattern
        if xlsx_path.exists():
            try:
                df = pd.read_excel(xlsx_path, engine='openpyxl')
                print(f"    Loaded {pattern} ({len(df)} rows)")
                break
            except Exception as e:
                print(f"  Error reading {xlsx_path}: {e}")
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Check if action counts are missing
    has_actions = 'Actions_Total' in df.columns and df['Actions_Total'].notna().any()
    
    if not has_actions:
        # Try to extract action counts from JSON files
        print(f"    Extracting action counts from JSON files...")
        json_actions = extract_action_counts_from_jsons(exp_path)
        
        if json_actions:
            # Add action count columns
            df['Actions_Remove'] = 0
            df['Actions_Flip'] = 0
            df['Actions_Change'] = 0
            df['Actions_Revise'] = 0
            df['Actions_Total'] = 0
            
            matched = 0
            for idx, row in df.iterrows():
                cld = row.get('CLD', '')
                run = row.get('Run', 0)
                
                if cld in json_actions and run in json_actions[cld]:
                    counts = json_actions[cld][run]
                    df.at[idx, 'Actions_Remove'] = counts.get('Actions_Remove', 0)
                    df.at[idx, 'Actions_Flip'] = counts.get('Actions_Flip', 0)
                    df.at[idx, 'Actions_Change'] = counts.get('Actions_Change', 0)
                    df.at[idx, 'Actions_Revise'] = counts.get('Actions_Revise', 0)
                    df.at[idx, 'Actions_Total'] = counts.get('Actions_Total', 0)
                    matched += 1
            
            print(f"    Matched {matched}/{len(df)} rows with JSON action counts")
    else:
        print(f"    Action counts present in Excel")
    
    return df


def extract_f1_delta_by_prompt(base_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Extract F1 delta values for each experiment and prompt."""
    results = {}
    
    for exp_type, config in CORRECTOR_EXPERIMENTS.items():
        results[exp_type] = {}
        
        for prompt, exp_dir in config['dirs'].items():
            df = load_corrector_results(base_dir, exp_dir)
            
            if df.empty:
                print(f"  WARNING: No data for {exp_type} / {prompt}")
                continue
            
            if 'F1_Delta' in df.columns:
                f1_deltas = df['F1_Delta'].dropna().tolist()
                results[exp_type][prompt] = f1_deltas
            elif 'F1 Delta' in df.columns:
                f1_deltas = df['F1 Delta'].dropna().tolist()
                results[exp_type][prompt] = f1_deltas
    
    return results


def extract_full_results_by_prompt(base_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Extract full results including action counts for each experiment and prompt."""
    results = {}
    
    for exp_type, config in CORRECTOR_EXPERIMENTS.items():
        results[exp_type] = {}
        
        for prompt, exp_dir in config['dirs'].items():
            df = load_corrector_results(base_dir, exp_dir)
            
            if df.empty:
                print(f"  WARNING: No data for {exp_type} / {prompt}")
                continue
            
            results[exp_type][prompt] = df
    
    return results


def aggregate_action_counts(full_results: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate action counts per prompt for each experiment type."""
    action_cols = ['Actions_Remove', 'Actions_Flip', 'Actions_Change', 'Actions_Revise', 'Actions_Total']
    aggregated = {}
    
    for exp_type, prompts_data in full_results.items():
        aggregated[exp_type] = {}
        
        for prompt, df in prompts_data.items():
            action_sums = {}
            for col in action_cols:
                if col in df.columns:
                    action_sums[col] = df[col].sum()
                else:
                    action_sums[col] = None  # Mark as unavailable
            
            aggregated[exp_type][prompt] = action_sums
    
    return aggregated


def compute_repeated_measures_anova(data_by_subject: Dict[str, Dict[str, float]]) -> dict:
    """Compute repeated-measures ANOVA with partial eta-squared.
    
    For sensitivity analysis, this properly accounts for the blocking structure
    where each subject (CLD × run) is measured under multiple conditions (prompts).
    
    Args:
        data_by_subject: Dict of {subject_id: {prompt: f1_delta_value}}
    
    Returns:
        dict with partial_eta_squared, f_statistic, p_value, df_effect, df_error, n_subjects, k_levels
    """
    if not data_by_subject:
        return {'partial_eta_squared': np.nan, 'eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': 0, 'df_error': 0, 'n_subjects': 0, 'k_levels': 0, 'k': 0, 'N': 0, 'effect_size': '—'}
    
    # Get all prompts (conditions) that appear in ANY subject
    all_prompts = set()
    for subj_data in data_by_subject.values():
        all_prompts.update(subj_data.keys())
    prompts = sorted(all_prompts)
    k = len(prompts)
    
    if k < 2:
        return {'partial_eta_squared': np.nan, 'eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': 0, 'df_error': 0, 'n_subjects': 0, 'k_levels': k, 'k': k, 'N': 0, 'effect_size': '—'}
    
    # Filter to subjects with complete data (all prompts measured)
    complete_subjects = {subj: data for subj, data in data_by_subject.items() 
                         if all(p in data for p in prompts)}
    n = len(complete_subjects)
    
    if n < 2:
        return {'partial_eta_squared': np.nan, 'eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': k-1, 'df_error': 0, 'n_subjects': n, 'k_levels': k, 'k': k, 'N': n*k, 'effect_size': '—'}
    
    # Build data matrix: rows = subjects, cols = prompts
    Y = np.zeros((n, k))
    for i, (subj, data) in enumerate(complete_subjects.items()):
        for j, prompt in enumerate(prompts):
            Y[i, j] = data[prompt]

    # -------------------------------------------------------------------------
    # Assumption checks (secondary; RM-ANOVA used as sensitivity summary)
    # -------------------------------------------------------------------------
    grand_mean = np.mean(Y)
    subject_means = np.mean(Y, axis=1, keepdims=True)
    prompt_means = np.mean(Y, axis=0, keepdims=True)
    resid = Y - subject_means - prompt_means + grand_mean

    # Shapiro-Wilk on residuals
    try:
        shapiro_w, shapiro_p = stats.shapiro(resid.flatten())
    except Exception:
        shapiro_w, shapiro_p = np.nan, np.nan

    # Mauchly + Greenhouse-Geisser
    mauchly_w, mauchly_p = np.nan, np.nan
    eps_gg, p_gg = np.nan, np.nan
    try:
        Yc = Y - np.mean(Y, axis=1, keepdims=True)
        S = np.cov(Yc, rowvar=False, bias=False)
        trS = np.trace(S)
        sign, logdet = np.linalg.slogdet(S)
        if trS > 0 and sign <= 0:
            ridge = 1e-6 * (trS / k)
            S = S + ridge * np.eye(k)
            sign, logdet = np.linalg.slogdet(S)

        if trS > 0 and sign > 0 and k > 2 and n > 2:
            W = float(np.exp(logdet - k * np.log(trS / k)))
            df_m = (k * (k - 1) // 2) - 1
            c = (2 * (k ** 2) + k + 2) / (6 * (k - 1) * (n - 1))
            chi2 = -(n - 1) * (1 - c) * np.log(W)
            mauchly_w = float(W)
            mauchly_p = float(1 - stats.chi2.cdf(chi2, df_m)) if df_m > 0 else np.nan

            trS2 = np.trace(S @ S)
            if trS2 > 0 and (k - 1) > 0:
                eps_gg = float((trS ** 2) / ((k - 1) * trS2))
                eps_gg = max(min(eps_gg, 1.0), 1.0 / (k - 1))
    except Exception:
        pass
    
    # Compute sums of squares
    grand_mean = np.mean(Y)
    
    # SS_total
    ss_total = np.sum((Y - grand_mean) ** 2)
    
    # SS_subjects (between-subject variance)
    subject_means = np.mean(Y, axis=1)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    
    # SS_effect (prompt effect)
    prompt_means = np.mean(Y, axis=0)
    ss_effect = n * np.sum((prompt_means - grand_mean) ** 2)
    
    # SS_error (residual)
    ss_error = ss_total - ss_subjects - ss_effect
    
    # Degrees of freedom
    df_effect = k - 1
    df_error = (n - 1) * (k - 1)
    
    if df_error <= 0 or ss_error <= 0:
        return {'partial_eta_squared': np.nan, 'eta_squared': np.nan, 'f_statistic': np.nan, 'p_value': np.nan,
                'df_effect': df_effect, 'df_error': df_error, 'n_subjects': n, 'k_levels': k, 'k': k, 'N': n*k, 'effect_size': '—'}
    
    # Mean squares and F-statistic
    ms_effect = ss_effect / df_effect
    ms_error = ss_error / df_error
    f_stat = ms_effect / ms_error
    p_value = 1 - stats.f.cdf(f_stat, df_effect, df_error)

    if not np.isnan(eps_gg) and eps_gg > 0:
        p_gg = float(1 - stats.f.cdf(f_stat, df_effect * eps_gg, df_error * eps_gg))
    
    # Partial eta-squared
    partial_eta_sq = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else 0.0
    
    # Effect size interpretation
    if np.isnan(partial_eta_sq):
        effect = "—"
    elif partial_eta_sq < 0.01:
        effect = "negligible"
    elif partial_eta_sq < 0.06:
        effect = "small"
    elif partial_eta_sq < 0.14:
        effect = "medium"
    else:
        effect = "large"
    
    return {
        'partial_eta_squared': partial_eta_sq,
        'eta_squared': partial_eta_sq,  # For compatibility
        'f_statistic': f_stat,
        'p_value': p_value,
        'p_value_gg': p_gg,
        'df_effect': df_effect,
        'df_error': df_error,
        'eps_gg': eps_gg,
        'shapiro_w': float(shapiro_w) if not np.isnan(shapiro_w) else np.nan,
        'shapiro_p': float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        'mauchly_w': mauchly_w,
        'mauchly_p': mauchly_p,
        'n_subjects': n,
        'k_levels': k,
        'k': k,
        'N': n * k,
        'effect_size': effect
    }


def generate_assumptions_table(sobol_metrics: Dict[str, dict], output_dir: Path) -> str:
    """Generate LaTeX table summarizing RM-ANOVA assumption checks (Appendix K)."""
    experiments = ['Synthetic', 'Ground Truth']
    exp_titles = {'Synthetic': 'Synth.', 'Ground Truth': 'GT Lit'}

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{RM-ANOVA Assumption Checks for Corrector Prompt Sensitivity (Appendix K)}")
    lines.append(r"\label{tab:corrector_sensitivity_assumptions}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{lccccccl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Exp.} & \textbf{$n$} & \textbf{$k$} & \textbf{Shapiro $p$} & \textbf{Mauchly $p$} & \textbf{$\epsilon_{GG}$} & \textbf{$p$ (unc.)} & \textbf{$p$ (GG)} \\")
    lines.append(r"\midrule")

    def fmt_p(x):
        if x is None or np.isnan(x):
            return r"---"
        if x < 0.001:
            return r"$<$.001"
        return f"{x:.3f}"

    for exp in experiments:
        m = sobol_metrics.get(exp, {})
        n = int(m.get('n_subjects', 0) or 0)
        k = int(m.get('k_levels', m.get('k', 0)) or 0)
        shp = m.get('shapiro_p', np.nan)
        mau = m.get('mauchly_p', np.nan)
        eps = m.get('eps_gg', np.nan)
        p_unc = m.get('p_value', np.nan)
        p_gg = m.get('p_value_gg', np.nan)
        eps_str = "---" if eps is None or np.isnan(eps) else f"{eps:.2f}"
        lines.append(
            f"{exp_titles.get(exp, exp)} & {n} & {k} & {fmt_p(shp)} & {fmt_p(mau)} & {eps_str} & {fmt_p(p_unc)} & {fmt_p(p_gg)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\scriptsize")
    lines.append(r"\item \textit{Note.} Shapiro-Wilk is applied to RM-ANOVA residuals (subject + prompt additive model). Mauchly's test assesses sphericity of the within-subject covariance. $\epsilon_{GG}$ is Greenhouse--Geisser epsilon; $p$ (GG) uses $\epsilon_{GG}$-adjusted degrees of freedom. For $k=2$, sphericity is not defined and Mauchly/GG are omitted.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    latex_content = "\n".join(lines)
    out_path = output_dir / "corrector_sensitivity_assumptions_table.tex"
    out_path.write_text(latex_content, encoding="utf-8")
    print(f"✓ LaTeX assumptions table saved to: {out_path}")
    return latex_content


def convert_results_to_subject_format(results: Dict[str, Dict[str, List[float]]], 
                                       full_results: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Convert results to subject-level format for repeated-measures ANOVA.
    
    Args:
        results: Dict of {exp_type: {prompt: [f1_delta_values]}}
        full_results: Dict of {exp_type: {prompt: DataFrame with CLD, Run, F1_Delta columns}}
    
    Returns:
        Dict of {exp_type: {subject_id: {prompt: f1_delta}}}
    """
    subject_data = {}
    
    for exp_type, prompts_data in full_results.items():
        subject_data[exp_type] = {}
        
        for prompt, df in prompts_data.items():
            if df is None or df.empty:
                continue
            
            for _, row in df.iterrows():
                cld = str(row.get('CLD', 'Unknown'))
                run = row.get('Run', 0)
                f1_delta = row.get('F1_Delta', np.nan)
                
                if pd.isna(f1_delta):
                    continue
                
                subject_id = f"{cld}_run_{run}"
                
                if subject_id not in subject_data[exp_type]:
                    subject_data[exp_type][subject_id] = {}
                
                subject_data[exp_type][subject_id][prompt] = f1_delta
    
    return subject_data


def compute_all_sobol_metrics(results: Dict[str, Dict[str, List[float]]], 
                               full_results: Dict[str, pd.DataFrame] = None) -> Dict[str, dict]:
    """Compute repeated-measures ANOVA metrics for all experiments."""
    all_metrics = {}
    
    if full_results:
        # Use repeated-measures ANOVA with subject-level data
        subject_data = convert_results_to_subject_format(results, full_results)
        
        for exp_type, subj_data in subject_data.items():
            all_metrics[exp_type] = compute_repeated_measures_anova(subj_data)
    else:
        # Fallback to one-way ANOVA (not recommended)
        for exp_type, prompts_data in results.items():
            # Simple fallback - just compute basic stats
            groups = [np.array(vals) for vals in prompts_data.values() if len(vals) > 0]
            if len(groups) >= 2:
                _, p_value = stats.f_oneway(*groups)
                all_metrics[exp_type] = {'p_value': p_value, 'eta_squared': np.nan, 'effect_size': '—', 'k': len(groups), 'N': sum(len(g) for g in groups)}
            else:
                all_metrics[exp_type] = {'p_value': np.nan, 'eta_squared': np.nan, 'effect_size': '—', 'k': 0, 'N': 0}
    
    return all_metrics


def compute_anova_p_values(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    """Compute ANOVA p-values for each experiment (legacy function)."""
    all_metrics = compute_all_sobol_metrics(results)
    return {exp: m['p_value'] for exp, m in all_metrics.items()}


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(results: Dict[str, Dict[str, List[float]]], 
                         distances: Dict[str, float],
                         p_values: Dict[str, float],
                         sobol_metrics: Dict[str, dict],
                         output_dir: Path):
    """Create corrector sensitivity figure with Sobol first-order indices."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    experiments = ['Synthetic', 'Ground Truth']
    # More descriptive titles for the figure
    exp_titles = {
        'Synthetic': 'Corrector (Synthetic Corruption)',
        'Ground Truth': 'Corrector (Ground Truth)'
    }
    prompts = ['Baseline', 'CoT', 'Mechanistic']
    
    for idx, exp_type in enumerate(experiments):
        ax = axes[idx]
        exp_data = results.get(exp_type, {})
        
        if not exp_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp_titles.get(exp_type, exp_type), fontweight='bold')
            continue
        
        x_pos = np.arange(len(prompts))
        bar_width = 0.6
        
        f1_means = []
        f1_ci_errs = []  # CI halfwidth for error bars (inferential, per uncertainty rule)
        colors = []
        
        for prompt in prompts:
            if prompt in exp_data and len(exp_data[prompt]) > 0:
                values = exp_data[prompt]
                mean = np.mean(values)
                f1_means.append(mean)
                # Calculate 95% CI halfwidth using t-distribution
                ci_low, ci_high = calc_95ci(values)
                ci_err = (ci_high - ci_low) / 2 if not (np.isnan(ci_low) or np.isnan(ci_high)) else 0
                f1_ci_errs.append(ci_err)
                colors.append(PROMPT_COLORS[prompt])
            else:
                f1_means.append(0)
                f1_ci_errs.append(0)
                colors.append('#cccccc')
        
        bars = ax.bar(x_pos, f1_means, bar_width, yerr=f1_ci_errs,
                      color=colors, edgecolor='black', linewidth=1.5,
                      capsize=5, error_kw={'linewidth': 1.5})
        
        # Add value labels
        for bar, mean, ci_err in zip(bars, f1_means, f1_ci_errs):
            height = bar.get_height()
            y_pos = height + ci_err + 0.01 if height >= 0 else height - ci_err - 0.03
            ax.annotate(f'{mean:+.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=11, fontweight='bold')
        
        # Add Sobol S_i and significance annotation
        p_val = p_values.get(exp_type, np.nan)
        metrics = sobol_metrics.get(exp_type, {})
        eta_sq = metrics.get('eta_squared', np.nan)
        
        if not np.isnan(p_val):
            sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            if not np.isnan(eta_sq):
                ann_text = f'η²_p={eta_sq:.2f}, p={p_val:.3f} ({sig_str})'
            else:
                ann_text = f'p={p_val:.3f} ({sig_str})'
            ax.text(0.98, 0.95, ann_text,
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_xticks(x_pos)
        # Simpler labels without distances (distances shown in table)
        ax.set_xticklabels(prompts, fontsize=10)
        ax.set_ylabel('F1 Δ (Post - Pre Correction)', fontsize=11)
        ax.set_title(exp_titles.get(exp_type, exp_type), fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=prompt)
                       for prompt, color in PROMPT_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Corrector Prompt Sensitivity: F1 Change by Prompt Type\n(η²_p = partial eta-squared; RM-ANOVA, α=0.05: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant)',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'corrector_sensitivity_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'corrector_sensitivity_figure.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Figure saved to: {output_dir / 'corrector_sensitivity_figure.png'}")


def create_sensitivity_bar_chart(sobol_metrics: Dict[str, dict], output_dir: Path):
    """Create a bar chart showing η²_p (Sobol sensitivity index) for corrector experiments."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    experiments = ['Synthetic', 'Ground Truth']
    exp_display = ['Synthetic\nCorruption', 'Ground Truth\nHallucinations']
    
    eta_values = []
    p_values = []
    for exp in experiments:
        metrics = sobol_metrics.get(exp, {})
        eta_values.append(metrics.get('eta_squared', 0))
        p_values.append(metrics.get('p_value', 1.0))
    
    x_pos = np.arange(len(experiments))
    
    # Color by effect size
    colors = []
    for eta in eta_values:
        if eta >= 0.14:
            colors.append('#e74c3c')  # Large
        elif eta >= 0.06:
            colors.append('#f39c12')  # Medium
        elif eta >= 0.01:
            colors.append('#3498db')  # Small
        else:
            colors.append('#95a5a6')  # Negligible
    
    bars = ax.bar(x_pos, eta_values, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
    
    for i, (bar, eta, p) in enumerate(zip(bars, eta_values, p_values)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.annotate(f'{eta:.2f}{sig}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.axhline(y=0.14, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1, label='Large (≥0.14)')
    ax.axhline(y=0.06, color='#f39c12', linestyle='--', alpha=0.5, linewidth=1, label='Medium (≥0.06)')
    ax.axhline(y=0.01, color='#3498db', linestyle='--', alpha=0.5, linewidth=1, label='Small (≥0.01)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_display, fontsize=11)
    ax.set_ylabel('Partial Eta-Squared (η²_p)', fontsize=12)
    ax.set_title('Corrector Prompt Sensitivity Index\n(η²_p = within-subject variance explained by prompt choice)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(eta_values) * 1.3 if max(eta_values) > 0 else 0.2)
    ax.legend(loc='upper right', fontsize=9, title='Effect Size')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'corrector_sensitivity_index_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'corrector_sensitivity_index_figure.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Corrector sensitivity index figure saved to: {output_dir / 'corrector_sensitivity_index_figure.png'}")


def generate_latex_table(results: Dict[str, Dict[str, List[float]]],
                         distances: Dict[str, float],
                         p_values: Dict[str, float],
                         sobol_metrics: Dict[str, dict],
                         output_dir: Path,
                         action_counts: Dict[str, Dict[str, Dict[str, float]]] = None,
                         bonferroni_results: dict = None) -> str:
    """Generate LaTeX table for corrector sensitivity with Distance, Sobol S_i and action counts."""
    
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\caption{Corrector Prompt Sensitivity Analysis}")
    lines.append("\\label{tab:corrector_sensitivity}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\begin{tabular}{@{}llccccccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Experiment} & \\textbf{Prompt} & \\textbf{Dist.} & \\textbf{F1$\\Delta$} & \\textbf{Std} & \\textbf{95\\% CI} & \\textbf{Actions} & \\textbf{$\\eta^2_p$} & \\textbf{p} \\\\")
    lines.append("\\midrule")
    
    for exp_type in ['Synthetic', 'Ground Truth']:
        exp_data = results.get(exp_type, {})
        metrics = sobol_metrics.get(exp_type, {})
        p_val = metrics.get('p_value', np.nan)
        eta_sq = metrics.get('eta_squared', np.nan)
        effect = metrics.get('effect_size', '—')
        n_prompts = int(metrics.get('k', 0))
        
        first_row = True
        exp_short = 'Synth.' if exp_type == 'Synthetic' else 'GT'
        for prompt in ['Baseline', 'CoT', 'Mechanistic']:
            if prompt not in exp_data:
                continue
            
            values = exp_data[prompt]
            if len(values) == 0:
                continue
            
            exp_name = exp_short if first_row else ''
            f1_delta = f"{np.mean(values):+.2f}"
            std = f"{np.std(values):.2f}"
            
            # Calculate 95% CI (t-distribution for N=9 blocks)
            ci_low, ci_high = calc_95ci(values)
            ci_cell = f"[{ci_low:+.2f}, {ci_high:+.2f}]" if not (np.isnan(ci_low) or np.isnan(ci_high)) else "—"
            
            # Get action counts (total only)
            if action_counts and exp_type in action_counts and prompt in action_counts[exp_type]:
                act = action_counts[exp_type][prompt]
                actions_total = int(act.get('Actions_Total', 0) or 0)
            else:
                actions_total = '—'
            
            # Get distance for this prompt (use abs to avoid -0.00)
            dist = abs(distances.get(prompt, 0.0))
            dist_str = f"{dist:.2f}"
            
            if first_row:
                # Use Bonferroni-adjusted p-value if available
                p_adj = p_val
                if bonferroni_results and exp_type in bonferroni_results.get('results', {}):
                    p_adj = bonferroni_results['results'][exp_type].get('p_adjusted', p_val)
                
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
                p_str = f"$<$.001{sig}" if p_adj < 0.001 else (f"{p_adj:.3f}{sig}" if not np.isnan(p_adj) else "—")
                eta_str = f"{eta_sq:.2f}" if not np.isnan(eta_sq) else "—"
                # Use multirow
                eta_cell = f"\\multirow{{{n_prompts}}}{{*}}{{{eta_str}}}"
                p_cell = f"\\multirow{{{n_prompts}}}{{*}}{{{p_str}}}"
            else:
                eta_cell = ""
                p_cell = ""
            
            lines.append(f"{exp_name} & {prompt} & {dist_str} & {f1_delta} & {std} & {ci_cell} & {actions_total} & {eta_cell} & {p_cell} \\\\")
            first_row = False
        
        lines.append("\\midrule")
    
    lines[-1] = "\\bottomrule"
    
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\scriptsize")
    lines.append("\\item \\textit{Note.} Dist. = Jina v2 cosine distance from baseline. $\\eta^2_p$ = partial eta-squared (RM-ANOVA). N=9 per prompt.")
    lines.append("\\item 95\\% CI computed using t-distribution ($df=8$) for block-level meta-inference per uncertainty reporting rule.")
    lines.append("\\item p-values are Bonferroni-corrected (2 tests, $\\alpha_{adj}$=0.025). * $p<.05$, ** $p<.01$, *** $p<.001$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    
    latex_content = '\n'.join(lines)
    
    with open(output_dir / 'corrector_sensitivity_table.tex', 'w') as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX table saved to: {output_dir / 'corrector_sensitivity_table.tex'}")
    
    return latex_content


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run corrector sensitivity analysis."""
    
    print("=" * 70)
    print("CORRECTOR PROMPT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Script location: final_runs/supp_prompt_sensitivity/analysis_scripts/
    project_root = Path(__file__).parent.parent.parent.parent
    # RQ1b data is in RQ1b_corrector_ablation/Data/ after data restoration
    base_dir = project_root / "final_runs" / "RQ1b_corrector_ablation" / "Data"
    output_dir = project_root / "final_runs" / "Sensitivity_analysis_simple"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Compute prompt distances (from unique portions to avoid truncation issue)
    print("Computing prompt semantic distances (unique portions)...")
    distances, length_ratios = compute_prompt_distances()
    print("\n  Final distances:")
    for prompt, dist in distances.items():
        print(f"    {prompt}: {dist:.3f} (length: {length_ratios.get(prompt, 1.0):.2f}x)")
    print()
    
    # Extract full results including action counts
    print("Extracting full results (F1 delta + action counts)...")
    full_results = extract_full_results_by_prompt(base_dir)
    
    # Also extract just F1 deltas for backward compatibility
    results = extract_f1_delta_by_prompt(base_dir)
    
    # Aggregate action counts
    action_counts = aggregate_action_counts(full_results)
    
    for exp_type, prompts_data in results.items():
        print(f"\n{exp_type}:")
        for prompt, values in prompts_data.items():
            if values:
                action_info = action_counts.get(exp_type, {}).get(prompt, {})
                actions_total = action_info.get('Actions_Total')
                actions_str = f", actions={int(actions_total)}" if actions_total is not None else ""
                print(f"  {prompt}: n={len(values)}, mean={np.mean(values):+.3f}, std={np.std(values):.3f}{actions_str}")
    print()
    
    # Compute Sobol metrics (including ANOVA)
    print("Computing repeated-measures ANOVA (partial η²)...")
    sobol_metrics = compute_all_sobol_metrics(results, full_results)
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
        f_stat = metrics.get('f_statistic', 0)
        print(f"  {exp}: partial η²={eta:.3f} ({effect}), F({df_eff},{df_err})={f_stat:.2f}, p={p:.4f} ({sig}), n_subj={n_subj}, k={k}")
    print()
    
    # Apply Bonferroni correction (2 tests)
    print("Applying Bonferroni correction (2 tests)...")
    bonferroni_results = bonferroni_correction(p_values, alpha=0.05)
    print(f"  α_adjusted = {bonferroni_results['alpha_adjusted']:.4f}")
    for exp, res in bonferroni_results['results'].items():
        status = "SIG" if res['significant_adjusted'] else "ns"
        print(f"  {exp}: p={res['p_original']:.4f} → p_adj={res['p_adjusted']:.4f} ({status})")
    print()
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(results, distances, p_values, sobol_metrics, output_dir)
    
    # Create sensitivity index bar chart
    print("Creating sensitivity index visualization...")
    create_sensitivity_bar_chart(sobol_metrics, output_dir)
    
    # Generate LaTeX table with action counts and Sobol metrics
    print("Generating LaTeX table...")
    generate_latex_table(results, distances, p_values, sobol_metrics, output_dir, action_counts, bonferroni_results)

    # Generate RM-ANOVA assumption checks table (Shapiro residuals, Mauchly, GG)
    print("Generating RM-ANOVA assumption checks table...")
    _ = generate_assumptions_table(sobol_metrics, output_dir)
    
    # Save data with action counts
    all_data = []
    for exp_type, prompts_data in results.items():
        for prompt, values in prompts_data.items():
            if values:
                # Get action counts for this prompt
                action_info = action_counts.get(exp_type, {}).get(prompt, {})
                
                row = {
                    'Experiment': exp_type,
                    'Prompt': prompt,
                    'Distance': distances.get(prompt, 0),
                    'F1_Delta_Mean': np.mean(values),
                    'F1_Delta_Std': np.std(values),
                    'N': len(values),
                    'Actions_Remove': action_info.get('Actions_Remove'),
                    'Actions_Flip': action_info.get('Actions_Flip'),
                    'Actions_Change': action_info.get('Actions_Change'),
                    'Actions_Revise': action_info.get('Actions_Revise'),
                    'Actions_Total': action_info.get('Actions_Total')
                }
                all_data.append(row)
    
    df = pd.DataFrame(all_data)
    df.to_excel(output_dir / 'corrector_sensitivity_data.xlsx', index=False)
    print(f"✓ Data saved to: {output_dir / 'corrector_sensitivity_data.xlsx'}")
    
    # Print per-prompt action counts summary
    print("\nPER-PROMPT ACTION COUNTS:")
    for exp_type in ['Synthetic', 'Ground Truth']:
        print(f"\n  {exp_type}:")
        for prompt in ['Baseline', 'CoT', 'Mechanistic']:
            action_info = action_counts.get(exp_type, {}).get(prompt, {})
            if action_info:
                total = action_info.get('Actions_Total')
                if total is not None:
                    remove = action_info.get('Actions_Remove', 0) or 0
                    flip = action_info.get('Actions_Flip', 0) or 0
                    change = action_info.get('Actions_Change', 0) or 0
                    revise = action_info.get('Actions_Revise', 0) or 0
                    print(f"    {prompt}: Total={int(total)}, Remove={int(remove)}, Flip={int(flip)}, Change={int(change)}, Revise={int(revise)}")
                else:
                    print(f"    {prompt}: (no action data available)")
    
    print("\n" + "=" * 70)
    print("CORRECTOR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

