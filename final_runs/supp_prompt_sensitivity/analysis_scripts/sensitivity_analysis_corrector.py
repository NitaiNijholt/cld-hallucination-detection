"""
Corrector Prompt Sensitivity Analysis

Creates sensitivity analysis for RQ1b corrector experiments:
- Panel: F1 Δ by prompt for synthetic + ground truth experiments
- LaTeX table with semantic distances, Friedman (prompt effect), and Wilcoxon (efficacy + post-hoc)

References:
- Günther et al. (2023). Jina Embeddings 2: 8192-token general-purpose text embeddings. arXiv:2310.19923
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import yaml
from scipy import stats
from scipy.stats import shapiro, levene

def calc_95ci(values):
    """Calculate 95% CI using t-distribution (appropriate for N=9 blocks)."""
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)  # df=8 for N=9
    return mean - t_crit * sem, mean + t_crit * sem
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# Use Jina embeddings v2 for long-context support (8192 tokens vs MPNet's 384).
# For the thesis-facing analysis we REQUIRE real embeddings; do not silently fall back to zeros.
_JINA_MODEL = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence_transformers is required for corrector prompt distances "
        "(jinaai/jina-embeddings-v2-base-en). Install it and rerun."
    ) from e

def get_jina_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Get embeddings using Jina v2 (8192 token context)."""
    global _JINA_MODEL
    if _JINA_MODEL is None:
        print("  Loading Jina embeddings v2 (8192 tokens)...")
        _JINA_MODEL = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    # Normalize => cosine similarity is a dot product (stable + reproducible)
    return _JINA_MODEL.encode(texts, normalize_embeddings=True)


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
    repo_root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        Path(__file__).parent / "corrector_prompts.yaml",
        repo_root / "final_runs" / "prompts" / "RQ1b_corrector" / "corrector" / "corrector_prompts.yaml",
        repo_root / "final_runs" / "analysis_lib" / "corrector_prompts.yaml",
    ]

    yaml_path = next((p for p in candidate_paths if p.exists()), None)
    if yaml_path is None:
        raise FileNotFoundError(
            "Could not find corrector_prompts.yaml. Looked in: "
            + ", ".join(str(p) for p in candidate_paths)
        )

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    prompts = {
        "Baseline": (data.get("baseline_prompt") or "").strip(),
        "CoT": (data.get("cot_prompt") or "").strip(),
        "Mechanistic": (data.get("mechanistic_prompt") or "").strip(),
    }

    missing = [k for k, v in prompts.items() if not v]
    if missing:
        raise ValueError(f"Missing prompt text(s) in {yaml_path}: {missing}")

    print(f"  Loaded corrector prompts from: {yaml_path}")
    for k in ["Baseline", "CoT", "Mechanistic"]:
        print(f"  {k} prompt length: {len(prompts[k])} chars")

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
    
    baseline = prompts['Baseline']
    cot = prompts['CoT']
    mech = prompts['Mechanistic']
    
    # Get embeddings of FULL prompts using Jina (8192 tokens - no truncation)
    prompt_texts = [baseline, cot, mech]
    embeddings = get_jina_embeddings(prompt_texts)
    
    # Compute distances from baseline
    baseline_emb = embeddings[0]
    distances = {}
    for i, name in enumerate(['Baseline', 'CoT', 'Mechanistic']):
        # normalized embeddings => cosine similarity is dot product
        sim = float(np.dot(baseline_emb, embeddings[i]))
        distances[name] = 1.0 - sim
    
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


def _get_f1_delta_column(df: pd.DataFrame) -> str:
    """Return the column name holding F1 delta values."""
    for col in ["F1_Delta", "F1 Delta", "F1_Delta_Mean", "F1 Δ", "F1_Delta_Post_Pre"]:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find an F1 delta column in df columns: {list(df.columns)}")


def build_block_matrix(
    full_results: Dict[str, Dict[str, pd.DataFrame]],
    exp_type: str,
    prompts: List[str],
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    Build a (n_blocks x k_prompts) matrix of F1 deltas aligned by (CLD, Run).

    This enforces the repeated-measures / blocked design:
    - blocks = (CLD, run)
    - treatments = corrector prompt variant
    """
    if exp_type not in full_results:
        return np.empty((0, len(prompts))), []

    per_prompt_series: Dict[str, pd.Series] = {}
    for prompt in prompts:
        df = full_results[exp_type].get(prompt)
        if df is None or df.empty:
            continue
        if "CLD" not in df.columns or "Run" not in df.columns:
            raise KeyError(f"Expected CLD and Run columns for {exp_type}/{prompt}, got: {list(df.columns)}")

        delta_col = _get_f1_delta_column(df)
        # Reduce to unique block values (if duplicates exist, average them)
        s = (
            df[["CLD", "Run", delta_col]]
            .dropna()
            .groupby(["CLD", "Run"], as_index=True)[delta_col]
            .mean()
        )
        per_prompt_series[prompt] = s

    if len(per_prompt_series) < 2:
        return np.empty((0, len(prompts))), []

    # Intersect blocks across all requested prompts present
    common_index = None
    for prompt in prompts:
        if prompt not in per_prompt_series:
            continue
        common_index = per_prompt_series[prompt].index if common_index is None else common_index.intersection(per_prompt_series[prompt].index)

    if common_index is None or len(common_index) == 0:
        return np.empty((0, len(prompts))), []

    common_index = common_index.sort_values()
    block_keys = [(str(cld), int(run)) for cld, run in common_index.tolist()]

    matrix_cols = []
    used_prompts = []
    for prompt in prompts:
        if prompt not in per_prompt_series:
            continue
        matrix_cols.append(per_prompt_series[prompt].reindex(common_index).to_numpy(dtype=float))
        used_prompts.append(prompt)

    matrix = np.column_stack(matrix_cols) if matrix_cols else np.empty((0, 0))
    if len(used_prompts) != len(prompts):
        # We require a full prompt set for Friedman; otherwise we cannot compare treatments consistently.
        missing = [p for p in prompts if p not in used_prompts]
        raise ValueError(f"Missing prompt(s) for blocked comparison in {exp_type}: {missing}")

    return matrix, block_keys


def compute_friedman_prompt_effect(
    full_results: Dict[str, Dict[str, pd.DataFrame]],
    exp_type: str,
    prompts: List[str],
) -> Dict[str, Any]:
    """Friedman test for prompt effect with (CLD, run) blocks."""
    matrix, blocks = build_block_matrix(full_results, exp_type, prompts)
    n_blocks = matrix.shape[0]
    k = matrix.shape[1]
    if n_blocks < 2 or k < 2:
        return {"p_value": np.nan, "chi2": np.nan, "kendalls_w": np.nan, "n_blocks": int(n_blocks), "k": int(k)}

    chi2, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])
    w = (chi2 / (n_blocks * (k - 1))) if (n_blocks > 0 and k > 1) else np.nan
    return {
        "p_value": float(p),
        "chi2": float(chi2),
        "kendalls_w": float(w),
        "n_blocks": int(n_blocks),
        "k": int(k),
        "block_matrix": matrix,
        "block_keys": blocks,
    }


def compute_one_sample_wilcoxon_vs_zero(values: List[float]) -> Dict[str, Any]:
    """One-sample Wilcoxon signed-rank test of median(values) == 0."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"p_value": np.nan, "stat": np.nan, "n": 0}
    if np.allclose(x, 0.0):
        return {"p_value": 1.0, "stat": 0.0, "n": int(x.size)}
    try:
        stat, p = stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox")
        return {"p_value": float(p), "stat": float(stat), "n": int(x.size)}
    except Exception:
        # Fallback: if wilcoxon fails (e.g. too many ties), return NaNs rather than crashing
        return {"p_value": np.nan, "stat": np.nan, "n": int(x.size)}


def compute_posthoc_wilcoxon_pairs_from_block_matrix(
    block_matrix: np.ndarray,
    prompts: List[str],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Post-hoc paired Wilcoxon signed-rank tests between prompt variants.

    Input:
      - block_matrix: shape (n_blocks, k_prompts), aligned by (CLD, run)
      - prompts: list of prompt names in the same column order as block_matrix

    Output:
      - raw and Bonferroni-adjusted p-values per pair
      - direction based on median paired difference (PromptA - PromptB)
    """
    k = len(prompts)
    if block_matrix.size == 0 or k < 2:
        return {"n_pairs": 0, "alpha": alpha, "alpha_adj": alpha, "pairs": {}, "significant_pairs_adj": []}

    pairs: Dict[str, Any] = {}
    pair_keys: List[str] = []
    for i in range(k):
        for j in range(i + 1, k):
            a = prompts[i]
            b = prompts[j]
            diffs = block_matrix[:, i] - block_matrix[:, j]
            diffs = diffs[~np.isnan(diffs)]
            if diffs.size == 0 or np.allclose(diffs, 0.0):
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")

            median_diff = float(np.median(diffs)) if diffs.size else 0.0
            direction = f"{a}>{b}" if median_diff > 0 else f"{b}>{a}" if median_diff < 0 else f"{a}={b}"
            key = f"{a}__vs__{b}"
            pair_keys.append(key)
            pairs[key] = {
                "a": a,
                "b": b,
                "n": int(diffs.size),
                "stat": float(stat),
                "p_raw": float(p),
                "median_diff": median_diff,
                "direction": direction,
            }

    m = len(pair_keys)
    alpha_adj = alpha / m if m else alpha
    sig_adj: List[str] = []
    for key in pair_keys:
        p_raw = pairs[key]["p_raw"]
        p_adj = min(p_raw * m, 1.0)
        pairs[key]["p_adj"] = float(p_adj)
        pairs[key]["significant_adj"] = bool(p_raw < alpha_adj)
        if p_raw < alpha_adj:
            sig_adj.append(key)

    return {
        "n_pairs": m,
        "alpha": float(alpha),
        "alpha_adj": float(alpha_adj),
        "pairs": pairs,
        "significant_pairs_adj": sig_adj,
    }


def compute_cohens_d_from_delta(delta_values: List[float]) -> float:
    """
    Compute Cohen's d for paired data (pre-post correction).
    
    d = mean(Δ) / SD(Δ)
    
    For corrector experiments, Δ = F1_post - F1_pre for each run.
    
    Interpretation (Cohen, 1988):
    - |d| < 0.20: negligible
    - 0.20 <= |d| < 0.50: small
    - 0.50 <= |d| < 0.80: medium
    - |d| >= 0.80: large
    
    Args:
        delta_values: List of F1 changes (post - pre)
    
    Returns:
        Cohen's d value (can be negative if mean delta is negative)
    """
    if len(delta_values) < 2:
        return np.nan
    mean_delta = np.mean(delta_values)
    sd_delta = np.std(delta_values, ddof=1)  # Use sample SD (n-1)
    if sd_delta == 0:
        return np.nan
    return mean_delta / sd_delta


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size (Cohen, 1988).
    
    Args:
        d: Cohen's d value
    
    Returns:
        Interpretation string
    """
    if np.isnan(d):
        return "—"
    d_abs = abs(d)
    if d_abs < 0.20:
        return "negligible"
    elif d_abs < 0.50:
        return "small"
    elif d_abs < 0.80:
        return "medium"
    else:
        return "large"


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
        return {
            'method': 'Bonferroni',
            'n_tests': 0,
            'alpha_original': alpha,
            'alpha_adjusted': alpha,
            'tests': {},
            'n_significant_original': 0,
            'n_significant_adjusted': 0
        }
    alpha_adj = alpha / m
    
    tests = {}
    for name, p in p_values.items():
        p_adj = min(p * m, 1.0)  # Adjusted p-value
        tests[name] = {
            'p_original': p,
            'p_adjusted': p_adj,
            'significant_original': p < alpha,
            'significant_adjusted': p < alpha_adj,
            'changed': (p < alpha) != (p < alpha_adj)
        }
    
    return {
        'method': 'Bonferroni',
        'n_tests': m,
        'alpha_original': alpha,
        'alpha_adjusted': alpha_adj,
        'tests': tests,
        'n_significant_original': sum(1 for r in tests.values() if r['significant_original']),
        'n_significant_adjusted': sum(1 for r in tests.values() if r['significant_adjusted'])
    }


def compute_eta_squared(groups: List[np.ndarray]) -> float:
    """
    Compute eta-squared (η²) effect size for ANOVA.
    
    η² = SS_between / SS_total
    """
    if len(groups) < 2:
        return np.nan
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total if ss_total > 0 else 0.0


def compute_cohens_f(eta_squared: float) -> float:
    """Convert η² to Cohen's f."""
    if np.isnan(eta_squared) or eta_squared >= 1.0:
        return np.nan
    if eta_squared <= 0.0:
        return 0.0
    return np.sqrt(eta_squared / (1 - eta_squared))


def test_anova_assumptions(groups_dict: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Test ANOVA assumptions: normality (Shapiro-Wilk) and homogeneity of variance (Levene's).
    
    Returns dict with 'normality' (per group) and 'homogeneity' results.
    """
    results = {'normality': {}, 'homogeneity': {}, 'assumptions_met': True}
    
    # Normality per group (Shapiro-Wilk)
    all_normal = True
    for name, vals in groups_dict.items():
        if len(vals) >= 3:
            stat, p = shapiro(vals)
            is_normal = p > 0.05
            results['normality'][name] = {'W': stat, 'p': p, 'normal': is_normal}
            if not is_normal:
                all_normal = False
        else:
            results['normality'][name] = {'W': np.nan, 'p': np.nan, 'normal': None}
    
    # Homogeneity of variance (Levene's test)
    groups_list = [v for v in groups_dict.values() if len(v) >= 2]
    if len(groups_list) >= 2:
        lev_stat, lev_p = levene(*groups_list)
        equal_var = lev_p > 0.05
        results['homogeneity'] = {'W': lev_stat, 'p': lev_p, 'equal_var': equal_var}
        if not equal_var:
            results['assumptions_met'] = False
    else:
        results['homogeneity'] = {'W': np.nan, 'p': np.nan, 'equal_var': None}
    
    results['assumptions_met'] = all_normal and results['homogeneity'].get('equal_var', True)
    
    return results


def compute_corrector_effect_sizes(
    results: Dict[str, Dict[str, List[float]]],
    full_results: Dict[str, Dict[str, pd.DataFrame]],
    prompts: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Compute all effect sizes for corrector experiments.
    
    Returns dict with:
    - Prompt effect (Friedman): p-value, Kendall's W, n_blocks
    - Efficacy (one-sample Wilcoxon vs 0): p-values per prompt
    - Per-prompt: Cohen's d for each prompt's delta values (magnitude only)
    - (Legacy) ANOVA assumption checks are no longer thesis-critical and are not used for inference.
    """
    effect_sizes = {}

    prompt_list = prompts or ["Baseline", "CoT", "Mechanistic"]

    for exp_type, prompts_data in results.items():
        exp_stats = {
            'friedman_p': np.nan,
            'kendalls_w': np.nan,
            'friedman_chi2': np.nan,
            'n_blocks': 0,
            'k_groups': len(prompt_list),
            'n_total': sum(len(v) for v in prompts_data.values()),
            'per_prompt': {},
            'efficacy_vs_zero': {},
        }

        # Prompt effect (Friedman with CLD×run blocks)
        fried = compute_friedman_prompt_effect(full_results, exp_type, prompt_list)
        exp_stats['friedman_p'] = fried.get("p_value", np.nan)
        exp_stats['friedman_chi2'] = fried.get("chi2", np.nan)
        exp_stats['kendalls_w'] = fried.get("kendalls_w", np.nan)
        exp_stats['n_blocks'] = fried.get("n_blocks", 0)
        # Post-hoc pairwise Wilcoxon between prompts (paired within blocks)
        block_matrix = fried.get("block_matrix", np.empty((0, len(prompt_list))))
        exp_stats['posthoc_wilcoxon'] = compute_posthoc_wilcoxon_pairs_from_block_matrix(
            block_matrix, prompt_list, alpha=0.05
        )
        
        # Per-prompt Cohen's d
        for prompt in prompt_list:
            values = prompts_data.get(prompt, [])
            if not values:
                continue
            d = compute_cohens_d_from_delta(values)
            exp_stats['per_prompt'][prompt] = {
                'cohens_d': d,
                'cohens_d_interp': interpret_cohens_d(d),
                'mean_delta': float(np.mean(values)),
                'sd_delta': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'n': int(len(values))
            }

            # Efficacy vs 0 (one-sample Wilcoxon)
            exp_stats['efficacy_vs_zero'][prompt] = compute_one_sample_wilcoxon_vs_zero(values)
        
        effect_sizes[exp_type] = exp_stats
    
    return effect_sizes


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(
    results: Dict[str, Dict[str, List[float]]],
    distances: Dict[str, float],
    effect_sizes: Dict[str, Dict],
    output_dir: Path,
):
    """Create corrector sensitivity figure."""
    
    # Apply Bonferroni correction across the two Friedman prompt-effect tests (Synthetic, Ground Truth)
    p_values_valid = {
        exp: float(stats_dict.get("friedman_p", np.nan))
        for exp, stats_dict in effect_sizes.items()
        if stats_dict.get("friedman_p", np.nan) == stats_dict.get("friedman_p", np.nan)  # not NaN
    }
    bonf_results = bonferroni_correction(p_values_valid, alpha=0.05) if p_values_valid else None
    n_tests = bonf_results['n_tests'] if bonf_results else 2
    
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
        
        # Add prompt-effect annotation (Friedman + Kendall's W), with Bonferroni-adjusted p
        if bonf_results and exp_type in bonf_results['tests']:
            p_adj = bonf_results['tests'][exp_type]['p_adjusted']
            sig_str = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
            w = effect_sizes.get(exp_type, {}).get("kendalls_w", np.nan)
            w_str = f"W={w:.2f}" if w == w else "W=—"
            ax.text(
                0.98,
                0.95,
                f"Friedman p_adj={p_adj:.3f} ({sig_str}), {w_str}",
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=10,
                style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_xticks(x_pos)
        # Add distances to x-axis labels
        xlabels = [f"{p}\n(d={distances.get(p, 0):.3f})" for p in prompts]
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_ylabel('F1 Δ (Post - Pre Correction)', fontsize=11)
        ax.set_title(exp_titles.get(exp_type, exp_type), fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=prompt)
                       for prompt, color in PROMPT_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle(
        f'Corrector Prompt Sensitivity: F1 Change by Prompt Type\n(Friedman test with CLD×run blocking; Bonferroni n={n_tests}: * p_adj<.05, ** p_adj<.01, *** p_adj<.001, ns = not significant)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'corrector_sensitivity_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'corrector_sensitivity_figure.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Figure saved to: {output_dir / 'corrector_sensitivity_figure.png'}")


def generate_latex_table(
    results: Dict[str, Dict[str, List[float]]],
    distances: Dict[str, float],
    length_ratios: Dict[str, float],
    output_dir: Path,
    effect_sizes: Dict[str, Dict] = None,
) -> str:
    """Generate thesis-facing LaTeX table for corrector sensitivity (includes Distance + Length)."""
    
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Corrector Prompt Sensitivity Analysis: F1 Change by Prompt Type}")
    lines.append("\\label{tab:corrector_sensitivity}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llccccccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Experiment} & \\textbf{Prompt} & \\textbf{Distance} & \\textbf{Length} & \\textbf{F1 $\\Delta$} & \\textbf{Std} & \\textbf{95\\% CI} & \\textbf{N} & \\textbf{Efficacy $p$} & \\textbf{Friedman $p$} & \\textbf{W} \\\\")
    lines.append("\\midrule")
    
    for exp_type in ['Synthetic', 'Ground Truth']:
        exp_data = results.get(exp_type, {})
        fried_p = np.nan
        w = np.nan
        if effect_sizes and exp_type in effect_sizes:
            fried_p = effect_sizes[exp_type].get('friedman_p', np.nan)
            w = effect_sizes[exp_type].get('kendalls_w', np.nan)
        
        first_row = True
        for prompt in ['Baseline', 'CoT', 'Mechanistic']:
            if prompt not in exp_data:
                continue
            
            values = exp_data[prompt]
            if len(values) == 0:
                continue
            
            exp_name = exp_type if first_row else ''
            dist_cell = f"{distances.get(prompt, 0.0):.3f}"
            lr_cell = f"{length_ratios.get(prompt, 1.0):.2f}$\\times$"
            f1_delta = f"{np.mean(values):+.3f}"
            std = f"{np.std(values):.3f}"
            
            # Calculate 95% CI (t-distribution for N=9 blocks)
            ci_low, ci_high = calc_95ci(values)
            ci_cell = f"[{ci_low:+.3f}, {ci_high:+.3f}]" if not (np.isnan(ci_low) or np.isnan(ci_high)) else "—"
            
            n = str(len(values))

            # Efficacy vs 0 p-value (one-sample Wilcoxon)
            eff_p = np.nan
            if effect_sizes and exp_type in effect_sizes:
                eff_p = effect_sizes[exp_type].get("efficacy_vs_zero", {}).get(prompt, {}).get("p_value", np.nan)
            eff_cell = f"{eff_p:.3f}" if eff_p == eff_p else "—"
            
            if first_row:
                p_cell = f"{fried_p:.3f}" if fried_p == fried_p else "—"
                w_cell = f"{w:.2f}" if w == w else "—"
            else:
                p_cell = ""
                w_cell = ""

            lines.append(f"{exp_name} & {prompt} & {dist_cell} & {lr_cell} & {f1_delta} & {std} & {ci_cell} & {n} & {eff_cell} & {p_cell} & {w_cell} \\\\")
            first_row = False
        
        lines.append("\\midrule")
    
    lines[-1] = "\\bottomrule"
    
    lines.append("\\end{tabular}")
    lines.append("}%")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Note.} Prompt effect: Friedman test with (CLD, run) blocking ($n=9$ blocks) compares prompt variants on matched CLD$\\times$run instances; W = Kendall's W effect size (0.1 small, 0.3 medium, 0.5 large).")
    lines.append("\\item 95\\% CI computed using t-distribution ($df=8$) for block-level meta-inference per uncertainty reporting rule.")
    lines.append("\\item Efficacy $p$: one-sample Wilcoxon signed-rank test of median(F1 $\\Delta$) = 0 for each prompt (two-sided).")
    lines.append("\\item \\textit{Post-hoc:} Pairwise Wilcoxon signed-rank tests compare prompt pairs within the same blocks; Bonferroni correction over 3 pairs per experiment ($\\alpha_{\\text{adj}}=0.0167$).")
    lines.append("\\item Distance = Jina v2 cosine distance from baseline prompt (8192-token context). Length = prompt length ratio (cumulative design). F1 $\\Delta$ = post- minus pre-correction.")
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
    
    final_runs = Path(__file__).parent.parent.parent  # final_runs
    base_dir = final_runs / "RQ1b_corrector_ablation" / "Data"  # Where corrector experiment data lives
    output_dir = final_runs / "Sensitivity_analysis_simple"
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
    
    # Compute blocked prompt effects (Friedman) + efficacy vs 0 (Wilcoxon) + effect sizes
    print("Computing prompt-effect (Friedman) + efficacy (Wilcoxon vs 0) + effect sizes...")
    prompts = ["Baseline", "CoT", "Mechanistic"]
    effect_sizes = compute_corrector_effect_sizes(results, full_results, prompts=prompts)

    for exp, st in effect_sizes.items():
        p = st.get("friedman_p", np.nan)
        w = st.get("kendalls_w", np.nan)
        n_blocks = st.get("n_blocks", 0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {exp}: Friedman p = {p:.4f}{sig}, W = {w:.3f}, n_blocks={n_blocks}")

        # Per-prompt: Cohen's d + efficacy p (vs 0)
        for prompt in prompts:
            pstats = st.get("per_prompt", {}).get(prompt, {})
            if not pstats:
                continue
            d = pstats.get("cohens_d", np.nan)
            d_interp = pstats.get("cohens_d_interp", "—")
            eff_p = st.get("efficacy_vs_zero", {}).get(prompt, {}).get("p_value", np.nan)
            eff_sig = '***' if eff_p < 0.001 else '**' if eff_p < 0.01 else '*' if eff_p < 0.05 else ''
            print(f"    {prompt}: Cohen's d = {d:+.3f} ({d_interp}); Wilcoxon vs 0 p={eff_p:.4f}{eff_sig}")

        # Print post-hoc prompt pair comparisons
        posthoc = st.get("posthoc_wilcoxon", {})
        if posthoc and posthoc.get("n_pairs", 0) > 0:
            alpha_adj = posthoc.get("alpha_adj", 0.05)
            print(f"    Post-hoc Wilcoxon pairs (Bonferroni α_adj={alpha_adj:.4f}):")
            for key, r in posthoc.get("pairs", {}).items():
                print(
                    f"      {r['a']} vs {r['b']}: p={r['p_raw']:.4f}, p_adj={r['p_adj']:.4f}, dir={r['direction']}"
                )
    print()
    
    # Apply Bonferroni correction for multiple comparisons within RQ1b
    print("Applying Bonferroni correction across the 2 Friedman prompt-effect tests (RQ1b family)...")
    friedman_p_values = {exp: float(st.get("friedman_p", np.nan)) for exp, st in effect_sizes.items() if st.get("friedman_p", np.nan) == st.get("friedman_p", np.nan)}
    correction_results = bonferroni_correction(friedman_p_values, alpha=0.05)
    
    print(f"  α_original = {correction_results['alpha_original']:.3f}")
    print(f"  α_adjusted = {correction_results['alpha_adjusted']:.4f} (Bonferroni, m={correction_results['n_tests']})")
    print(f"  Significant before correction: {correction_results['n_significant_original']}")
    print(f"  Significant after correction: {correction_results['n_significant_adjusted']}")
    print()
    
    for exp, res in correction_results['tests'].items():
        status = "STILL SIG" if res['significant_adjusted'] else "NOW NS" if res['changed'] else "NS"
        print(f"  {exp}: p_raw={res['p_original']:.4f}, p_adj={res['p_adjusted']:.4f} ({status})")
    print()
    
    # Store correction results in effect_sizes for output
    for exp, res in correction_results['tests'].items():
        if exp in effect_sizes:
            effect_sizes[exp]['p_adjusted'] = res['p_adjusted']
            effect_sizes[exp]['significant_adjusted'] = res['significant_adjusted']
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(results, distances, effect_sizes, output_dir)
    
    # Generate thesis-facing LaTeX table (includes Distance + Length)
    print("Generating LaTeX table...")
    generate_latex_table(results, distances, length_ratios, output_dir, effect_sizes=effect_sizes)
    
    # Save data with action counts
    all_data = []
    for exp_type, prompts_data in results.items():
        for prompt, values in prompts_data.items():
            if values:
                # Get action counts for this prompt
                action_info = action_counts.get(exp_type, {}).get(prompt, {})
                
                # Get effect sizes for this prompt
                prompt_effects = effect_sizes.get(exp_type, {}).get('per_prompt', {}).get(prompt, {})
                exp_effects = effect_sizes.get(exp_type, {})
                
                row = {
                    'Experiment': exp_type,
                    'Prompt': prompt,
                    'Distance': distances.get(prompt, 0),
                    'F1_Delta_Mean': np.mean(values),
                    'F1_Delta_Std': np.std(values),
                    'F1_Delta_Min': np.min(values),
                    'F1_Delta_Max': np.max(values),
                    'F1_Delta_Values': values,  # Store individual values for reproducibility
                    'Cohens_d': prompt_effects.get('cohens_d', np.nan),
                    'Cohens_d_Interp': prompt_effects.get('cohens_d_interp', '—'),
                    'Eta_Squared': exp_effects.get('eta_squared', np.nan),
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
    
    # Save comprehensive statistics JSON (per uncertainty reporting rule)
    import json
    comprehensive_stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }
    
    for row in all_data:
        exp = row['Experiment']
        prompt = row['Prompt']
        
        if exp not in comprehensive_stats['experiments']:
            comprehensive_stats['experiments'][exp] = {
                'prompts': {},
                'friedman_p': float(effect_sizes.get(exp, {}).get('friedman_p', np.nan)),
                'kendalls_w': float(effect_sizes.get(exp, {}).get('kendalls_w', np.nan)),
                'friedman_p_adj': float(effect_sizes.get(exp, {}).get('p_adjusted', np.nan)),
                'n_blocks': int(effect_sizes.get(exp, {}).get('n_blocks', 0)),
                'posthoc_wilcoxon': effect_sizes.get(exp, {}).get('posthoc_wilcoxon', {}),
            }
        
        values = row.get('F1_Delta_Values', [])
        efficacy_p = effect_sizes.get(exp, {}).get("efficacy_vs_zero", {}).get(prompt, {}).get("p_value", np.nan)
        comprehensive_stats['experiments'][exp]['prompts'][prompt] = {
            'n': int(row['N']),
            'values': [float(v) for v in values] if values else [],
            'mean': float(row['F1_Delta_Mean']),
            'std': float(row['F1_Delta_Std']),
            'min': float(row['F1_Delta_Min']),
            'max': float(row['F1_Delta_Max']),
            'distance': float(row['Distance']),
            'efficacy_p_vs_zero': float(efficacy_p) if efficacy_p == efficacy_p else None,
            'cohens_d': float(row['Cohens_d']) if not np.isnan(row['Cohens_d']) else None,
            'latex_primary': f"{row['F1_Delta_Mean']:+.3f} [{row['F1_Delta_Min']:+.3f}, {row['F1_Delta_Max']:+.3f}]"
        }
    
    with open(output_dir / 'corrector_comprehensive_stats.json', 'w') as f:
        json.dump(comprehensive_stats, f, indent=2, default=str)
    print(f"✓ Comprehensive stats saved to: {output_dir / 'corrector_comprehensive_stats.json'}")
    
    print("\n" + "=" * 70)
    print("CORRECTOR ANALYSIS COMPLETE")
    print("=" * 70)


def run_assumption_tests_only():
    """Run only the ANOVA assumption tests for corrector experiments."""
    import json
    
    print("="*70)
    print("RQ1b CORRECTOR ANOVA ASSUMPTION TESTS (Standalone)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_dir = Path(__file__).resolve().parent.parent.parent  # /home/nitai/code/causalix.ai/final_runs
    output_dir = base_dir / "Sensitivity_analysis_simple"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract F1 delta data
    print("Extracting F1 delta data...")
    results = extract_f1_delta_by_prompt(base_dir)
    
    # Compute effect sizes including assumption tests
    print("Computing ANOVA statistics and assumption tests...")
    effect_sizes = compute_corrector_effect_sizes(results)
    
    # Print results
    print("\nANOVA Results:")
    print("-" * 70)
    for exp_type, stats in effect_sizes.items():
        p = stats['p_value']
        eta = stats['eta_squared']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {exp_type}: p={p:.4f}{sig}, η²={eta:.3f}")
    
    print("\nAssumption Tests:")
    print("-" * 70)
    
    results_summary = []
    for exp_type, stats in effect_sizes.items():
        assumptions = stats.get('assumptions', {})
        normality = assumptions.get('normality', {})
        homogeneity = assumptions.get('homogeneity', {})
        
        print(f"\n{exp_type}:")
        print("  Normality (Shapiro-Wilk):")
        
        all_normal = True
        for group, norm_stats in normality.items():
            if norm_stats.get('p') is not None and not np.isnan(norm_stats.get('p', np.nan)):
                status = "normal" if norm_stats['normal'] else "NON-NORMAL"
                if not norm_stats['normal']:
                    all_normal = False
                print(f"    {group}: W={norm_stats['W']:.4f}, p={norm_stats['p']:.4f} ({status})")
        
        if homogeneity.get('p') is not None and not np.isnan(homogeneity.get('p', np.nan)):
            hom_status = "equal" if homogeneity['equal_var'] else "UNEQUAL"
            print(f"  Homogeneity (Levene's): W={homogeneity['W']:.4f}, p={homogeneity['p']:.4f} ({hom_status})")
        
        met = assumptions.get('assumptions_met', None)
        print(f"  Assumptions met: {'Yes' if met else 'No'}")
        
        results_summary.append({
            'Experiment': exp_type,
            'ANOVA_p': stats['p_value'],
            'eta_squared': stats['eta_squared'],
            'all_normal': all_normal,
            'levene_p': homogeneity.get('p', np.nan),
            'equal_variance': homogeneity.get('equal_var', None),
            'assumptions_met': met
        })
    
    # Save to file
    output_file = output_dir / 'rq1b_assumption_tests.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("ASSUMPTION TESTS COMPLETE")
    print("="*70)
    
    return results_summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--assumptions-only':
        run_assumption_tests_only()
    else:
        main()

