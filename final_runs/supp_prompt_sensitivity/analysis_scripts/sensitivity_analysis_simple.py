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
from typing import Any, Dict, List, Tuple
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
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import sys

# Add project root(s) to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))           # /.../final_runs
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))   # /home/nitai/code/causalix.ai
from data_science.logit_metrics import get_embedding_local


# =============================================================================
# BONFERRONI CORRECTION HELPER
# =============================================================================

def bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction to a family of p-values.
    
    Args:
        p_values: Dictionary mapping test names to raw p-values
        alpha: Family-wise error rate to control (default 0.05)
    
    Returns:
        Dictionary with correction results including adjusted p-values
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return {
            'n_tests': 0,
            'alpha_original': float(alpha),
            'alpha_adjusted': float(alpha),
            'tests': {},
            'n_significant_original': 0,
            'n_significant_adjusted': 0
        }
    adjusted_alpha = alpha / n_tests
    
    results = {
        'n_tests': n_tests,
        'alpha_original': float(alpha),
        'alpha_adjusted': float(adjusted_alpha),
        'tests': {}
    }
    
    n_sig_original = 0
    n_sig_adjusted = 0
    
    for name, p in p_values.items():
        p_adjusted = min(float(p) * n_tests, 1.0)  # Bonferroni-adjusted p-value
        sig_original = bool(p < alpha)
        sig_adjusted = bool(p_adjusted < alpha)
        
        if sig_original:
            n_sig_original += 1
        if sig_adjusted:
            n_sig_adjusted += 1
        
        results['tests'][name] = {
            'p_original': float(p),
            'p_adjusted': float(p_adjusted),
            'significant_original': sig_original,
            'significant_adjusted': sig_adjusted,
            'changed': sig_original != sig_adjusted
        }
    
    results['n_significant_original'] = n_sig_original
    results['n_significant_adjusted'] = n_sig_adjusted
    
    return results


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    'Corr. Citation': {
        'name': 'RQ1a_gt_synth_citation',
        'judge_type': 'citation',
        'is_ground_truth': False,
        'prompts': {
            'Baseline': ('prompts_citation_baseline.yaml', 'judgeCitation'),
            'CoT': ('prompts_citation_cot.yaml', 'judgeCitation'),
            'Mechanistic': ('prompts_citation_mechanistic.yaml', 'judgeCitation'),
            'Mechanistic-Lit': ('prompts_citation_mechanistic_lit.yaml', 'judgeCitation')
        }
    },
    'Corr. Correctness': {
        'name': 'RQ1a_gt_synth_correctness',
        'judge_type': 'correctness',
        'is_ground_truth': False,
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
        'prompts': {
            'Baseline': ('prompts_citation_baseline.yaml', 'judgeCitation'),
            'CoT': ('prompts_citation_cot.yaml', 'judgeCitation'),
            'Mechanistic': ('prompts_citation_mechanistic.yaml', 'judgeCitation'),
            'Mechanistic-Lit': ('prompts_citation_mechanistic_lit.yaml', 'judgeCitation')
        }
    },
    'GT Correctness': {
        'name': 'RQ1a_gt_lit_correctness',
        'judge_type': 'correctness',
        'is_ground_truth': True,
        'prompts': {
            'Baseline': ('prompts_correctness_baseline.yaml', 'judgeCorrectness'),
            'CoT': ('prompts_correctness_cot.yaml', 'judgeCorrectness'),
            'Mechanistic': ('prompts_correctness_mechanistic.yaml', 'judgeCorrectness')
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

# Prompt colors and markers
PROMPT_COLORS = {
    'Baseline': '#3498db',
    'CoT': '#2ecc71',
    'Mechanistic': '#e74c3c',
    'Mechanistic-Lit': '#f39c12'
}

# Markers for prompt types (kept for potential scatter plots)
PROMPT_MARKERS = {
    'Baseline': 'o',
    'CoT': 's',
    'Mechanistic': '^',
    'Mechanistic-Lit': 'v'
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
    """Load prompts and compute embeddings.

    Primary: MPNet via `get_embedding_local` (requires `sentence_transformers`).
    Fallback: TF-IDF embeddings (keeps the script runnable in minimal envs).
    """
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

    # Fallback: if local embedding model isn't available, use TF-IDF vectors.
    if not embeddings or len(embeddings) != len(prompt_names):
        try:
            vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            X = vec.fit_transform(prompt_texts)  # sparse
            # L2-normalize so cosine similarity is well-defined
            norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
            norms[norms == 0] = 1.0
            X = X.multiply(1.0 / norms[:, None])
            X = X.toarray()
            embeddings = {name: X[i] for i, name in enumerate(prompt_names)}
            print("  NOTE: Using TF-IDF fallback embeddings (sentence_transformers not available).")
        except Exception:
            return {}

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
            
            # y_pred: Does judge predict hallucination?
            # Use score-threshold definition (consistent with main RQ1 analysis): Aggregate Score <= 0.5 => hallucination
            if 'Aggregate Score' in df.columns:
                score = pd.to_numeric(df['Aggregate Score'], errors='coerce')
                mask = y_true.notna() & score.notna()
                y_true = y_true[mask]
                y_pred = (score[mask] <= 0.5).astype(int)
            else:
                # Fallback: verdict-based mapping
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
            
            y_true_raw = df['Is Corrupted']
            # Some runs contain NaNs in Is Corrupted; drop missing rows rather than failing the whole file.
            if 'Aggregate Score' in df.columns:
                score = pd.to_numeric(df['Aggregate Score'], errors='coerce')
                mask = y_true_raw.notna() & score.notna()
                y_true = y_true_raw[mask].astype(int)
                y_pred = (score[mask] <= 0.5).astype(int)
            else:
                verdict = df['Judge Verdict'].astype(str).str.lower().str.strip()
                mask = y_true_raw.notna()
                y_true = y_true_raw[mask].astype(int)
                verdict = verdict[mask]
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


def extract_f1_per_prompt(exp_dir: Path, judge_type: str, is_ground_truth: bool = False) -> Dict[str, List[float]]:
    """Extract F1 scores for each prompt across all CLDs and runs."""
    f1_by_prompt = {}
    
    cld_dirs = [d for d in exp_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('enhanced') and not d.name.startswith('.')]
    
    for cld_dir in cld_dirs:
        run_dirs = [d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        for run_dir in run_dirs:
            # Skip backup files
            excel_files = [f for f in run_dir.glob('judged_*.xlsx') 
                          if not f.name.endswith('.backup.xlsx')]
            
            for excel_file in excel_files:
                fname_lower = excel_file.name.lower()
                if 'baseline' in fname_lower:
                    prompt = 'Baseline'
                elif 'mechanistic_lit' in fname_lower or 'mech_lit' in fname_lower or 'mechanistic-lit' in fname_lower:
                    prompt = 'Mechanistic-Lit'
                elif 'mechanistic' in fname_lower:
                    prompt = 'Mechanistic'
                elif 'cot' in fname_lower:
                    prompt = 'CoT'
                else:
                    continue
                
                f1 = compute_f1_from_file(excel_file, judge_type, is_ground_truth)
                if not np.isnan(f1):
                    if prompt not in f1_by_prompt:
                        f1_by_prompt[prompt] = []
                    f1_by_prompt[prompt].append(f1)
    
    return f1_by_prompt


# =============================================================================
# BLOCKED (REPEATED-MEASURES) PROMPT-EFFECT TESTS (Friedman + Wilcoxon)
# =============================================================================

def extract_f1_by_prompt_by_block(
    exp_dir: Path,
    judge_type: str,
    prompts: List[str],
    is_ground_truth: bool = False,
) -> Dict[str, Dict[Tuple[str, int], float]]:
    """Extract per-block F1 values for each prompt.

    Block = (CLD_name, run_num). If multiple judged files match a prompt within a block
    (e.g., reruns), we keep the newest file (by mtime) to avoid double-counting.
    """
    prompts_set = set(prompts)
    best: Dict[str, Dict[Tuple[str, int], Tuple[float, float]]] = {p: {} for p in prompts}  # (mtime, f1)

    cld_dirs = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and not d.name.startswith("enhanced") and not d.name.startswith(".")
    ]

    for cld_dir in cld_dirs:
        cld_name = cld_dir.name
        run_dirs = [d for d in cld_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

        for run_dir in run_dirs:
            try:
                run_num = int(run_dir.name.split("_", 1)[1])
            except Exception:
                continue

            block = (cld_name, run_num)
            excel_files = [f for f in run_dir.glob("judged_*.xlsx") if not f.name.endswith(".backup.xlsx")]

            for excel_file in excel_files:
                fname_lower = excel_file.name.lower()
                if "baseline" in fname_lower:
                    prompt = "Baseline"
                elif "mechanistic_lit" in fname_lower or "mech_lit" in fname_lower or "mechanistic-lit" in fname_lower:
                    prompt = "Mechanistic-Lit"
                elif "mechanistic" in fname_lower:
                    prompt = "Mechanistic"
                elif "cot" in fname_lower:
                    prompt = "CoT"
                else:
                    continue

                if prompt not in prompts_set:
                    continue

                f1 = compute_f1_from_file(excel_file, judge_type, is_ground_truth)
                if np.isnan(f1):
                    continue
                try:
                    mtime = float(excel_file.stat().st_mtime)
                except Exception:
                    mtime = 0.0

                prev = best[prompt].get(block)
                if (prev is None) or (mtime >= prev[0]):
                    best[prompt][block] = (mtime, float(f1))

    out: Dict[str, Dict[Tuple[str, int], float]] = {p: {} for p in prompts}
    for prompt, by_block in best.items():
        for block, (mtime, f1) in by_block.items():
            out[prompt][block] = float(f1)
    return out


def build_block_matrix(
    f1_by_prompt_by_block: Dict[str, Dict[Tuple[str, int], float]],
    prompts: List[str],
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Create an (n_blocks x k) matrix, dropping blocks missing any prompt."""
    blocks = set()
    for p in prompts:
        blocks |= set(f1_by_prompt_by_block.get(p, {}).keys())

    complete_blocks = [b for b in blocks if all(b in f1_by_prompt_by_block.get(p, {}) for p in prompts)]
    complete_blocks.sort(key=lambda x: (x[0], x[1]))

    if not complete_blocks:
        return np.empty((0, len(prompts))), []

    mat = np.zeros((len(complete_blocks), len(prompts)), dtype=float)
    for i, b in enumerate(complete_blocks):
        for j, p in enumerate(prompts):
            mat[i, j] = float(f1_by_prompt_by_block[p][b])
    return mat, complete_blocks


def friedman_test_from_block_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """Friedman test for prompt effect with (CLD, run) blocks; effect size = Kendall's W."""
    n_blocks = int(matrix.shape[0])
    k = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    if n_blocks < 2 or k < 2:
        return {"p_value": np.nan, "chi2": np.nan, "kendalls_w": np.nan, "n_blocks": n_blocks, "k": k}

    chi2, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])
    w = float(chi2) / float(n_blocks * (k - 1)) if n_blocks * (k - 1) > 0 else np.nan
    return {
        "p_value": float(p),
        "chi2": float(chi2),
        "kendalls_w": float(w),
        "n_blocks": n_blocks,
        "k": k,
    }


def posthoc_wilcoxon_pairs_from_block_matrix(
    matrix: np.ndarray,
    prompts: List[str],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """Post-hoc paired Wilcoxon signed-rank tests between prompt pairs (within blocks).

    Bonferroni correction is applied within each experiment across the number of pairs.
    """
    n_blocks, k = matrix.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    m = len(pairs)
    alpha_adj = alpha / m if m > 0 else alpha

    out: Dict[str, dict] = {"m_pairs": int(m), "alpha_adj": float(alpha_adj), "pairs": {}}
    for i, j in pairs:
        a, b = prompts[i], prompts[j]
        diffs = matrix[:, i] - matrix[:, j]
        try:
            stat, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            p = float(p)
        except Exception:
            stat, p = np.nan, np.nan

        p_adj = min(p * m, 1.0) if not np.isnan(p) and m > 0 else np.nan
        out["pairs"][f"{a} vs {b}"] = {
            "p_value": float(p) if not np.isnan(p) else np.nan,
            "p_adj": float(p_adj) if not np.isnan(p_adj) else np.nan,
            "significant_adj": bool(p_adj < alpha) if not np.isnan(p_adj) else False,
            "n_blocks": int(n_blocks),
        }
    return out


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
            distances = {prompt: np.nan for prompt in config['prompts'].keys()}
        else:
            distances = compute_distance_from_baseline(embeddings)
        
        # Extract blocked F1 scores (one value per (CLD, run) block per prompt)
        prompts = list(config['prompts'].keys())
        f1_blocked = extract_f1_by_prompt_by_block(exp_path, config['judge_type'], prompts, is_ground_truth)
        mat, blocks = build_block_matrix(f1_blocked, prompts)
        if len(blocks) == 0:
            print(f"  WARNING: No complete blocked data found (need all prompts per (CLD, run))")
            continue

        # Get baseline F1 for delta calculation (same complete blocks)
        baseline_vals = [f1_blocked['Baseline'][b] for b in blocks] if 'Baseline' in f1_blocked else [np.nan]
        baseline_f1 = float(np.mean(baseline_vals))
        
        # Compile data for each prompt
        for prompt in prompts:
            if prompt not in f1_blocked:
                continue

            f1_values = [float(f1_blocked[prompt][b]) for b in blocks]
            f1_mean = float(np.mean(f1_values))
            f1_std = float(np.std(f1_values))
            f1_min = float(np.min(f1_values))
            f1_max = float(np.max(f1_values))
            delta_f1 = float(f1_mean - baseline_f1)
            
            # Calculate 95% CI using t-distribution (for N=9 blocks)
            ci_low, ci_high = calc_95ci(f1_values)
            
            all_data.append({
                'Experiment': exp_label,
                'Prompt': prompt,
                'Distance': distances.get(prompt, np.nan),
                'F1': f1_mean,
                'F1 Std': f1_std,
                'F1 CI Low': float(ci_low) if not np.isnan(ci_low) else f1_mean,
                'F1 CI High': float(ci_high) if not np.isnan(ci_high) else f1_mean,
                'F1 Min': f1_min,
                'F1 Max': f1_max,
                'ΔF1': delta_f1,
                'N': int(len(f1_values)),
                'F1 Values': list(f1_values)  # Store individual values for reproducibility
            })
    
    return pd.DataFrame(all_data)


def compute_eta_squared(groups: list) -> float:
    """
    Compute eta-squared (η²) effect size for ANOVA.
    
    η² = SS_between / SS_total
    
    Interpretation:
    - η² < 0.01: negligible
    - 0.01 <= η² < 0.06: small
    - 0.06 <= η² < 0.14: medium
    - η² >= 0.14: large
    
    Args:
        groups: List of arrays, one per group
    
    Returns:
        eta-squared value (0.0 to 1.0)
    """
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total if ss_total > 0 else 0.0


def compute_cohens_f(eta_squared: float) -> float:
    """
    Convert η² to Cohen's f effect size for ANOVA.
    
    f = sqrt(η² / (1 - η²))
    
    Interpretation (Cohen, 1988):
    - f < 0.10: negligible
    - 0.10 <= f < 0.25: small
    - 0.25 <= f < 0.40: medium
    - f >= 0.40: large
    
    Args:
        eta_squared: Eta-squared value from ANOVA
    
    Returns:
        Cohen's f value
    """
    if eta_squared >= 1.0:
        return float('inf')
    if eta_squared <= 0.0:
        return 0.0
    return np.sqrt(eta_squared / (1 - eta_squared))


def interpret_cohens_f(f: float) -> str:
    """
    Interpret Cohen's f effect size (Cohen, 1988).
    
    Args:
        f: Cohen's f value
    
    Returns:
        Interpretation string
    """
    if np.isnan(f) or np.isinf(f):
        return "—"
    if f < 0.10:
        return "negligible"
    elif f < 0.25:
        return "small"
    elif f < 0.40:
        return "medium"
    else:
        return "large"


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


def compute_anova_stats(base_dir: Path) -> Dict[str, Dict[str, float]]:
    """Compute ANOVA p-value, eta-squared, and assumption tests for each experiment."""
    anova_stats = {}
    
    for exp_label, config in EXPERIMENTS.items():
        exp_name = config['name']
        exp_path = base_dir / exp_name
        is_ground_truth = config.get('is_ground_truth', False)
        
        if not exp_path.exists():
            continue
        
        f1_by_prompt = extract_f1_per_prompt(exp_path, config['judge_type'], is_ground_truth)
        
        groups = [np.array(vals) for vals in f1_by_prompt.values() if len(vals) > 0]
        groups_dict = {name: np.array(vals) for name, vals in f1_by_prompt.items() if len(vals) > 0}
        
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            eta_sq = compute_eta_squared(groups)
            cohens_f = compute_cohens_f(eta_sq)
            
            # Test assumptions
            assumption_results = test_anova_assumptions(groups_dict)
            
            anova_stats[exp_label] = {
                'p_value': p_value,
                'eta_squared': eta_sq,
                'cohens_f': cohens_f,
                'cohens_f_interp': interpret_cohens_f(cohens_f),
                'f_statistic': f_stat,
                'n_total': sum(len(g) for g in groups),
                'k_groups': len(groups),
                'assumptions': assumption_results
            }
    
    return anova_stats


def compute_friedman_stats(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Compute blocked Friedman prompt-effect stats (and post-hoc Wilcoxon) for each RQ1a experiment.

    Omnibus family-wise correction is applied across the 4 experiments via Bonferroni.
    """
    out: Dict[str, Dict[str, Any]] = {}
    raw_p: Dict[str, float] = {}

    # Compute raw p-values first
    for exp_label, config in EXPERIMENTS.items():
        exp_path = base_dir / config["name"]
        is_ground_truth = bool(config.get("is_ground_truth", False))
        prompts = list(config["prompts"].keys())

        if not exp_path.exists():
            continue

        blocked = extract_f1_by_prompt_by_block(
            exp_path,
            judge_type=config["judge_type"],
            prompts=prompts,
            is_ground_truth=is_ground_truth,
        )
        mat, blocks = build_block_matrix(blocked, prompts)
        fried = friedman_test_from_block_matrix(mat)

        out[exp_label] = {
            "prompts": prompts,
            "blocks": blocks,
            "friedman": fried,
            "posthoc_wilcoxon": posthoc_wilcoxon_pairs_from_block_matrix(mat, prompts, alpha=0.05),
        }
        raw_p[exp_label] = fried.get("p_value", np.nan)

    # Bonferroni across the 4 omnibus tests (RQ1a family)
    bonf = bonferroni_correction({k: v for k, v in raw_p.items() if not np.isnan(v)}, alpha=0.05)
    for exp, d in out.items():
        if bonf and exp in bonf.get("tests", {}):
            d["friedman_p_adj"] = bonf["tests"][exp]["p_adjusted"]
            d["friedman_sig_adj"] = bonf["tests"][exp]["significant_adjusted"]
        else:
            d["friedman_p_adj"] = np.nan
            d["friedman_sig_adj"] = False
        d["bonferroni_family"] = bonf

    return out


def create_visualization(data: pd.DataFrame, friedman_stats: Dict[str, Dict[str, Any]], output_dir: Path, experiments_config: Dict[str, dict]):
    """Create a grouped bar chart showing F1 by prompt for each experiment (Friedman + Kendall's W)."""
    
    # RQ1a prompt-effect family size (4 omnibus tests)
    n_tests = 4
    
    # Set up the figure with 2x2 subplots (one per experiment)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    experiments = ['Corr. Citation', 'Corr. Correctness', 'GT Citation', 'GT Correctness']
    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        exp_data = data[data['Experiment'] == exp]
        
        if exp_data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp, fontweight='bold')
            continue
        
        # Get data for each prompt
        prompts = list(experiments_config.get(exp, {}).get('prompts', {}).keys())
        x_pos = np.arange(len(prompts))
        bar_width = 0.6
        
        f1_means = []
        f1_ci_errs = []  # CI halfwidth for error bars (inferential, per uncertainty rule)
        colors = []
        
        for prompt in prompts:
            prompt_data = exp_data[exp_data['Prompt'] == prompt]
            if len(prompt_data) > 0:
                mean = prompt_data['F1'].values[0]
                f1_means.append(mean)
                # Calculate CI halfwidth from stored CI bounds
                ci_low = prompt_data['F1 CI Low'].values[0]
                ci_high = prompt_data['F1 CI High'].values[0]
                ci_err = (ci_high - ci_low) / 2
                f1_ci_errs.append(ci_err)
                colors.append(PROMPT_COLORS.get(prompt, '#cccccc'))
            else:
                f1_means.append(0)
                f1_ci_errs.append(0)
                colors.append('#cccccc')
        
        # Create bars with 95% CI error bars
        bars = ax.bar(x_pos, f1_means, bar_width, yerr=f1_ci_errs, 
                      color=colors, edgecolor='black', linewidth=1.5,
                      capsize=5, error_kw={'linewidth': 1.5})
        
        # Add value labels on bars
        for bar, mean, ci_err in zip(bars, f1_means, f1_ci_errs):
            height = bar.get_height()
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height + ci_err + 0.02),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add prompt-effect annotation with Bonferroni-adjusted p across the 4 RQ1a omnibus tests
        exp_stats = friedman_stats.get(exp, {})
        fried = exp_stats.get("friedman", {}) if exp_stats else {}
        p_raw = fried.get("p_value", np.nan)
        w = fried.get("kendalls_w", np.nan)
        p_adj = exp_stats.get("friedman_p_adj", np.nan)
        if not np.isnan(p_raw):
            p_for_star = p_adj if not np.isnan(p_adj) else p_raw
            sig_str = '***' if p_for_star < 0.001 else '**' if p_for_star < 0.01 else '*' if p_for_star < 0.05 else 'ns'
            w_str = f", W={w:.2f}" if not np.isnan(w) else ", W=—"
            if not np.isnan(p_adj):
                txt = f"Friedman p_adj={p_adj:.3f} ({sig_str}){w_str}"
            else:
                txt = f"Friedman p={p_raw:.3f} ({sig_str}){w_str}"
            ax.text(0.98, 0.95, txt,
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(prompts, fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title(exp, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(f1_means) * 1.4 if max(f1_means) > 0 else 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add baseline reference line
        if f1_means[0] > 0:
            ax.axhline(y=f1_means[0], color='#3498db', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add legend
    legend_prompts = set(data['Prompt'].unique())
    legend_elements = [mpatches.Patch(facecolor=PROMPT_COLORS[p], edgecolor='black', label=p)
                       for p in legend_prompts if p in PROMPT_COLORS]
    fig.legend(handles=legend_elements, loc='upper center', ncol=max(3, len(legend_elements)), 
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle(f'Prompt Sensitivity: F1 Performance by Prompt Type\n(Friedman test with CLD×run blocking; Bonferroni n={n_tests} across omnibus tests: * p_adj<.05, ** p_adj<.01, *** p_adj<.001, ns = not significant)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prompt_sensitivity_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'prompt_sensitivity_figure.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Figure saved to: {output_dir / 'prompt_sensitivity_figure.png'}")


def generate_latex_table(data: pd.DataFrame, friedman_stats: Dict[str, Dict[str, Any]], output_dir: Path, experiments_config: Dict[str, dict]):
    """Generate a LaTeX table for the thesis (Friedman p and Kendall's W)."""
    
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Prompt Sensitivity Analysis: Performance by Prompt Type Across Experiments}")
    lines.append("\\label{tab:prompt_sensitivity_app}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Experiment} & \\textbf{Prompt} & \\textbf{Distance} & \\textbf{F1} & \\textbf{Std} & \\textbf{95\\% CI} & \\textbf{$\\Delta$F1} & \\textbf{Friedman $p$} & \\textbf{W} \\\\")
    lines.append("\\midrule")
    
    experiments = ['Corr. Citation', 'Corr. Correctness', 'GT Citation', 'GT Correctness']
    exp_latex_names = {
        'Corr. Citation': 'Corruption Citation',
        'Corr. Correctness': 'Corruption Correctness', 
        'GT Citation': 'Ground Truth Citation',
        'GT Correctness': 'Ground Truth Correctness'
    }
    
    for exp in experiments:
        exp_data = data[data['Experiment'] == exp]
        exp_stats = friedman_stats.get(exp, {})
        fried = exp_stats.get("friedman", {}) if exp_stats else {}
        p_val = fried.get("p_value", np.nan)
        w_val = fried.get("kendalls_w", np.nan)
        
        first_row = True
        prompts = list(experiments_config.get(exp, {}).get('prompts', {}).keys())
        for prompt in prompts:
            row = exp_data[exp_data['Prompt'] == prompt]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            
            exp_name = exp_latex_names[exp] if first_row else ''
            dist = f"{row['Distance']:.3f}"
            f1 = f"{row['F1']:.3f}"
            f1_std = f"{row['F1 Std']:.3f}"
            # Format 95% CI
            ci_low = row.get('F1 CI Low', row['F1'])
            ci_high = row.get('F1 CI High', row['F1'])
            ci_cell = f"[{ci_low:.3f}, {ci_high:.3f}]"
            delta_f1 = f"{row['ΔF1']:+.3f}"
            
            if first_row:
                # Raw p-value with significance stars
                p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "—"
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                p_cell = f"{p_str}{sig}"
                w_cell = f"{w_val:.2f}" if not np.isnan(w_val) else "—"
            else:
                p_cell = ""
                w_cell = ""
            
            lines.append(f"{exp_name} & {prompt} & {dist} & {f1} & {f1_std} & {ci_cell} & {delta_f1} & {p_cell} & {w_cell} \\\\")
            first_row = False
        
        lines.append("\\midrule")
    
    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    
    lines.append("\\end{tabular}")
    lines.append("}%")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Note.} Statistical test: Friedman test with (CLD, run) blocking ($n=9$ blocks) compares prompt variants on matched CLD$\\times$run instances. Distance = cosine distance from baseline prompt (MPNet if available; otherwise TF-IDF fallback). Std = standard deviation across blocks. $\\Delta$F1 = change from baseline.")
    lines.append("\\item 95\\% CI computed using t-distribution ($df=8$) for block-level meta-inference per uncertainty reporting rule.")
    lines.append("\\item W = Kendall's W effect size: $<$0.1 negligible, 0.1--0.3 small, 0.3--0.5 medium, $\\geq$0.5 large.")
    lines.append("\\item Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$; unmarked = not significant.")
    lines.append("\\item \\textit{Post-hoc tests:} Pairwise Wilcoxon signed-rank tests with Bonferroni correction (per experiment) identify specific prompt differences; see figure annotations.")
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
    
    base_dir = Path(__file__).parent.parent.parent  # /home/nitai/code/causalix.ai/final_runs
    prompts_dir = Path(__file__).parent.parent.parent.parent / "data_science" / "parameter_tuning_experiments" / "alternative_prompts"
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
    
    # Compute blocked prompt-effect tests (Friedman) + Kendall's W + post-hoc Wilcoxon
    print("Computing prompt-effect tests (Friedman with CLD×run blocking) and Kendall's W...")
    friedman_stats = compute_friedman_stats(base_dir)
    for exp, stats_dict in friedman_stats.items():
        fried = stats_dict.get("friedman", {})
        p = fried.get("p_value", np.nan)
        w = fried.get("kendalls_w", np.nan)
        p_adj = stats_dict.get("friedman_p_adj", np.nan)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        sig_adj = ' (sig after Bonferroni)' if (not np.isnan(p_adj) and p_adj < 0.05) else ''
        print(f"  {exp}: p = {p:.4f}{sig}, p_adj = {p_adj:.4f}{sig_adj}, W = {w:.3f}")
    print()
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(data, friedman_stats, output_dir, EXPERIMENTS)
    
    # Generate LaTeX table
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(data, friedman_stats, output_dir, EXPERIMENTS)
    
    # Save data to Excel
    data.to_excel(output_dir / 'sensitivity_data.xlsx', index=False)
    print(f"✓ Data saved to: {output_dir / 'sensitivity_data.xlsx'}")
    
    # Save comprehensive statistics JSON (per uncertainty reporting rule)
    import json
    comprehensive_stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }
    
    for _, row in data.iterrows():
        exp = row['Experiment']
        prompt = row['Prompt']
        
        if exp not in comprehensive_stats['experiments']:
            comprehensive_stats['experiments'][exp] = {
                'prompts': {},
                'prompt_effect': friedman_stats.get(exp, {})
            }
        
        comprehensive_stats['experiments'][exp]['prompts'][prompt] = {
            'n': int(row['N']),
            'values': row.get('F1 Values', []),
            'mean': float(row['F1']),
            'std': float(row['F1 Std']),
            'min': float(row.get('F1 Min', np.nan)),
            'max': float(row.get('F1 Max', np.nan)),
            'distance': float(row['Distance']),
            'delta_f1': float(row['ΔF1']),
            'latex_primary': f"{row['F1']:.3f} [{row.get('F1 Min', 0):.3f}, {row.get('F1 Max', 0):.3f}]"
        }
    
    # Make ANOVA stats JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return [make_serializable(v) for v in sorted(obj)]
        elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj
    
    comprehensive_stats = make_serializable(comprehensive_stats)
    
    try:
        with open(output_dir / 'sensitivity_comprehensive_stats.json', 'w') as f:
            json.dump(comprehensive_stats, f, indent=2)
        print(f"✓ Comprehensive stats saved to: {output_dir / 'sensitivity_comprehensive_stats.json'}")
    except TypeError as exc:
        print(f"WARNING: Could not save comprehensive stats due to serialization issue: {exc}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


def run_assumption_tests_only():
    """Run only the ANOVA assumption tests without requiring embeddings."""
    print("="*70)
    print("RQ1a ANOVA ASSUMPTION TESTS (Standalone)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_dir = Path(__file__).parent.parent.parent  # /home/nitai/code/causalix.ai/final_runs
    output_dir = base_dir / "Sensitivity_analysis_simple"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute ANOVA stats (this extracts F1 directly from Excel, no embeddings needed)
    print("Computing ANOVA statistics and assumption tests...")
    anova_stats = compute_anova_stats(base_dir)
    
    # Print results
    print("\nANOVA Results:")
    print("-" * 70)
    for exp, stats_dict in anova_stats.items():
        p = stats_dict['p_value']
        eta = stats_dict['eta_squared']
        f = stats_dict['cohens_f']
        f_interp = stats_dict['cohens_f_interp']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {exp}: F={stats_dict['f_statistic']:.4f}, p={p:.4f}{sig}, η²={eta:.3f}, Cohen's f={f:.3f} ({f_interp})")
    
    print("\nAssumption Tests:")
    print("-" * 70)
    
    results_summary = []
    for exp, stats_dict in anova_stats.items():
        assumptions = stats_dict.get('assumptions', {})
        normality = assumptions.get('normality', {})
        homogeneity = assumptions.get('homogeneity', {})
        
        print(f"\n{exp}:")
        print("  Normality (Shapiro-Wilk):")
        
        all_normal = True
        for group, norm_stats in normality.items():
            if norm_stats['p'] is not None and not np.isnan(norm_stats['p']):
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
            'Experiment': exp,
            'ANOVA_F': stats_dict['f_statistic'],
            'ANOVA_p': stats_dict['p_value'],
            'eta_squared': eta,
            'cohens_f': f,
            'all_normal': all_normal,
            'levene_p': homogeneity.get('p', np.nan),
            'equal_variance': homogeneity.get('equal_var', None),
            'assumptions_met': met
        })
    
    # Save to file
    import json
    output_file = output_dir / 'rq1a_assumption_tests.json'
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

