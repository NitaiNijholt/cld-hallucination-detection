#!/usr/bin/env python3
"""
Temperature Sensitivity Experiment for CLD Generation

Tests whether generator temperature significantly impacts edge F1 performance.
Uses the smallest CLD (Social norms, 11 edges) for cost efficiency.

Design:
- Temperature values: 0.0, 0.3, 0.5, 0.7, 1.0
- Runs: T=0.0 gets 1 run (deterministic), others get 3 runs each
- Total: 13 runs
- Estimated cost: ~$2-3

Usage:
    python run_temperature_sensitivity.py
    
Output:
    - Per-run Excel files with edge classifications
    - temperature_sensitivity_results.json with all metrics
    - temperature_sensitivity_analysis.txt with ANOVA results
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

# Load environment variables from .env.dev
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable not set!")

print("OpenAI API key is set:", os.getenv('OPENAI_API_KEY')[:10] + "..." + os.getenv('OPENAI_API_KEY')[-4:])

# Experimental configuration
TEMPERATURE_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
RUNS_PER_TEMP = {0.0: 1, 0.3: 3, 0.5: 3, 0.7: 3, 1.0: 3}  # 13 total
SEEDS = [10, 20, 30]  # Seeds for stochastic runs

# CLD configuration - use smallest CLD for cost efficiency
CLD_CONFIG = {
    'name': 'Social_norms_and_obesity_prevalence',
    'excel': 'Social_norms_and_obesity_prevalence.xlsx',
    'target_variable': 'Obesity Prevalence'  # Required override for this CLD
}


def generate_cld_with_temperature(
    cld_name: str, 
    excel_file: str, 
    output_dir: Path, 
    temperature: float,
    seed: int = None,
    run_idx: int = 0
):
    """
    Generate a CLD with specified temperature.
    
    Args:
        cld_name: Name of the CLD
        excel_file: Path to Excel file
        output_dir: Output directory for results
        temperature: Generator temperature (0.0 to 1.0)
        seed: Random seed for reproducibility
        run_idx: Run index for logging
    
    Returns:
        dict with session_id, excel_path, and metrics
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_idx}: T={temperature}, seed={seed}")
    print(f"{'='*80}")
    print(f"Excel file: {excel_file}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Generator config with varying temperature
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": temperature,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": 5
    }
    
    # Dummy configs (not used for generation-only)
    dummy_corruptor_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    dummy_judge_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0
    }
    
    # Load the experiment YAML
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    # Run CLD generation with parallel edge discovery
    print(f"Generating CLD with T={temperature}...")
    result = run_discovery_experiment(
        excel_path=str(excel_file),
        yaml_path=str(yaml_path),
        dev_mode=False,
        generator_config=generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=dummy_judge_config,
        corruption_rate=0.0,
        judge_edges=False,
        generation_parallel=True,
        generation_max_workers=10,
        overide_target_variable=CLD_CONFIG['target_variable'],
        citation_search_provider="brave",
        output_json_prefix=f"temp_{temperature}_seed_{seed}",
        output_dir=str(output_dir),
        experiment_description=f"TempSens_T{temperature}_s{seed}",
        embedding_enable=False,  # Skip CI metrics for speed
        ci_compute_embeddings=False,
        node_comparison_enable=True,  # Enable comparison for F1 calculation
    )
    
    session_id = result.get('session_id')
    excel_path = result.get('result_excel_path')
    
    # Extract F1 metrics from result
    edge_metrics = result.get('edge_comparison', {})
    node_metrics = result.get('node_comparison', {})
    
    print(f"\n✅ RUN COMPLETE!")
    print(f"   Session ID: {session_id}")
    print(f"   Excel: {excel_path}")
    if edge_metrics:
        print(f"   Edge F1: {edge_metrics.get('f1', 'N/A')}")
        print(f"   Edge Precision: {edge_metrics.get('precision', 'N/A')}")
        print(f"   Edge Recall: {edge_metrics.get('recall', 'N/A')}")
    
    return {
        'temperature': temperature,
        'seed': seed,
        'run_idx': run_idx,
        'session_id': session_id,
        'excel_path': excel_path,
        'edge_f1': edge_metrics.get('f1'),
        'edge_precision': edge_metrics.get('precision'),
        'edge_recall': edge_metrics.get('recall'),
        'node_f1': node_metrics.get('f1'),
        'timestamp': datetime.now().isoformat()
    }


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


def run_anova_analysis(results: list) -> dict:
    """
    Run one-way ANOVA to test if temperature significantly affects F1.
    
    Args:
        results: List of run results with temperature and edge_f1
    
    Returns:
        dict with ANOVA results including eta-squared effect size
    """
    # Group F1 scores by temperature
    temp_groups = {}
    for r in results:
        temp = r['temperature']
        f1 = r.get('edge_f1')
        if f1 is not None:
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(f1)
    
    # Compute descriptive stats per temperature
    stats_per_temp = {}
    for temp, f1_scores in sorted(temp_groups.items()):
        stats_per_temp[temp] = {
            'n': len(f1_scores),
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0,
            'min': np.min(f1_scores),
            'max': np.max(f1_scores),
            'values': f1_scores
        }
    
    # Run ANOVA (only for temperatures with multiple samples)
    groups_for_anova = [np.array(v) for k, v in temp_groups.items() if len(v) > 1]
    
    if len(groups_for_anova) >= 2:
        f_stat, p_value = stats.f_oneway(*groups_for_anova)
        eta_sq = compute_eta_squared(groups_for_anova)
        
        # Compute degrees of freedom
        k = len(groups_for_anova)  # number of groups
        n = sum(len(g) for g in groups_for_anova)  # total observations
        df_between = k - 1
        df_within = n - k
        
        anova_result = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_sq,
            'df_between': df_between,
            'df_within': df_within,
            'significant': p_value < 0.05
        }
    else:
        anova_result = {
            'f_statistic': None,
            'p_value': None,
            'eta_squared': None,
            'df_between': None,
            'df_within': None,
            'significant': None,
            'note': 'Insufficient groups for ANOVA'
        }
    
    # Overall stats
    all_f1 = [r['edge_f1'] for r in results if r.get('edge_f1') is not None]
    overall = {
        'n': len(all_f1),
        'mean': np.mean(all_f1),
        'std': np.std(all_f1, ddof=1) if len(all_f1) > 1 else 0.0,
    }
    
    return {
        'per_temperature': stats_per_temp,
        'anova': anova_result,
        'overall': overall
    }


def write_analysis_report(output_dir: Path, results: list, analysis: dict):
    """Write human-readable analysis report."""
    report_path = output_dir / "temperature_sensitivity_analysis.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEMPERATURE SENSITIVITY ANALYSIS FOR CLD GENERATION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"CLD: {CLD_CONFIG['name']}\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Temperature values tested: {TEMPERATURE_VALUES}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("EDGE F1 BY TEMPERATURE\n")
        f.write("-"*80 + "\n\n")
        
        for temp, stats in sorted(analysis['per_temperature'].items()):
            f.write(f"T={temp}:\n")
            f.write(f"  n={stats['n']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}\n")
            f.write(f"  range=[{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"  values={[f'{v:.4f}' for v in stats['values']]}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("ANOVA RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        anova = analysis['anova']
        if anova['f_statistic'] is not None:
            f.write(f"F-statistic: F({anova['df_between']},{anova['df_within']}) = {anova['f_statistic']:.4f}\n")
            f.write(f"p-value: {anova['p_value']:.4f}\n")
            f.write(f"Eta-squared (η²): {anova['eta_squared']:.4f}\n")
            
            # Effect size interpretation
            eta_sq = anova['eta_squared']
            if eta_sq < 0.01:
                effect_interp = "negligible"
            elif eta_sq < 0.06:
                effect_interp = "small"
            elif eta_sq < 0.14:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            f.write(f"Effect size interpretation: {effect_interp}\n")
            f.write(f"Significant (p < 0.05): {anova['significant']}\n\n")
        else:
            f.write(f"Note: {anova.get('note', 'Could not compute ANOVA')}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("-"*80 + "\n\n")
        
        overall = analysis['overall']
        if anova['p_value'] is not None:
            eta_sq = anova['eta_squared']
            if not anova['significant']:
                f.write(f"Temperature sensitivity analysis (n={overall['n']}, ")
                f.write(f"T in {{{', '.join(map(str, TEMPERATURE_VALUES))}}}) ")
                f.write(f"showed NO significant effect on edge F1 ")
                f.write(f"(ANOVA F({anova['df_between']},{anova['df_within']})={anova['f_statistic']:.2f}, ")
                f.write(f"p={anova['p_value']:.3f}, η²={eta_sq:.2f}), ")
                f.write(f"justifying T=0.7 as a reasonable default.\n")
            else:
                f.write(f"Temperature sensitivity analysis (n={overall['n']}, ")
                f.write(f"T in {{{', '.join(map(str, TEMPERATURE_VALUES))}}}) ")
                f.write(f"showed a SIGNIFICANT effect on edge F1 ")
                f.write(f"(ANOVA F({anova['df_between']},{anova['df_within']})={anova['f_statistic']:.2f}, ")
                f.write(f"p={anova['p_value']:.3f}, η²={eta_sq:.2f}).\n")
                f.write(f"Further investigation of optimal temperature is warranted.\n")
        else:
            f.write("Could not determine statistical significance.\n")
        
        f.write(f"\nOverall F1: {overall['mean']:.4f} +/- {overall['std']:.4f}\n")
    
    print(f"\n📄 Analysis report written to: {report_path}")
    return report_path


def main():
    """Run temperature sensitivity experiment."""
    
    print("="*80)
    print("TEMPERATURE SENSITIVITY EXPERIMENT")
    print("="*80)
    print(f"CLD: {CLD_CONFIG['name']}")
    print(f"Temperature values: {TEMPERATURE_VALUES}")
    print(f"Runs per temperature: {RUNS_PER_TEMP}")
    print(f"Total runs: {sum(RUNS_PER_TEMP.values())}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"temperature_sensitivity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # CLD file path
    cld_folder = PROJECT_ROOT / "data_science" / "parameter_tuning_experiments" / "ground_truth_clds_for_experiments"
    excel_file = cld_folder / CLD_CONFIG['excel']
    
    if not excel_file.exists():
        raise FileNotFoundError(f"CLD file not found: {excel_file}")
    
    # Run experiments
    results = []
    run_idx = 0
    
    for temp in TEMPERATURE_VALUES:
        n_runs = RUNS_PER_TEMP[temp]
        
        for run in range(n_runs):
            run_idx += 1
            
            # For T=0.0 (deterministic), seed doesn't matter
            # For stochastic temps, use different seeds
            seed = None if temp == 0.0 else SEEDS[run]
            
            print(f"\n{'#'*80}")
            print(f"EXPERIMENT {run_idx}/{sum(RUNS_PER_TEMP.values())}: T={temp}, run={run+1}/{n_runs}")
            print(f"{'#'*80}")
            
            try:
                result = generate_cld_with_temperature(
                    cld_name=CLD_CONFIG['name'],
                    excel_file=excel_file,
                    output_dir=output_dir,
                    temperature=temp,
                    seed=seed,
                    run_idx=run_idx
                )
                results.append(result)
            except Exception as e:
                print(f"\n❌ ERROR in run {run_idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'temperature': temp,
                    'seed': seed,
                    'run_idx': run_idx,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    # Save raw results
    results_file = output_dir / "temperature_sensitivity_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'cld': CLD_CONFIG,
                'temperatures': TEMPERATURE_VALUES,
                'runs_per_temp': RUNS_PER_TEMP,
                'seeds': SEEDS
            },
            'timestamp': timestamp,
            'total_runs': len(results),
            'results': results
        }, f, indent=2)
    print(f"\n📊 Results saved to: {results_file}")
    
    # Run analysis
    print("\n" + "="*80)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*80)
    
    analysis = run_anova_analysis(results)
    
    # Save analysis
    analysis_file = output_dir / "temperature_sensitivity_analysis.json"
    with open(analysis_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(analysis), f, indent=2)
    print(f"📈 Analysis saved to: {analysis_file}")
    
    # Write human-readable report
    report_path = write_analysis_report(output_dir, results, analysis)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nTotal runs: {len(results)}")
    print(f"Successful runs: {len([r for r in results if 'error' not in r])}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - {results_file.name}")
    print(f"  - {analysis_file.name}")
    print(f"  - {report_path.name}")
    
    # Print ANOVA result
    anova = analysis['anova']
    if anova['p_value'] is not None:
        print(f"\nANOVA: F({anova['df_between']},{anova['df_within']})={anova['f_statistic']:.3f}, p={anova['p_value']:.4f}, η²={anova['eta_squared']:.3f}")
        if not anova['significant']:
            print("✅ No significant effect of temperature on F1 (p >= 0.05)")
        else:
            print("⚠️  Significant effect of temperature on F1 (p < 0.05)")
    
    print()


if __name__ == "__main__":
    main()







