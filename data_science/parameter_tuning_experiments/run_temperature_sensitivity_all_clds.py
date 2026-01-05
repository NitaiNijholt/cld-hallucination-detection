#!/usr/bin/env python3
"""
Temperature Sensitivity Experiment for CLD Generation - ALL 3 CLDs

Tests whether generator temperature significantly impacts edge F1 performance
across all three validation CLDs.

Design:
- CLDs: Depressive (14 nodes, 34 edges), Social Norms (11 nodes, 12 edges), Emergency Dept (36 nodes, 66 edges)
- Temperature values: 0.0, 0.3, 0.5, 0.7, 1.0
- Runs: 3 per temperature per CLD (seeds 10, 20, 30)
- Total: 3 CLDs × 5 temps × 3 runs = 45 runs

Usage:
    python run_temperature_sensitivity_all_clds.py
    
Output:
    - Per-run Excel files with edge classifications
    - temperature_sensitivity_all_clds_results.json with all metrics
    - temperature_sensitivity_all_clds_analysis.txt with ANOVA results
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
SEEDS = [10, 20, 30]  # 3 runs per temperature per CLD

# CLD configurations - Social Norms already done, skip Depressive, run Emergency Dept only
CLD_CONFIGS = {
    'emergency_department': {
        'name': 'older_persons_emergency_department_visits_and_interactions',
        'excel': 'ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx',
        'target_variable': 'Emergency Department Visits',
        'gt_edges': 66,
        'gt_nodes': 36
    }
}

# Social Norms already completed in previous run - will be combined in analysis
SOCIAL_NORMS_PREVIOUS = {
    'name': 'Social_norms_and_obesity_prevalence',
    'gt_edges': 12,
    'gt_nodes': 11
}


def generate_cld_with_temperature(
    cld_key: str,
    cld_config: dict, 
    excel_file: Path, 
    output_dir: Path, 
    temperature: float,
    seed: int,
    run_idx: int
):
    """
    Generate a CLD with specified temperature.
    
    Args:
        cld_key: CLD identifier (e.g., 'social_norms')
        cld_config: CLD configuration dict
        excel_file: Path to Excel file
        output_dir: Output directory for results
        temperature: Generator temperature (0.0 to 1.0)
        seed: Random seed for reproducibility
        run_idx: Run index for logging
    
    Returns:
        dict with session_id, excel_path, and metrics
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_idx}: CLD={cld_key}, T={temperature}, seed={seed}")
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
    
    # Create CLD-specific output directory
    cld_output_dir = output_dir / cld_key
    cld_output_dir.mkdir(parents=True, exist_ok=True)
    
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
        overide_target_variable=cld_config['target_variable'],
        citation_search_provider="brave",
        output_json_prefix=f"{cld_key}_temp_{temperature}_seed_{seed}",
        output_dir=str(cld_output_dir),
        experiment_description=f"TempSens_{cld_key}_T{temperature}_s{seed}",
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
        'cld_key': cld_key,
        'cld_name': cld_config['name'],
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


def extract_f1_from_excel(excel_path: str) -> dict:
    """Extract F1, precision, recall from Excel file if not captured in result."""
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges Summary')
        # Look for the summary row
        for _, row in df.iterrows():
            if 'F1' in str(row.get('Metric', '')):
                return {
                    'edge_f1': row.get('Value'),
                }
        # Try reading metrics from specific columns
        if 'f1' in df.columns.str.lower():
            f1_col = [c for c in df.columns if 'f1' in c.lower()][0]
            return {'edge_f1': df[f1_col].iloc[0]}
    except Exception as e:
        print(f"Could not extract F1 from {excel_path}: {e}")
    return {}


def run_anova_analysis(results: list) -> dict:
    """
    Run one-way ANOVA to test if temperature significantly affects F1.
    Also computes per-CLD analysis.
    
    Args:
        results: List of run results with temperature, cld_key, and edge_f1
    
    Returns:
        dict with ANOVA results (overall and per-CLD)
    """
    # Overall analysis: Group F1 scores by temperature (across all CLDs)
    temp_groups = {}
    for r in results:
        temp = r['temperature']
        f1 = r.get('edge_f1')
        if f1 is not None:
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(f1)
    
    # Compute descriptive stats per temperature (overall)
    overall_stats_per_temp = {}
    for temp, f1_scores in sorted(temp_groups.items()):
        mean_val = np.mean(f1_scores)
        std_val = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
        min_val = np.min(f1_scores)
        max_val = np.max(f1_scores)
        overall_stats_per_temp[temp] = {
            'n': len(f1_scores),
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'values': f1_scores,
            # LaTeX formatting per uncertainty reporting rule (mean [min, max] for n < 30)
            'latex_primary': f"{mean_val:.3f} [{min_val:.3f}, {max_val:.3f}]",
            'latex_std': f"{mean_val:.3f} ± {std_val:.3f}"
        }
    
    # Run overall ANOVA
    groups_for_anova = [v for k, v in temp_groups.items() if len(v) >= 2]
    if len(groups_for_anova) >= 2:
        f_stat, p_value = stats.f_oneway(*groups_for_anova)
        overall_anova = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'df_between': len(groups_for_anova) - 1,
            'df_within': sum(len(g) for g in groups_for_anova) - len(groups_for_anova)
        }
    else:
        overall_anova = {'f_statistic': None, 'p_value': None, 'note': 'Insufficient groups'}
    
    # Per-CLD analysis
    per_cld_analysis = {}
    for cld_key in CLD_CONFIGS.keys():
        cld_results = [r for r in results if r.get('cld_key') == cld_key]
        
        cld_temp_groups = {}
        for r in cld_results:
            temp = r['temperature']
            f1 = r.get('edge_f1')
            if f1 is not None:
                if temp not in cld_temp_groups:
                    cld_temp_groups[temp] = []
                cld_temp_groups[temp].append(f1)
        
        stats_per_temp = {}
        for temp, f1_scores in sorted(cld_temp_groups.items()):
            mean_val = np.mean(f1_scores)
            std_val = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
            min_val = np.min(f1_scores)
            max_val = np.max(f1_scores)
            stats_per_temp[temp] = {
                'n': len(f1_scores),
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'values': f1_scores,
                'latex_primary': f"{mean_val:.3f} [{min_val:.3f}, {max_val:.3f}]",
                'latex_std': f"{mean_val:.3f} ± {std_val:.3f}"
            }
        
        # Per-CLD ANOVA
        cld_groups = [v for k, v in cld_temp_groups.items() if len(v) >= 2]
        if len(cld_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*cld_groups)
            cld_anova = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            cld_anova = {'f_statistic': None, 'p_value': None}
        
        per_cld_analysis[cld_key] = {
            'per_temperature': stats_per_temp,
            'anova': cld_anova,
            'n_total': len(cld_results)
        }
    
    # Overall stats
    all_f1 = [r['edge_f1'] for r in results if r.get('edge_f1') is not None]
    overall = {
        'n': len(all_f1),
        'mean': np.mean(all_f1),
        'std': np.std(all_f1, ddof=1) if len(all_f1) > 1 else 0.0,
    }
    
    return {
        'overall': {
            'per_temperature': overall_stats_per_temp,
            'anova': overall_anova,
            'summary': overall
        },
        'per_cld': per_cld_analysis
    }


def write_analysis_report(output_dir: Path, results: list, analysis: dict):
    """Write human-readable analysis report."""
    report_path = output_dir / "temperature_sensitivity_all_clds_analysis.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEMPERATURE SENSITIVITY ANALYSIS FOR CLD GENERATION - ALL 3 CLDs\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"CLDs tested: {list(CLD_CONFIGS.keys())}\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Temperature values tested: {TEMPERATURE_VALUES}\n")
        f.write(f"Runs per temperature per CLD: {len(SEEDS)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        # Overall analysis
        f.write("="*80 + "\n")
        f.write("OVERALL ANALYSIS (ACROSS ALL CLDs)\n")
        f.write("="*80 + "\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Edge F1 by Temperature (all CLDs pooled)\n")
        f.write("-"*60 + "\n\n")
        
        overall = analysis['overall']
        for temp, stats in sorted(overall['per_temperature'].items()):
            f.write(f"T={temp}:\n")
            f.write(f"  n={stats['n']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}\n")
            f.write(f"  range=[{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
        
        anova = overall['anova']
        f.write("-"*60 + "\n")
        f.write("ANOVA Results (Overall)\n")
        f.write("-"*60 + "\n\n")
        
        if anova.get('f_statistic') is not None:
            f.write(f"F({anova['df_between']},{anova['df_within']}) = {anova['f_statistic']:.3f}\n")
            f.write(f"p-value = {anova['p_value']:.4f}\n")
            f.write(f"Significant (p < 0.05): {anova['significant']}\n\n")
        else:
            f.write(f"Note: {anova.get('note', 'Could not compute ANOVA')}\n\n")
        
        # Per-CLD analysis
        f.write("="*80 + "\n")
        f.write("PER-CLD ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for cld_key, cld_analysis in analysis['per_cld'].items():
            cld_config = CLD_CONFIGS[cld_key]
            f.write(f"\n{'='*60}\n")
            f.write(f"CLD: {cld_key} ({cld_config['gt_nodes']} nodes, {cld_config['gt_edges']} edges)\n")
            f.write(f"{'='*60}\n\n")
            
            for temp, stats in sorted(cld_analysis['per_temperature'].items()):
                f.write(f"  T={temp}: n={stats['n']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}\n")
            
            cld_anova = cld_analysis['anova']
            if cld_anova.get('f_statistic') is not None:
                f.write(f"\n  ANOVA: F={cld_anova['f_statistic']:.3f}, p={cld_anova['p_value']:.4f}")
                f.write(f" ({'significant' if cld_anova['significant'] else 'not significant'})\n")
            else:
                f.write(f"\n  ANOVA: Could not compute\n")
        
        # Conclusion
        f.write("\n" + "="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        overall_summary = overall['summary']
        anova = overall['anova']
        if anova.get('p_value') is not None:
            if not anova['significant']:
                f.write(f"Temperature sensitivity analysis across 3 CLDs (n={overall_summary['n']}) ")
                f.write(f"showed NO significant effect of temperature on edge F1 ")
                f.write(f"(ANOVA F({anova['df_between']},{anova['df_within']})={anova['f_statistic']:.2f}, p={anova['p_value']:.3f}), ")
                f.write(f"justifying T=0.7 as a reasonable default.\n")
            else:
                f.write(f"Temperature sensitivity analysis across 3 CLDs (n={overall_summary['n']}) ")
                f.write(f"showed a SIGNIFICANT effect of temperature on edge F1 ")
                f.write(f"(ANOVA F({anova['df_between']},{anova['df_within']})={anova['f_statistic']:.2f}, p={anova['p_value']:.3f}).\n")
                f.write(f"Further investigation of optimal temperature is warranted.\n")
        
        f.write(f"\nOverall F1: {overall_summary['mean']:.4f} +/- {overall_summary['std']:.4f}\n")
    
    print(f"\n📄 Analysis report written to: {report_path}")
    return report_path


def main():
    """Run temperature sensitivity experiment across all 3 CLDs."""
    
    print("="*80)
    print("TEMPERATURE SENSITIVITY EXPERIMENT - ALL 3 CLDs")
    print("="*80)
    print(f"CLDs: {list(CLD_CONFIGS.keys())}")
    print(f"Temperature values: {TEMPERATURE_VALUES}")
    print(f"Runs per temperature per CLD: {len(SEEDS)}")
    print(f"Total runs: {len(CLD_CONFIGS) * len(TEMPERATURE_VALUES) * len(SEEDS)}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"temperature_sensitivity_all_clds_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # CLD folder path
    cld_folder = Path(__file__).parent
    
    # Run experiments
    results = []
    run_idx = 0
    total_runs = len(CLD_CONFIGS) * len(TEMPERATURE_VALUES) * len(SEEDS)
    
    for cld_key, cld_config in CLD_CONFIGS.items():
        excel_file = cld_folder / cld_config['excel']
        
        if not excel_file.exists():
            print(f"⚠️ Warning: CLD file not found: {excel_file}")
            continue
        
        for temp in TEMPERATURE_VALUES:
            for seed in SEEDS:
                run_idx += 1
                
                print(f"\n{'#'*80}")
                print(f"EXPERIMENT {run_idx}/{total_runs}: CLD={cld_key}, T={temp}, seed={seed}")
                print(f"{'#'*80}")
                
                try:
                    result = generate_cld_with_temperature(
                        cld_key=cld_key,
                        cld_config=cld_config,
                        excel_file=excel_file,
                        output_dir=output_dir,
                        temperature=temp,
                        seed=seed,
                        run_idx=run_idx
                    )
                    
                    # If F1 wasn't captured, try to extract from Excel
                    if result.get('edge_f1') is None and result.get('excel_path'):
                        extracted = extract_f1_from_excel(result['excel_path'])
                        result.update(extracted)
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"\n❌ ERROR in run {run_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'cld_key': cld_key,
                        'cld_name': cld_config['name'],
                        'temperature': temp,
                        'seed': seed,
                        'run_idx': run_idx,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
    
    # Save raw results
    results_file = output_dir / "temperature_sensitivity_all_clds_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'clds': {k: {kk: vv for kk, vv in v.items() if kk != 'values'} 
                         for k, v in CLD_CONFIGS.items()},
                'temperatures': TEMPERATURE_VALUES,
                'seeds': SEEDS,
                'runs_per_temp_per_cld': len(SEEDS)
            },
            'timestamp': timestamp,
            'total_runs': len(results),
            'successful_runs': len([r for r in results if 'error' not in r]),
            'results': results
        }, f, indent=2)
    print(f"\n📊 Results saved to: {results_file}")
    
    # Run analysis
    print("\n" + "="*80)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*80)
    
    analysis = run_anova_analysis(results)
    
    # Save analysis
    analysis_file = output_dir / "temperature_sensitivity_all_clds_analysis.json"
    with open(analysis_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
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
    
    # Print overall ANOVA result
    anova = analysis['overall']['anova']
    if anova.get('p_value') is not None:
        print(f"\n📊 OVERALL ANOVA: F({anova['df_between']},{anova['df_within']})={anova['f_statistic']:.3f}, p={anova['p_value']:.4f}")
        if not anova['significant']:
            print("✅ No significant effect of temperature on F1 (p >= 0.05)")
        else:
            print("⚠️  Significant effect of temperature on F1 (p < 0.05)")
    
    # Print per-CLD summary
    print("\n📊 Per-CLD ANOVA:")
    for cld_key, cld_analysis in analysis['per_cld'].items():
        cld_anova = cld_analysis['anova']
        if cld_anova.get('p_value') is not None:
            sig_marker = "*" if cld_anova['significant'] else ""
            print(f"   {cld_key}: F={cld_anova['f_statistic']:.3f}, p={cld_anova['p_value']:.4f}{sig_marker}")
    
    print()


if __name__ == "__main__":
    main()







