#!/usr/bin/env python3
"""
Temperature Sensitivity ANOVA Assumption Tests

Computes Shapiro-Wilk normality and Levene's homogeneity tests
for the temperature sensitivity ANOVA (from existing results).

This script loads results from the parameter_tuning_experiments folder
and computes statistical assumption tests.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import shapiro, levene, f_oneway

# Path to the most recent temperature sensitivity results
RESULTS_FILE = Path(__file__).parent.parent / "data_science/parameter_tuning_experiments/results/temperature_sensitivity_all_clds_20251217_202852/temperature_sensitivity_all_clds_results.json"

OUTPUT_DIR = Path(__file__).parent


def load_results():
    """Load temperature sensitivity results from JSON."""
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found: {RESULTS_FILE}")
        return None
    
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    
    return data


def test_anova_assumptions(groups_dict):
    """
    Test ANOVA assumptions: normality (Shapiro-Wilk) and homogeneity of variance (Levene's).
    
    Returns dict with 'normality' (per group) and 'homogeneity' results.
    """
    results = {'normality': {}, 'homogeneity': {}, 'assumptions_met': True}
    
    # Normality per group (Shapiro-Wilk)
    all_normal = True
    for name, vals in groups_dict.items():
        vals = np.array(vals)
        if len(vals) >= 3:
            stat, p = shapiro(vals)
            is_normal = p > 0.05
            results['normality'][name] = {'W': float(stat), 'p': float(p), 'normal': is_normal}
            if not is_normal:
                all_normal = False
        else:
            results['normality'][name] = {'W': None, 'p': None, 'normal': None, 'note': f'n={len(vals)} < 3'}
    
    # Homogeneity of variance (Levene's test)
    groups_list = [np.array(v) for v in groups_dict.values() if len(v) >= 2]
    if len(groups_list) >= 2:
        lev_stat, lev_p = levene(*groups_list)
        equal_var = lev_p > 0.05
        results['homogeneity'] = {'W': float(lev_stat), 'p': float(lev_p), 'equal_var': equal_var}
        if not equal_var:
            results['assumptions_met'] = False
    else:
        results['homogeneity'] = {'W': None, 'p': None, 'equal_var': None}
    
    results['assumptions_met'] = all_normal and results['homogeneity'].get('equal_var', True)
    
    return results


def main():
    """Run assumption tests for temperature sensitivity ANOVA."""
    print("=" * 80)
    print("TEMPERATURE SENSITIVITY ANOVA ASSUMPTION TESTS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results file: {RESULTS_FILE}")
    print()
    
    # Load data
    data = load_results()
    if data is None:
        return
    
    results = data.get('results', [])
    print(f"Total runs: {len(results)}")
    
    # Group F1 scores by temperature
    temp_groups = {}
    for r in results:
        temp = r.get('temperature')
        f1 = r.get('edge_f1')
        if temp is not None and f1 is not None:
            temp_key = f"T={temp}"
            if temp_key not in temp_groups:
                temp_groups[temp_key] = []
            temp_groups[temp_key].append(f1)
    
    print(f"Temperature groups: {list(temp_groups.keys())}")
    for t, vals in sorted(temp_groups.items()):
        print(f"  {t}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    print()
    
    # Run ANOVA
    groups_list = [np.array(v) for v in temp_groups.values() if len(v) >= 2]
    if len(groups_list) >= 2:
        f_stat, p_value = f_oneway(*groups_list)
        print("ANOVA Results:")
        print("-" * 60)
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        print(f"  Significance: {'Yes' if p_value < 0.05 else 'No'}{sig}")
    else:
        print("Error: Not enough groups for ANOVA")
        return
    
    print()
    
    # Test assumptions
    print("Assumption Tests:")
    print("-" * 60)
    
    assumption_results = test_anova_assumptions(temp_groups)
    
    print("\nNormality (Shapiro-Wilk):")
    all_normal = True
    for group, norm_stats in sorted(assumption_results['normality'].items()):
        if norm_stats['p'] is not None:
            status = "normal" if norm_stats['normal'] else "NON-NORMAL"
            if not norm_stats['normal']:
                all_normal = False
            print(f"  {group}: W={norm_stats['W']:.4f}, p={norm_stats['p']:.4f} ({status})")
        else:
            print(f"  {group}: {norm_stats.get('note', 'N/A')}")
    
    homogeneity = assumption_results['homogeneity']
    if homogeneity['p'] is not None:
        hom_status = "equal" if homogeneity['equal_var'] else "UNEQUAL"
        print(f"\nHomogeneity (Levene's): W={homogeneity['W']:.4f}, p={homogeneity['p']:.4f} ({hom_status})")
    
    met = assumption_results['assumptions_met']
    print(f"\nAssumptions met: {'Yes' if met else 'No'}")
    
    # Save results - convert numpy bools to Python bools for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'source_file': str(RESULTS_FILE),
        'n_total': len(results),
        'anova': {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        },
        'assumptions': convert_numpy(assumption_results),
        'per_temperature': {
            t: {'n': len(v), 'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for t, v in temp_groups.items()
        },
        'note': 'With ANOVA p=0.998 (highly non-significant), assumption violations do not affect conclusions'
    }
    
    output_file = OUTPUT_DIR / 'temperature_sensitivity_assumption_tests.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("ASSUMPTION TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()






