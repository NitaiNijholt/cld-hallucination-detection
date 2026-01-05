#!/usr/bin/env python3
"""
Post-hoc power analysis for node ablation study.

Analyzes whether the lack of statistical significance is due to:
1. Insufficient sample size (Type II error)
2. Genuinely small/negligible effects
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower


def calculate_post_hoc_power():
    """Calculate post-hoc statistical power for the ANOVA."""
    
    # Study parameters
    k = 7  # number of groups (prompts)
    n_per_group = 15  # sample size per group
    alpha = 0.05
    
    # Observed effect sizes (from results)
    effect_sizes = {
        'V6_Andrew': 0.437,    # small
        'V3_CoT_System': 0.164,  # negligible
        'V5_Nitai_C': -0.040,    # negligible
        'V1_Role_Only': -0.127,  # negligible
        'V4_CoT_User': -0.203,   # small
        'V2_Constraints': -0.262  # small
    }
    
    # Calculate mean absolute effect size
    mean_effect = np.mean([abs(d) for d in effect_sizes.values()])
    max_effect = max([abs(d) for d in effect_sizes.values()])
    
    print("="*80)
    print("POST-HOC POWER ANALYSIS: Node Generation Ablation Study")
    print("="*80)
    print(f"\nStudy Design:")
    print(f"  • Number of groups (prompts): {k}")
    print(f"  • Sample size per group: {n_per_group}")
    print(f"  • Total N: {k * n_per_group}")
    print(f"  • Alpha level: {alpha}")
    
    print(f"\nObserved Effect Sizes (Cohen's d vs. baseline):")
    for prompt, d in effect_sizes.items():
        magnitude = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium"
        print(f"  • {prompt:20s}: d = {d:+.3f} ({magnitude})")
    
    print(f"\nEffect Size Summary:")
    print(f"  • Mean |d|: {mean_effect:.3f}")
    print(f"  • Max |d|: {max_effect:.3f}")
    
    # Convert Cohen's d to Cohen's f (for ANOVA)
    # f = d / 2 (approximate for pairwise comparisons)
    # f = sqrt(sum((μᵢ - μ_grand)² / k) / σ²)
    
    # For conservative estimate, use mean effect size
    f_mean = mean_effect / 2
    f_max = max_effect / 2
    
    print(f"\nEffect Size (Cohen's f for ANOVA):")
    print(f"  • f (mean): {f_mean:.3f}")
    print(f"  • f (max): {f_max:.3f}")
    
    # Calculate post-hoc power
    power_calc = FTestAnovaPower()
    
    # Power for mean effect
    power_mean = power_calc.solve_power(
        effect_size=f_mean,
        nobs=n_per_group * k,
        alpha=alpha,
        k_groups=k
    )
    
    # Power for max effect
    power_max = power_calc.solve_power(
        effect_size=f_max,
        nobs=n_per_group * k,
        alpha=alpha,
        k_groups=k
    )
    
    print(f"\n{'='*80}")
    print("POST-HOC POWER")
    print(f"{'='*80}")
    print(f"  • Power to detect mean effect (f={f_mean:.3f}): {power_mean:.1%}")
    print(f"  • Power to detect max effect (f={f_max:.3f}): {power_max:.1%}")
    
    if power_mean < 0.80:
        print(f"\n⚠️  WARNING: Power is below 0.80 threshold!")
        print(f"   The study is UNDERPOWERED to detect the observed effects.")
    else:
        print(f"\n✓ Power is adequate (≥0.80)")
    
    # Calculate required sample size for 80% power
    print(f"\n{'='*80}")
    print("REQUIRED SAMPLE SIZE FOR 80% POWER")
    print(f"{'='*80}")
    
    for effect_label, f in [("mean effect", f_mean), ("max effect", f_max)]:
        n_required = power_calc.solve_power(
            effect_size=f,
            alpha=alpha,
            power=0.80,
            k_groups=k
        )
        n_per_group_required = int(np.ceil(n_required / k))
        total_experiments = n_per_group_required * k
        
        print(f"\nFor {effect_label} (f={f:.3f}):")
        print(f"  • Total N required: {int(np.ceil(n_required))}")
        print(f"  • N per group required: {n_per_group_required}")
        print(f"  • Total experiments needed: {total_experiments}")
        print(f"  • Current coverage: {n_per_group}/{n_per_group_required} = {n_per_group/n_per_group_required:.1%}")
        
        if n_per_group_required > n_per_group:
            multiplier = n_per_group_required / n_per_group
            print(f"  • Would need {multiplier:.1f}× more data")
    
    # Practical significance analysis
    print(f"\n{'='*80}")
    print("PRACTICAL SIGNIFICANCE")
    print(f"{'='*80}")
    
    # Load actual data to get performance values
    data_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/node_ablation_analyses/full_study_7x3x5/node_ablation_data_exp_20251009_full_ablation_7x3x5_combined.csv"
    df = pd.read_csv(data_file)
    
    # Calculate performance by prompt
    perf_by_prompt = df.groupby('prompt')['hybrid_f1'].agg(['mean', 'std'])
    perf_by_prompt = perf_by_prompt.sort_values('mean', ascending=False)
    
    best_prompt = perf_by_prompt.index[0]
    worst_prompt = perf_by_prompt.index[-1]
    best_mean = perf_by_prompt.loc[best_prompt, 'mean']
    worst_mean = perf_by_prompt.loc[worst_prompt, 'mean']
    performance_range = best_mean - worst_mean
    
    print(f"\nPerformance Range:")
    print(f"  • Best prompt: {best_prompt} (F1 = {best_mean:.4f})")
    print(f"  • Worst prompt: {worst_prompt} (F1 = {worst_mean:.4f})")
    print(f"  • Absolute difference: {performance_range:.4f} ({performance_range/worst_mean*100:.1f}% relative)")
    
    # Cohen (1988) interpretation:
    # d < 0.2: negligible
    # d = 0.2: small
    # d = 0.5: medium
    # d = 0.8: large
    
    print(f"\nEffect Size Interpretation (Cohen, 1988):")
    print(f"  • Max effect size (d={max_effect:.3f}) is {'small' if max_effect < 0.5 else 'medium'}")
    print(f"  • Most effects (5/6) are negligible or small (|d| < 0.3)")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    if power_mean < 0.80 and max_effect >= 0.3:
        conclusion = """
The study is UNDERPOWERED to reliably detect the observed small effects.
However, the effects themselves are also SMALL in practical terms:
  • Performance range is only ~6%
  • Most effect sizes are negligible (|d| < 0.2)
  • Only 1 prompt shows a small-to-medium effect (d=0.44)

RECOMMENDATION:
1. For thesis: Report as "no significant differences" but note power limitation
2. Acknowledge that larger N MIGHT reveal statistical significance
3. Emphasize that even if significant, practical differences are small
4. Consider stating: "Prompt engineering has minimal impact on node generation
   (unlike edge generation), with all variants performing within 6% of each other"
"""
    elif power_mean >= 0.80:
        conclusion = """
The study has ADEQUATE power to detect the observed effects.
The lack of significance reflects genuinely small/negligible effects.

RECOMMENDATION:
Report with confidence: "No significant differences between prompts"
"""
    else:
        conclusion = """
The study is underpowered AND the observed effects are negligible.

RECOMMENDATION:
Interpret as: "Prompt engineering has minimal impact on node generation"
"""
    
    print(conclusion)
    
    return {
        'power_mean': power_mean,
        'power_max': power_max,
        'n_required_mean': int(np.ceil(power_calc.solve_power(f_mean, alpha, 0.80, k_groups=k))),
        'n_required_max': int(np.ceil(power_calc.solve_power(f_max, alpha, 0.80, k_groups=k))),
        'effect_size_mean': f_mean,
        'effect_size_max': f_max,
        'performance_range': performance_range
    }


if __name__ == "__main__":
    results = calculate_post_hoc_power()
