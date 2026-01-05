#!/usr/bin/env python3
"""
Simple power analysis for node ablation study using basic formulas.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Study parameters
k = 7  # number of groups (prompts)
n_per_group = 15  # sample size per group
N = k * n_per_group  # total sample size
alpha = 0.05

# Load data
data_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/node_ablation_analyses/full_study_7x3x5/node_ablation_data_exp_20251009_full_ablation_7x3x5_combined.csv"
df = pd.read_csv(data_file)

# Calculate statistics by prompt
prompt_stats = df.groupby('prompt')['hybrid_f1'].agg(['mean', 'std', 'count'])
prompt_stats = prompt_stats.sort_values('mean', ascending=False)

# Calculate baseline vs others
baseline_mean = df[df['prompt'] == 'Node_V0_Minimal']['hybrid_f1'].mean()
baseline_std = df[df['prompt'] == 'Node_V0_Minimal']['hybrid_f1'].std()

print("="*80)
print("POST-HOC POWER ANALYSIS: Node Generation Ablation Study")
print("="*80)
print(f"\nStudy Design:")
print(f"  • Number of groups (prompts): {k}")
print(f"  • Sample size per group: {n_per_group}")
print(f"  • Total N: {N}")
print(f"  • Alpha level: {alpha}")
print(f"  • Baseline prompt: Node_V0_Minimal (F1 = {baseline_mean:.4f})")

print(f"\n{'-'*80}")
print("OBSERVED PERFORMANCE BY PROMPT")
print(f"{'-'*80}")
for prompt in prompt_stats.index:
    mean = prompt_stats.loc[prompt, 'mean']
    std = prompt_stats.loc[prompt, 'std']
    diff = mean - baseline_mean
    cohens_d = diff / np.sqrt((baseline_std**2 + std**2) / 2)  # pooled std
    
    magnitude = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    
    print(f"{prompt:25s}: F1={mean:.4f} (±{std:.4f}), Δ={diff:+.4f}, d={cohens_d:+.3f} ({magnitude})")

# Performance range
best_mean = prompt_stats['mean'].max()
worst_mean = prompt_stats['mean'].min()
performance_range = best_mean - worst_mean

print(f"\n{'-'*80}")
print("PERFORMANCE RANGE")
print(f"{'-'*80}")
print(f"  • Best prompt: {prompt_stats.index[0]} (F1 = {best_mean:.4f})")
print(f"  • Worst prompt: {prompt_stats.index[-1]} (F1 = {worst_mean:.4f})")
print(f"  • Absolute difference: {performance_range:.4f}")
print(f"  • Relative difference: {(performance_range/worst_mean)*100:.1f}%")

# Calculate effect size for ANOVA (eta-squared from observed F-statistic)
# From report: F=0.807, p=0.567
F_observed = 0.807
df_between = k - 1  # 6
df_within = N - k  # 98

eta_squared = (F_observed * df_between) / (F_observed * df_between + df_within)
omega_squared = max(0, (F_observed - 1) * df_between / (F_observed * df_between + df_within + 1))

# Cohen's f from eta-squared: f = sqrt(eta² / (1 - eta²))
f_observed = np.sqrt(eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0

print(f"\n{'-'*80}")
print("EFFECT SIZE (ANOVA)")
print(f"{'-'*80}")
print(f"  • F-statistic: {F_observed:.3f} (p = 0.567)")
print(f"  • Eta-squared (η²): {eta_squared:.4f}")
print(f"  • Omega-squared (ω²): {omega_squared:.4f}")
print(f"  • Cohen's f: {f_observed:.3f}")

# Interpret effect size
if f_observed < 0.10:
    f_interp = "negligible"
elif f_observed < 0.25:
    f_interp = "small"
elif f_observed < 0.40:
    f_interp = "medium"
else:
    f_interp = "large"

print(f"  • Interpretation: {f_interp} (Cohen, 1988)")

# Power estimation using non-central F distribution
# For observed effect size
from scipy.stats import ncf, f as f_dist

# Critical F value
F_crit = f_dist.ppf(1 - alpha, df_between, df_within)

# Non-centrality parameter: λ = N * f²
ncp = N * (f_observed ** 2)

# Power = P(F > F_crit | λ)
power_observed = 1 - ncf.cdf(F_crit, df_between, df_within, ncp)

print(f"\n{'-'*80}")
print("POST-HOC POWER")
print(f"{'-'*80}")
print(f"  • Power to detect observed effect (f={f_observed:.3f}): {power_observed:.1%}")

if power_observed < 0.80:
    print(f"  • ⚠️  WARNING: Study is UNDERPOWERED (< 80%)")
else:
    print(f"  • ✓ Study has adequate power (≥ 80%)")

# Required sample size for 80% power
target_power = 0.80
# Use iterative approach to find required N
for N_test in range(N, N * 10, 10):
    ncp_test = N_test * (f_observed ** 2)
    df_within_test = N_test - k
    power_test = 1 - ncf.cdf(F_crit, df_between, df_within_test, ncp_test)
    if power_test >= target_power:
        N_required = N_test
        break
else:
    N_required = None

if N_required:
    n_per_group_required = int(np.ceil(N_required / k))
    print(f"\n{'-'*80}")
    print("REQUIRED SAMPLE SIZE FOR 80% POWER")
    print(f"{'-'*80}")
    print(f"  • Total N required: {N_required}")
    print(f"  • N per group required: {n_per_group_required}")
    print(f"  • Total experiments needed: {n_per_group_required * k}")
    print(f"  • Current coverage: {n_per_group}/{n_per_group_required} = {n_per_group/n_per_group_required:.1%}")
    if n_per_group_required > n_per_group:
        multiplier = n_per_group_required / n_per_group
        print(f"  • Would need {multiplier:.1f}× more data")

# Calculate power for "small" effect (f=0.25 by convention)
f_small = 0.25
ncp_small = N * (f_small ** 2)
power_small = 1 - ncf.cdf(F_crit, df_between, df_within, ncp_small)

print(f"\n{'-'*80}")
print("POWER FOR CONVENTIONAL EFFECT SIZES")
print(f"{'-'*80}")
print(f"  • Power to detect small effect (f=0.25): {power_small:.1%}")

# For small effect, calculate required N
for N_test in range(N, N * 20, 10):
    ncp_test = N_test * (f_small ** 2)
    df_within_test = N_test - k
    power_test = 1 - ncf.cdf(F_crit, df_between, df_within_test, ncp_test)
    if power_test >= 0.80:
        N_required_small = N_test
        break
else:
    N_required_small = None

if N_required_small:
    n_per_group_small = int(np.ceil(N_required_small / k))
    print(f"  • To achieve 80% power for small effect:")
    print(f"    - Total N: {N_required_small}")
    print(f"    - N per group: {n_per_group_small}")
    print(f"    - Would need {n_per_group_small/n_per_group:.1f}× more data")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if power_observed < 0.80:
    if f_observed < 0.15:
        conclusion = f"""
The study has LOW power ({power_observed:.1%}) BUT the observed effect is also NEGLIGIBLE (f={f_observed:.3f}).

Key observations:
  • Performance range is only {performance_range:.1%} ({(performance_range/worst_mean)*100:.1f}% relative)
  • Effect size is below "small" threshold (f < 0.25)
  • Even with {n_per_group_required if N_required else '???'}× more data, differences would remain minimal

THESIS RECOMMENDATION:
"No statistically significant differences were found between prompt variants
(ANOVA: F={F_observed:.2f}, p=0.567). While the study may be underpowered to detect
very small effects, the observed performance differences are minimal (within {(performance_range/worst_mean)*100:.1f}%),
suggesting that prompt engineering techniques have limited practical impact on 
node generation quality—a finding that contrasts with edge generation results."
"""
    else:
        conclusion = f"""
The study is UNDERPOWERED ({power_observed:.1%}) to detect the observed effects (f={f_observed:.3f}).

Approximately {n_per_group_required/n_per_group:.1f}× more data would be needed for 80% power.

However, practical significance is still limited:
  • Performance range: {(performance_range/worst_mean)*100:.1f}%
  • Effect sizes are small (d < 0.5 for most comparisons)

THESIS RECOMMENDATION:
"No statistically significant differences were found (ANOVA: F={F_observed:.2f}, p=0.567).
While sample size limits statistical power, observed differences are small in
practical terms (within {(performance_range/worst_mean)*100:.1f}%), suggesting minimal impact of
prompt engineering on node generation."
"""
else:
    conclusion = f"""
The study has ADEQUATE power ({power_observed:.1%}) to detect the observed effects.
The lack of significance reflects genuinely negligible differences.

THESIS RECOMMENDATION:
Report with confidence: "No significant differences between prompt variants"
"""

print(conclusion)

print(f"\n{'='*80}")
