#!/usr/bin/env python3
"""
Plot cost-efficiency tradeoff: Full DR vs Smart-Trigger DR deployment.

Compares:
- Full DR: Deploy DR on all 285 edges
- Smart-trigger DR: Deploy DR only on edges with low judge confidence (≤0.5)

Shows ΔF1 per $ spent to determine if smart-trigger is more cost-efficient.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).resolve().parent.parent))).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR = OUTPUT_ROOT / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMART_TRIGGER_RESULTS = INPUT_DIR / "validation/smart_trigger_enrichment_results.json"

# Cost parameters
COST_PER_EDGE_DR = 0.87  # $ per edge for Deep Research
COST_PER_EDGE_JUDGE = 0.0015  # $ per edge for GT Correctness judge (0.15¢)

# Full DR dataset counts (from RQ3 analysis)
FULL_DR_EDGES = 285
FULL_DR_FP_WITH_EVIDENCE = 111  # FP edges with direct_causal_found=True
FULL_DR_FN_WITH_EVIDENCE = 27   # FN edges with direct_causal_found=True

# Original confusion matrix
ORIG_TP = 44
ORIG_FP = 178
ORIG_FN = 63


def compute_f1(tp: float, fp: float, fn: float) -> float:
    """Compute F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def main() -> None:
    # Load smart-trigger results
    with open(SMART_TRIGGER_RESULTS) as f:
        st_results = json.load(f)
    
    # Extract smart-trigger counts
    st_fp_triggered_dr = st_results["trigger_counts"]["n_fp_triggered_dr_found"]
    st_fn_triggered_dr = st_results["trigger_counts"]["n_fn_triggered_dr_found"]
    
    # Extract calibration rates
    p_fp = st_results["rates"]["FP"]["p"]
    p_fn = st_results["rates"]["FN"]["p"]
    p_fp_low = st_results["rates"]["FP"]["p_low"]
    p_fn_low = st_results["rates"]["FN"]["p_low"]
    p_fp_high = st_results["rates"]["FP"]["p_high"]
    p_fn_high = st_results["rates"]["FN"]["p_high"]
    
    # Number of triggered edges (from script output: 142)
    n_triggered = 142
    
    # ---------------------------------------------------------------------------
    # Compute scenarios
    # ---------------------------------------------------------------------------
    
    # Original (no DR, no judge)
    orig_f1 = compute_f1(ORIG_TP, ORIG_FP, ORIG_FN)
    orig_cost = 0
    
    # Smart-trigger DR requires:
    # 1. Running GT Correctness judge on ALL edges first (to determine trigger)
    # 2. Running DR only on triggered edges
    st_judge_cost = FULL_DR_EDGES * COST_PER_EDGE_JUDGE  # Judge on all 285 edges
    st_dr_cost = n_triggered * COST_PER_EDGE_DR  # DR only on triggered
    st_cost = st_judge_cost + st_dr_cost
    
    st_fp_promote = st_fp_triggered_dr * p_fp
    st_fn_discover = st_fn_triggered_dr * p_fn
    st_f1_enrich = compute_f1(ORIG_TP + st_fp_promote, ORIG_FP - st_fp_promote, ORIG_FN)
    st_f1_system = compute_f1(ORIG_TP + st_fp_promote + st_fn_discover, 
                               ORIG_FP - st_fp_promote, 
                               ORIG_FN - st_fn_discover)
    
    # Full DR (deploy on all 285 edges, no judge needed)
    full_fp_promote = FULL_DR_FP_WITH_EVIDENCE * p_fp
    full_fn_discover = FULL_DR_FN_WITH_EVIDENCE * p_fn
    full_f1_enrich = compute_f1(ORIG_TP + full_fp_promote, ORIG_FP - full_fp_promote, ORIG_FN)
    full_f1_system = compute_f1(ORIG_TP + full_fp_promote + full_fn_discover,
                                 ORIG_FP - full_fp_promote,
                                 ORIG_FN - full_fn_discover)
    full_cost = FULL_DR_EDGES * COST_PER_EDGE_DR  # DR on all edges
    
    # Conservative estimates (using lower CI bounds)
    st_fp_promote_cons = st_fp_triggered_dr * p_fp_low
    st_fn_discover_cons = st_fn_triggered_dr * p_fn_low
    st_f1_system_cons = compute_f1(ORIG_TP + st_fp_promote_cons + st_fn_discover_cons,
                                    ORIG_FP - st_fp_promote_cons,
                                    ORIG_FN - st_fn_discover_cons)
    
    full_fp_promote_cons = FULL_DR_FP_WITH_EVIDENCE * p_fp_low
    full_fn_discover_cons = FULL_DR_FN_WITH_EVIDENCE * p_fn_low
    full_f1_system_cons = compute_f1(ORIG_TP + full_fp_promote_cons + full_fn_discover_cons,
                                      ORIG_FP - full_fp_promote_cons,
                                      ORIG_FN - full_fn_discover_cons)

    # Upper estimates (using upper CI bounds) for two-sided uncertainty visualization.
    st_fp_promote_high = st_fp_triggered_dr * p_fp_high
    st_fn_discover_high = st_fn_triggered_dr * p_fn_high
    st_f1_system_high = compute_f1(
        ORIG_TP + st_fp_promote_high + st_fn_discover_high,
        ORIG_FP - st_fp_promote_high,
        ORIG_FN - st_fn_discover_high,
    )

    full_fp_promote_high = FULL_DR_FP_WITH_EVIDENCE * p_fp_high
    full_fn_discover_high = FULL_DR_FN_WITH_EVIDENCE * p_fn_high
    full_f1_system_high = compute_f1(
        ORIG_TP + full_fp_promote_high + full_fn_discover_high,
        ORIG_FP - full_fp_promote_high,
        ORIG_FN - full_fn_discover_high,
    )
    
    # ---------------------------------------------------------------------------
    # Compute efficiency metrics
    # ---------------------------------------------------------------------------
    
    # ΔF1 / $ (using LLM+DR system scenario)
    st_delta_f1 = st_f1_system - orig_f1
    full_delta_f1 = full_f1_system - orig_f1
    
    st_efficiency = st_delta_f1 / st_cost * 100  # ΔF1 per $100
    full_efficiency = full_delta_f1 / full_cost * 100
    
    # Marginal efficiency of full vs smart-trigger
    marginal_cost = full_cost - st_cost
    marginal_delta_f1 = full_delta_f1 - st_delta_f1
    marginal_efficiency = marginal_delta_f1 / marginal_cost * 100 if marginal_cost > 0 else 0
    
    print("=" * 70)
    print("DR Deployment Cost-Efficiency Analysis")
    print("=" * 70)
    print(f"\nCost per edge:")
    print(f"  - DR: ${COST_PER_EDGE_DR:.2f}/edge")
    print(f"  - GT Correctness Judge: ${COST_PER_EDGE_JUDGE:.4f}/edge (0.15¢)")
    print(f"\nOriginal F1: {orig_f1:.3f}")
    print()
    print(f"{'Metric':<35} {'Smart-Trigger':<20} {'Full DR':<20}")
    print("-" * 75)
    print(f"{'Edges judged':<35} {FULL_DR_EDGES:<20} {'N/A':<20}")
    print(f"{'Edges sent to DR':<35} {n_triggered:<20} {FULL_DR_EDGES:<20}")
    print(f"{'Judge cost':<35} ${st_judge_cost:.2f}{'':<14} $0.00")
    print(f"{'DR cost':<35} ${st_dr_cost:.2f}{'':<12} ${full_cost:.2f}")
    print(f"{'Total cost':<35} ${st_cost:.2f}{'':<12} ${full_cost:.2f}")
    print(f"{'FP with evidence':<35} {st_fp_triggered_dr:<20} {FULL_DR_FP_WITH_EVIDENCE:<20}")
    print(f"{'FN with evidence':<35} {st_fn_triggered_dr:<20} {FULL_DR_FN_WITH_EVIDENCE:<20}")
    print(f"{'Expected FP promotions':<35} {st_fp_promote:.1f}{'':<14} {full_fp_promote:.1f}")
    print(f"{'Expected FN discoveries':<35} {st_fn_discover:.1f}{'':<14} {full_fn_discover:.1f}")
    print(f"{'F1 (LLM+DR system)':<35} {st_f1_system:.3f}{'':<14} {full_f1_system:.3f}")
    print(f"{'ΔF1 from original':<35} +{st_delta_f1:.3f}{'':<13} +{full_delta_f1:.3f}")
    print(f"{'Efficiency (ΔF1 per $100)':<35} {st_efficiency:.4f}{'':<14} {full_efficiency:.4f}")
    print()
    print(f"Marginal analysis (Full DR vs Smart-Trigger):")
    print(f"  Additional cost: ${marginal_cost:.2f}")
    print(f"  Additional ΔF1: +{marginal_delta_f1:.3f}")
    print(f"  Marginal efficiency: {marginal_efficiency:.4f} ΔF1 per $100")
    print()
    print(f"Smart-trigger efficiency ratio: {st_efficiency / full_efficiency:.2f}x better than full DR")
    
    # ---------------------------------------------------------------------------
    # Create plot
    # ---------------------------------------------------------------------------
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cost vs F1 (Pareto frontier)
    ax1 = axes[0]
    
    # Points
    costs = [orig_cost, st_cost, full_cost]
    f1s = [orig_f1, st_f1_system, full_f1_system]
    labels = ['Original\n(No DR)', 'Smart-Trigger\nDR', 'Full DR']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Plot points with CI error bars for smart-trigger and full
    # Two-sided error bars: [lower, upper] derived from Wilson 95% bounds on calibrated rates.
    f1_errors = [
        [0.0, 0.0],  # Original has no uncertainty
        [max(0.0, st_f1_system - st_f1_system_cons), max(0.0, st_f1_system_high - st_f1_system)],
        [max(0.0, full_f1_system - full_f1_system_cons), max(0.0, full_f1_system_high - full_f1_system)],
    ]
    
    for i, (cost, f1, label, color) in enumerate(zip(costs, f1s, labels, colors)):
        ax1.scatter(cost, f1, s=200, c=color, zorder=5, edgecolors='white', linewidths=2)
        ax1.annotate(label, (cost, f1), textcoords="offset points", 
                     xytext=(0, 15), ha='center', fontsize=10, fontweight='bold')
        if i > 0:  # Add error bar for DR scenarios
            ax1.errorbar(cost, f1, yerr=[[f1_errors[i][0]], [f1_errors[i][1]]], 
                        fmt='none', c=color, capsize=5, capthick=2, alpha=0.7)
    
    # Connect points with lines to show efficiency (slope)
    ax1.plot([orig_cost, st_cost], [orig_f1, st_f1_system], 
             'b--', alpha=0.7, linewidth=2, label=f'Smart-trigger path\n({st_efficiency:.4f} ΔF1/$100)')
    ax1.plot([orig_cost, full_cost], [orig_f1, full_f1_system], 
             'r:', alpha=0.7, linewidth=2, label=f'Full DR path\n({full_efficiency:.4f} ΔF1/$100)')
    ax1.plot([st_cost, full_cost], [st_f1_system, full_f1_system], 
             'purple', alpha=0.5, linewidth=1.5, linestyle='-.', 
             label=f'Marginal full→smart\n({marginal_efficiency:.4f} ΔF1/$100)')
    
    ax1.set_xlabel('Cost ($)', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Cost-Accuracy Tradeoff: DR Deployment Strategies', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, full_cost + 30)

    # Add a bit of headroom so the top point/error bar isn't clipped.
    # (This plot is used in the thesis; we prefer visible error bars over tight bounds.)
    y_min = 0.2
    y_err_low = [0.0, st_f1_system - st_f1_system_cons, full_f1_system - full_f1_system_cons]
    y_err_high = [0.0, st_f1_system_high - st_f1_system, full_f1_system_high - full_f1_system]
    y_max = min(1.0, max([f + eh for f, eh in zip(f1s, y_err_high)]) + 0.03)
    # Keep at least the original scale if padding doesn't push it above 0.7.
    y_max = max(0.7, y_max)
    ax1.set_ylim(y_min, y_max)
    
    # Add shaded region showing "efficient frontier"
    ax1.fill_between([orig_cost, st_cost, full_cost], 
                     [orig_f1, st_f1_system, full_f1_system],
                     [y_min, y_min, y_min], alpha=0.1, color='green')
    
    # Plot 2: Cost per unit F1 improvement as CLD scales (1 to 300 variables)
    ax2 = axes[1]
    
    # Observed trigger rate from our data
    trigger_rate = n_triggered / FULL_DR_EDGES  # ~50%
    
    # Cost per 0.1 F1 improvement (from our observed data at N≈15)
    st_cost_per_01f1 = st_cost / (st_delta_f1 * 10)  # cost per 0.1 F1
    full_cost_per_01f1 = full_cost / (full_delta_f1 * 10)
    
    # Generate scaling curves (1 to 300 variables)
    n_variables = np.arange(1, 301, 1)
    
    # Edge density: sparse CLDs
    edge_density = 0.3
    n_edges = edge_density * n_variables * (n_variables - 1)
    n_edges = np.maximum(n_edges, 1)
    
    # Scale factor relative to our 285 edges
    scale = n_edges / FULL_DR_EDGES
    
    # Cost per 0.1 F1 as N grows (scales linearly with edges, so with N²)
    st_cost_per_01f1_scaled = scale * st_cost_per_01f1
    full_cost_per_01f1_scaled = scale * full_cost_per_01f1
    
    # Plot cost per 0.1 F1
    ax2.plot(n_variables, full_cost_per_01f1_scaled, 'r-', linewidth=2.5, 
             label=f'Full DR')
    ax2.plot(n_variables, st_cost_per_01f1_scaled, 'b-', linewidth=2.5, 
             label=f'Smart-trigger')
    
    # Fill the savings gap
    ax2.fill_between(n_variables, st_cost_per_01f1_scaled, full_cost_per_01f1_scaled,
                     alpha=0.3, color='green', label='Savings')
    
    # Mark our actual data point (~15 vars)
    actual_n_vars = 15
    actual_idx = actual_n_vars - 1
    ax2.scatter([actual_n_vars, actual_n_vars], 
                [st_cost_per_01f1_scaled[actual_idx], full_cost_per_01f1_scaled[actual_idx]],
                s=100, c=['blue', 'red'], zorder=5, edgecolors='white', linewidths=2)
    ax2.axvline(x=actual_n_vars, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('Our CLDs', xy=(actual_n_vars, full_cost_per_01f1_scaled[actual_idx]),
                 xytext=(30, 0), textcoords='offset points', fontsize=9, ha='left')
    
    # Add annotations at key points showing the gap
    for n_var in [50, 150, 300]:
        idx = n_var - 1
        savings = full_cost_per_01f1_scaled[idx] - st_cost_per_01f1_scaled[idx]
        y_mid = (full_cost_per_01f1_scaled[idx] + st_cost_per_01f1_scaled[idx]) / 2
        ax2.annotate(f'Save ${savings:,.0f}', 
                     xy=(n_var, y_mid), fontsize=9, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel('Number of Variables in CLD', fontsize=12)
    ax2.set_ylabel('Cost per +0.1 F1 ($)', fontsize=12)
    ax2.set_title('Cost per Unit Accuracy vs CLD Size', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 300)
    ax2.set_ylim(0, None)
    
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / "dr_cost_efficiency_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}")
    
    # Also save a summary JSON
    summary = {
        "cost_parameters": {
            "dr_per_edge": COST_PER_EDGE_DR,
            "judge_per_edge": COST_PER_EDGE_JUDGE,
        },
        "original": {"f1": orig_f1, "cost": orig_cost},
        "smart_trigger": {
            "edges_judged": FULL_DR_EDGES,
            "edges_dr": n_triggered,
            "judge_cost": st_judge_cost,
            "dr_cost": st_dr_cost,
            "total_cost": st_cost,
            "fp_with_evidence": st_fp_triggered_dr,
            "fn_with_evidence": st_fn_triggered_dr,
            "f1_system": st_f1_system,
            "delta_f1": st_delta_f1,
            "efficiency_per_100": st_efficiency,
        },
        "full_dr": {
            "edges": FULL_DR_EDGES,
            "cost": full_cost,
            "fp_with_evidence": FULL_DR_FP_WITH_EVIDENCE,
            "fn_with_evidence": FULL_DR_FN_WITH_EVIDENCE,
            "f1_system": full_f1_system,
            "delta_f1": full_delta_f1,
            "efficiency_per_100": full_efficiency,
        },
        "marginal": {
            "additional_cost": marginal_cost,
            "additional_delta_f1": marginal_delta_f1,
            "efficiency_per_100": marginal_efficiency,
        },
        "efficiency_ratio": st_efficiency / full_efficiency,
    }
    
    summary_path = OUTPUT_DIR / "dr_cost_efficiency_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    # ---------------------------------------------------------------------------
    # Conclusions
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print(f"""
1. SMART-TRIGGER IS MORE COST-EFFICIENT:
   - Smart-trigger: {st_efficiency:.4f} ΔF1 per $100 spent
   - Full DR: {full_efficiency:.4f} ΔF1 per $100 spent
   - Smart-trigger is {st_efficiency/full_efficiency:.1f}× more efficient

2. DIMINISHING MARGINAL RETURNS:
   - First 142 edges (triggered): +{st_delta_f1:.3f} ΔF1 for ${st_cost:.0f}
   - Next 143 edges (non-triggered): +{marginal_delta_f1:.3f} ΔF1 for ${marginal_cost:.0f}
   - Marginal efficiency ({marginal_efficiency:.4f}) is {marginal_efficiency/st_efficiency:.1%} of smart-trigger

3. RECOMMENDATION:
   - If budget-constrained: Use smart-trigger (50% cost, {st_delta_f1/full_delta_f1:.0%} of full F1 gain)
   - If maximizing F1: Full DR still provides additional {marginal_delta_f1:.3f} F1 gain
   - Break-even: Smart-trigger captures {st_fp_triggered_dr/FULL_DR_FP_WITH_EVIDENCE:.0%} of FP evidence at 50% cost
""")
    
    plt.show()


if __name__ == "__main__":
    main()

