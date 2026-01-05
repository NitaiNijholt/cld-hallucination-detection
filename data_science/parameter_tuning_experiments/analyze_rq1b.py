#!/usr/bin/env python3
"""
RQ1b Corrector Analysis - Generate All Thesis Tables

Generates all RQ1b tables for the thesis:
- Table 7: Baseline corrector on synthetic corruptions (per-run mean ± std)
- Table 8: Prompt ablation on synthetic corruptions (totals)
- Table 9: Baseline corrector on ground truth (per-run mean ± std)
- Table 10: Prompt ablation on ground truth (totals)
- Appendix: Full action count breakdowns (including None/Error)

Metrics analyzed:
- F1 Delta (precision, recall)
- Judge Score Delta
- Action distribution (revise vs change_type)
- Per-CLD breakdown
- Macro-averaged aggregate statistics

Usage:
    python analyze_rq1b.py --type synthetic
    python analyze_rq1b.py --type groundtruth
    python analyze_rq1b.py --type both
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

VARIANTS = ['baseline', 'cot', 'mechanistic']
BASE_PATH = Path('final_runs')

# Folder patterns for each experiment type
FOLDER_PATTERNS = {
    'synthetic': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_{variant}',
    'groundtruth': 'RQ1b_corrector_experiment_ground_truth_correctness_{variant}'
}


def bonferroni_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction to a family of p-values.
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


def load_results(variant: str, experiment_type: str) -> pd.DataFrame:
    """Load results for a specific variant and experiment type."""
    folder_pattern = FOLDER_PATTERNS[experiment_type]
    folder = BASE_PATH / folder_pattern.format(variant=variant)
    results_file = folder / 'rq1b_all_results.xlsx'
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    # Try different sheet names (Excel structure varies)
    xl = pd.ExcelFile(results_file)
    sheet_names = xl.sheet_names
    
    if 'All_Results' in sheet_names:
        df = pd.read_excel(results_file, sheet_name='All_Results')
    elif 'Sheet1' in sheet_names:
        df = pd.read_excel(results_file, sheet_name='Sheet1')
    else:
        df = pd.read_excel(results_file, sheet_name=0)  # First sheet
    
    df['Variant'] = variant
    return df


def calculate_per_cld_means(combined: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-CLD means for macro averaging."""
    # Define action columns, handling optional ones that may not exist in older Excel files
    action_cols = ['Actions_Total', 'Actions_Revise', 'Actions_Change']
    optional_action_cols = ['Actions_None', 'Actions_Error', 'Actions_Remove', 'Actions_Flip']
    
    for col in optional_action_cols:
        if col in combined.columns:
            action_cols.append(col)
        else:
            combined[col] = 0  # Default to 0 for missing columns
            action_cols.append(col)
    
    per_cld_means = combined.groupby(['Variant', 'CLD']).agg({
        'F1_Delta': 'mean',
        'Precision_Delta': 'mean',
        'Recall_Delta': 'mean',
        'Judge_Score_Delta': 'mean',
        'Actions_Total': 'sum',
        'Actions_Revise': 'sum',
        'Actions_Change': 'sum',
        'Actions_None': 'sum',
        'Actions_Error': 'sum',
        'Actions_Remove': 'sum',
        'Actions_Flip': 'sum'
    }).reset_index()
    
    # VERIFY: Total should equal sum of sub-actions
    computed_total = (per_cld_means['Actions_Revise'] + per_cld_means['Actions_Change'] + 
                      per_cld_means['Actions_None'] + per_cld_means['Actions_Error'] + 
                      per_cld_means['Actions_Remove'] + per_cld_means['Actions_Flip'])
    
    mismatch = per_cld_means[per_cld_means['Actions_Total'] != computed_total]
    if len(mismatch) > 0:
        print("\n⚠️  ACTION COUNT MISMATCH DETECTED:")
        for _, row in mismatch.iterrows():
            print(f"   {row['Variant']}/{row['CLD']}: Total={int(row['Actions_Total'])}, "
                  f"Computed={int(computed_total[_])}")
            print(f"      Revise={int(row['Actions_Revise'])}, Change={int(row['Actions_Change'])}, "
                  f"None={int(row['Actions_None'])}, Error={int(row['Actions_Error'])}, "
                  f"Remove={int(row['Actions_Remove'])}, Flip={int(row['Actions_Flip'])}")
    else:
        print("✓ Action counts verified: Total = sum of sub-actions")
    
    return per_cld_means


def calculate_macro_average(per_cld_means: pd.DataFrame) -> pd.DataFrame:
    """Calculate macro-averaged statistics (mean ± std across CLDs)."""
    macro_avg = per_cld_means.groupby('Variant').agg({
        'F1_Delta': ['mean', 'std'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std'],
        'Actions_Total': 'sum',
        'Actions_Revise': 'sum',
        'Actions_Change': 'sum',
        'Actions_None': 'sum',
        'Actions_Error': 'sum',
        'Actions_Remove': 'sum',
        'Actions_Flip': 'sum'
    })
    
    # Flatten column names
    macro_avg.columns = ['_'.join(col).strip('_') for col in macro_avg.columns.values]
    return macro_avg


def generate_latex_table(combined: pd.DataFrame, experiment_type: str, output_dir: Path) -> str:
    """Generate LaTeX table with macro-averaged aggregate row."""
    
    # CLD order must match LaTeX header: Social Norms, Depressive, Emergency Dept
    clds = ['social_norms', 'depressive', 'emergency_department']
    cld_short_names = {
        'social_norms': 'Social Norms',
        'depressive': 'Depressive',
        'emergency_department': 'Emergency Dept'
    }
    
    # Calculate per-CLD means
    per_cld_means = calculate_per_cld_means(combined)
    
    # Calculate macro averages
    macro_avg = calculate_macro_average(per_cld_means)
    
    # Build table rows
    rows = []
    
    for variant in VARIANTS:
        variant_data = per_cld_means[per_cld_means['Variant'] == variant]
        macro = macro_avg.loc[variant]
        
        # Get per-CLD F1 values
        cld_f1s = []
        for cld in clds:
            cld_row = variant_data[variant_data['CLD'] == cld]
            if len(cld_row) > 0:
                f1 = cld_row['F1_Delta'].values[0]
                cld_f1s.append(f"+{f1:.3f}" if f1 >= 0 else f"{f1:.3f}")
            else:
                cld_f1s.append("N/A")
        
        # Format overall F1 (macro-averaged mean ± std)
        f1_mean = macro['F1_Delta_mean']
        f1_std = macro['F1_Delta_std']
        overall_f1 = f"${'+' if f1_mean >= 0 else ''}{f1_mean:.3f} \\pm {f1_std:.3f}$"
        
        # Format overall Judge Score (macro-averaged mean ± std)
        judge_mean = macro['Judge_Score_Delta_mean']
        judge_std = macro['Judge_Score_Delta_std']
        if np.isnan(judge_std):
            judge_std = 0.0  # Handle NaN (single value case)
        overall_judge = f"${'+' if judge_mean >= 0 else ''}{judge_mean:.3f} \\pm {judge_std:.3f}$"
        
        # Action totals - Total = Revise + Change (successful corrections only)
        revise = int(macro['Actions_Revise_sum'])
        change = int(macro['Actions_Change_sum'])
        total = revise + change  # Successful corrections only
        
        # Track None and Error for appendix
        none_count = int(macro.get('Actions_None_sum', 0))
        error_count = int(macro.get('Actions_Error_sum', 0))
        
        # Build row
        variant_name = variant.capitalize() if variant != 'cot' else 'CoT'
        row = f"{variant_name} & {overall_f1} & {overall_judge} & {cld_f1s[0]} & {cld_f1s[1]} & {cld_f1s[2]} & {total} & {revise} & {change} \\\\"
        rows.append(row)
    
    # Add aggregate row (macro-averaged across all variants and CLDs)
    aggregate_data = []
    for metric in ['F1_Delta', 'Precision_Delta', 'Recall_Delta', 'Judge_Score_Delta']:
        values = per_cld_means[metric].values
        aggregate_data.append({
            'metric': metric,
            'mean': values.mean(),
            'std': values.std()
        })
    
    # Total actions across all variants - Total = Revise + Change (successful corrections only)
    total_revise = int(per_cld_means['Actions_Revise'].sum())
    total_change = int(per_cld_means['Actions_Change'].sum())
    total_actions = total_revise + total_change  # Successful corrections only
    
    # Track None and Error for appendix
    total_none = int(per_cld_means['Actions_None'].sum())
    total_error = int(per_cld_means['Actions_Error'].sum())
    
    # Aggregate F1 per CLD (mean across variants)
    aggregate_per_cld = per_cld_means.groupby('CLD')['F1_Delta'].mean()
    agg_cld_f1s = []
    for cld in clds:
        f1 = aggregate_per_cld.get(cld, 0)
        agg_cld_f1s.append(f"+{f1:.3f}" if f1 >= 0 else f"{f1:.3f}")
    
    # Overall aggregate F1
    agg_f1_mean = per_cld_means['F1_Delta'].mean()
    agg_f1_std = per_cld_means['F1_Delta'].std()
    
    # Overall aggregate Judge Score
    agg_judge_mean = per_cld_means['Judge_Score_Delta'].mean()
    agg_judge_std = per_cld_means['Judge_Score_Delta'].std()
    if np.isnan(agg_judge_std):
        agg_judge_std = 0.0
    
    aggregate_row = f"\\midrule\n\\textbf{{Aggregate}} & ${'+' if agg_f1_mean >= 0 else ''}{agg_f1_mean:.3f}$ $\\pm$ {agg_f1_std:.3f} & ${'+' if agg_judge_mean >= 0 else ''}{agg_judge_mean:.3f}$ $\\pm$ {agg_judge_std:.3f} & {agg_cld_f1s[0]} & {agg_cld_f1s[1]} & {agg_cld_f1s[2]} & {total_actions} & {total_revise} & {total_change} \\\\"
    
    # Build full LaTeX table
    exp_type_title = "Synthetic Corruptions" if experiment_type == 'synthetic' else "Ground Truth"
    
    latex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Corrector Prompt Ablation on {exp_type_title}: Performance Comparison}}
\\label{{tab:rq1b_prompt_ablation_{experiment_type}}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{lcccccccc}}
\\toprule
\\textbf{{Prompt}} & \\textbf{{Overall F1 $\\Delta$}} & \\textbf{{Judge $\\Delta$}} & \\textbf{{Social Norms}} & \\textbf{{Depressive}} & \\textbf{{Emergency Dept}} & \\textbf{{Total}} & \\textbf{{Revise}} & \\textbf{{Change}} \\\\
\\textbf{{Variant}} & \\textbf{{(mean $\\pm$ std)}} & \\textbf{{(mean $\\pm$ std)}} & \\textbf{{F1 $\\Delta$}} & \\textbf{{F1 $\\Delta$}} & \\textbf{{F1 $\\Delta$}} & \\textbf{{Actions}} & & \\\\
\\midrule
{chr(10).join(rows)}
{aggregate_row}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Note.}} Ablation study comparing three corrector prompt variants (3 runs per CLD per prompt, 27 total experiments). 
Aggregate row shows macro-averaged mean $\\pm$ std across CLDs.
F1 $\\Delta$ = change in F1 score from pre- to post-correction; Judge $\\Delta$ = change in LLM judge score.
Total Actions = Revise + Change (successful corrections only); Revise = motivation revisions; Change = edge type changes.
Edges with no correction needed or API errors excluded (see Appendix Table~\\ref{{tab:rq1b_action_counts_{experiment_type}}} for full breakdown).
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    # Save LaTeX file
    latex_file = output_dir / f'ablation_table_{experiment_type}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to: {latex_file}")
    
    return latex


def generate_appendix_action_table(combined: pd.DataFrame, experiment_type: str, output_dir: Path) -> str:
    """Generate appendix table with full action count breakdown including None and Error."""
    
    # Calculate per-CLD means
    per_cld_means = calculate_per_cld_means(combined)
    
    # Calculate macro averages
    macro_avg = calculate_macro_average(per_cld_means)
    
    # Build table rows
    rows = []
    
    for variant in VARIANTS:
        macro = macro_avg.loc[variant]
        
        # Get all action counts
        revise = int(macro['Actions_Revise_sum'])
        change = int(macro['Actions_Change_sum'])
        none_count = int(macro.get('Actions_None_sum', 0))
        error_count = int(macro.get('Actions_Error_sum', 0))
        total = revise + change + none_count + error_count  # Full total
        successful = revise + change  # Successful only
        
        # Build row
        variant_name = variant.capitalize() if variant != 'cot' else 'CoT'
        row = f"{variant_name} & {total} & {successful} & {revise} & {change} & {none_count} & {error_count} \\\\"
        rows.append(row)
    
    # Aggregate row
    total_revise = int(per_cld_means['Actions_Revise'].sum())
    total_change = int(per_cld_means['Actions_Change'].sum())
    total_none = int(per_cld_means['Actions_None'].sum())
    total_error = int(per_cld_means['Actions_Error'].sum())
    full_total = total_revise + total_change + total_none + total_error
    successful_total = total_revise + total_change
    
    aggregate_row = f"\\midrule\n\\textbf{{Aggregate}} & {full_total} & {successful_total} & {total_revise} & {total_change} & {total_none} & {total_error} \\\\"
    
    # Build full LaTeX table
    exp_type_title = "Synthetic Corruptions" if experiment_type == 'synthetic' else "Ground Truth"
    
    latex = f"""\\begin{{table}}[H]
\\centering
\\caption{{RQ1b Corrector Action Counts - {exp_type_title} (Full Breakdown)}}
\\label{{tab:rq1b_action_counts_{experiment_type}}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Prompt}} & \\textbf{{Edges}} & \\textbf{{Successful}} & \\textbf{{Revise}} & \\textbf{{Change}} & \\textbf{{None}} & \\textbf{{Error}} \\\\
\\textbf{{Variant}} & \\textbf{{Processed}} & \\textbf{{Actions}} & & & & \\\\
\\midrule
{chr(10).join(rows)}
{aggregate_row}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Note.}} Full breakdown of corrector actions for verification. 
Edges Processed = total candidate edges for correction.
Successful Actions = Revise + Change (used in main text tables).
None = edges where corrector determined no change needed.
Error = API/parsing errors during correction.
Verification: Edges Processed = Revise + Change + None + Error.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    # Save LaTeX file
    latex_file = output_dir / f'appendix_action_counts_{experiment_type}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ Appendix action table saved to: {latex_file}")
    
    return latex


def generate_baseline_table(combined: pd.DataFrame, experiment_type: str, output_dir: Path) -> str:
    """Generate baseline-only table (Tables 7 and 9 in thesis) showing per-CLD breakdown."""
    
    # Filter to baseline only
    baseline_data = combined[combined['Variant'] == 'baseline'].copy()
    
    if baseline_data.empty:
        print(f"⚠️  No baseline data found for {experiment_type}")
        return ""
    
    # CLD order must match LaTeX header
    clds = ['social_norms', 'depressive', 'emergency_department']
    cld_display_names = {
        'social_norms': 'Social Norms',
        'depressive': 'Depressive',
        'emergency_department': 'Emergency Dept'
    }
    
    # Calculate per-CLD means - for actions, compute per-run mean ± std (not totals)
    # First compute Total = Revise + Change for each row
    baseline_data['Actions_Successful'] = baseline_data['Actions_Revise'] + baseline_data['Actions_Change']
    
    per_cld = baseline_data.groupby('CLD').agg({
        'F1_Delta': ['mean', 'std'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std'],
        'Actions_Successful': ['mean', 'std'],  # Per-run mean ± std
        'Actions_Revise': ['mean', 'std'],      # Per-run mean ± std
        'Actions_Change': ['mean', 'std'],      # Per-run mean ± std
        'Actions_None': 'sum',
        'Actions_Error': 'sum'
    })
    
    # Flatten column names
    per_cld.columns = ['_'.join(col).strip('_') for col in per_cld.columns.values]
    
    # Build table rows
    rows = []
    for cld in clds:
        if cld not in per_cld.index:
            continue
        row_data = per_cld.loc[cld]
        
        f1_mean = row_data['F1_Delta_mean']
        f1_std = row_data['F1_Delta_std']
        prec_mean = row_data['Precision_Delta_mean']
        prec_std = row_data['Precision_Delta_std']
        rec_mean = row_data['Recall_Delta_mean']
        rec_std = row_data['Recall_Delta_std']
        judge_mean = row_data['Judge_Score_Delta_mean']
        judge_std = row_data['Judge_Score_Delta_std']
        
        # Actions: Per-run mean ± std (Total = Revise + Change)
        total_mean = row_data['Actions_Successful_mean']
        total_std = row_data['Actions_Successful_std']
        revise_mean = row_data['Actions_Revise_mean']
        revise_std = row_data['Actions_Revise_std']
        change_mean = row_data['Actions_Change_mean']
        change_std = row_data['Actions_Change_std']
        
        # Format with sign
        def fmt(mean, std):
            sign = '+' if mean >= 0 else ''
            return f"${sign}{mean:.3f} \\pm {std:.3f}$"
        
        row = f"{cld_display_names[cld]} & {fmt(f1_mean, f1_std)} & {fmt(prec_mean, prec_std)} & {fmt(rec_mean, rec_std)} & {fmt(judge_mean, judge_std)} & ${total_mean:.1f} \\pm {total_std:.1f}$ & ${revise_mean:.1f} \\pm {revise_std:.1f}$ & ${change_mean:.1f} \\pm {change_std:.1f}$ \\\\"
        rows.append(row)
    
    # Macro average row - mean of per-CLD means ± std across CLDs
    macro_f1_mean = per_cld['F1_Delta_mean'].mean()
    macro_f1_std = per_cld['F1_Delta_mean'].std()
    macro_prec_mean = per_cld['Precision_Delta_mean'].mean()
    macro_prec_std = per_cld['Precision_Delta_mean'].std()
    macro_rec_mean = per_cld['Recall_Delta_mean'].mean()
    macro_rec_std = per_cld['Recall_Delta_mean'].std()
    macro_judge_mean = per_cld['Judge_Score_Delta_mean'].mean()
    macro_judge_std = per_cld['Judge_Score_Delta_mean'].std()
    
    # Aggregate actions - mean of per-CLD per-run means ± std across CLDs
    import numpy as np
    macro_total_mean = per_cld['Actions_Successful_mean'].mean()
    macro_total_std = per_cld['Actions_Successful_mean'].std()
    macro_revise_mean = per_cld['Actions_Revise_mean'].mean()
    macro_revise_std = per_cld['Actions_Revise_mean'].std()
    macro_change_mean = per_cld['Actions_Change_mean'].mean()
    macro_change_std = per_cld['Actions_Change_mean'].std()
    
    def fmt_macro(mean, std):
        sign = '+' if mean >= 0 else ''
        return f"${sign}{mean:.3f}$ $\\pm$ {std:.3f}"
    
    macro_row = f"\\midrule\nMacro Avg & {fmt_macro(macro_f1_mean, macro_f1_std)} & {fmt_macro(macro_prec_mean, macro_prec_std)} & {fmt_macro(macro_rec_mean, macro_rec_std)} & {fmt_macro(macro_judge_mean, macro_judge_std)} & ${macro_total_mean:.1f}$ $\\pm$ {macro_total_std:.1f} & ${macro_revise_mean:.1f}$ $\\pm$ {macro_revise_std:.1f} & ${macro_change_mean:.1f}$ $\\pm$ {macro_change_std:.1f} \\\\"
    
    # Build full LaTeX table
    exp_type_title = "Synthetically Corrupted Data" if experiment_type == 'synthetic' else "Ground Truth Data Using Correctness-Based Validation"
    label_suffix = "synthetic_correctness" if experiment_type == 'synthetic' else "groundtruth_correctness"
    
    latex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Corrector Performance on {exp_type_title}}}
\\label{{tab:rq1b_{label_suffix}}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
\\textbf{{CLD}} & \\textbf{{F1 $\\Delta$}} & \\textbf{{Precision $\\Delta$}} & \\textbf{{Recall $\\Delta$}} & \\textbf{{Judge Score $\\Delta$}} & \\textbf{{Total Actions}} & \\textbf{{Revise}} & \\textbf{{Change Type}} \\\\
\\midrule
{chr(10).join(rows)}
{macro_row}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Note.}} Results show change in performance ($\\Delta$) from pre- to post-correction (3 runs per CLD).
F1 $\\Delta$ = change in F1 score (Post-Pre); Precision $\\Delta$, Recall $\\Delta$, Judge Score $\\Delta$ = corresponding changes.
Total Actions = Revise + Change (successful corrections only); Revise = motivation revisions; Change Type = edge type changes.
Values presented as mean $\\pm$ std across 3 runs per CLD. Macro average shows mean of CLD means $\\pm$ std across CLDs.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    # Save LaTeX file
    latex_file = output_dir / f'baseline_table_{experiment_type}.tex'
    with open(latex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ Baseline table saved to: {latex_file}")
    
    return latex


def analyze_experiment(experiment_type: str, output_dir: Path):
    """Run full analysis for one experiment type."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {experiment_type.upper()} ABLATION")
    print(f"{'='*80}")
    
    # Load all results
    all_results = []
    for variant in VARIANTS:
        try:
            df = load_results(variant, experiment_type)
            all_results.append(df)
            print(f"✅ Loaded {variant}: {len(df)} experiments")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return None
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # 1. Overall Performance by Variant
    print("\n" + "-"*60)
    print("OVERALL PERFORMANCE BY VARIANT")
    print("-"*60)
    
    summary = combined.groupby('Variant').agg({
        'F1_Delta': ['mean', 'std', 'count'],
        'Precision_Delta': ['mean', 'std'],
        'Recall_Delta': ['mean', 'std'],
        'Judge_Score_Delta': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # 2. Per-CLD Breakdown
    print("\n" + "-"*60)
    print("PER-CLD BREAKDOWN (F1 Delta)")
    print("-"*60)
    
    for cld in sorted(combined['CLD'].unique()):
        print(f"\n{cld.upper()}:")
        cld_data = combined[combined['CLD'] == cld]
        cld_summary = cld_data.groupby('Variant')['F1_Delta'].agg(['mean', 'std', 'count']).round(4)
        print(cld_summary)
    
    # 3. Macro-Averaged Statistics
    print("\n" + "-"*60)
    print("MACRO-AVERAGED STATISTICS (mean ± std across CLDs)")
    print("-"*60)
    
    per_cld_means = calculate_per_cld_means(combined)
    macro_avg = calculate_macro_average(per_cld_means)
    print(macro_avg.round(4))
    
    # 4. Action Distribution
    print("\n" + "-"*60)
    print("ACTION DISTRIBUTION BY VARIANT")
    print("-"*60)
    
    action_summary = combined.groupby('Variant').agg({
        'Actions_Revise': 'sum',
        'Actions_Change': 'sum',
        'Actions_Remove': 'sum',
        'Actions_Flip': 'sum',
        'Actions_Total': 'sum'
    })
    
    print(action_summary)
    
    # 5. Statistical Tests
    print("\n" + "-"*60)
    print("STATISTICAL SIGNIFICANCE")
    print("-"*60)
    
    try:
        from scipy import stats
        
        baseline_f1 = combined[combined['Variant'] == 'baseline']['F1_Delta'].dropna()
        cot_f1 = combined[combined['Variant'] == 'cot']['F1_Delta'].dropna()
        mech_f1 = combined[combined['Variant'] == 'mechanistic']['F1_Delta'].dropna()
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(baseline_f1, cot_f1, mech_f1)
        print(f"\nANOVA (F1 Delta):")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  ✅ Significant difference between variants (p < 0.05)")
            
            # Pairwise t-tests with Bonferroni correction
            print("\nPairwise t-tests (with Bonferroni correction):")
            pairs = [('baseline', 'cot'), ('baseline', 'mechanistic'), ('cot', 'mechanistic')]
            pairwise_p_values = {}
            
            for v1, v2 in pairs:
                data1 = combined[combined['Variant'] == v1]['F1_Delta'].dropna()
                data2 = combined[combined['Variant'] == v2]['F1_Delta'].dropna()
                t_stat, p_val = stats.ttest_ind(data1, data2)
                pairwise_p_values[f"{v1}_vs_{v2}"] = p_val
            
            correction = bonferroni_correction(pairwise_p_values, alpha=0.05)
            print(f"  α_adjusted = {correction['alpha_adjusted']:.4f} (Bonferroni, m={correction['n_tests']})")
            
            for name, res in correction['results'].items():
                sig_raw = "✅" if res['significant_original'] else "  "
                sig_adj = "†" if res['significant_adjusted'] else " "
                print(f"  {sig_raw}{sig_adj} {name}: p={res['p_original']:.4f}, p_adj={res['p_adjusted']:.4f}")
        else:
            print("  ❌ No significant difference between variants (p >= 0.05)")
    
    except ImportError:
        print("⚠️  scipy not available, skipping statistical tests")
    except Exception as e:
        print(f"⚠️  Statistical tests failed: {e}")
    
    # 6. Generate LaTeX table with aggregate row
    print("\n" + "-"*60)
    print("GENERATING LATEX TABLE")
    print("-"*60)
    
    latex = generate_latex_table(combined, experiment_type, output_dir)
    
    # Also generate appendix table with full action counts
    appendix_latex = generate_appendix_action_table(combined, experiment_type, output_dir)
    
    # Also generate baseline-only table (Tables 7 and 9 in thesis)
    baseline_latex = generate_baseline_table(combined, experiment_type, output_dir)
    
    # 7. Export combined results
    output_file = output_dir / f'corrector_ablation_{experiment_type}_results.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='All_Results', index=False)
        per_cld_means.to_excel(writer, sheet_name='Per_CLD_Means', index=False)
        macro_avg.to_excel(writer, sheet_name='Macro_Average')
    
    print(f"✅ Results exported to: {output_file}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Analyze corrector ablation study results')
    parser.add_argument('--type', choices=['synthetic', 'groundtruth', 'both'], 
                       default='both',
                       help='Type of experiment to analyze')
    args = parser.parse_args()
    
    print("="*80)
    print("CORRECTOR ABLATION ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent.parent / 'final_runs' / 'RQ1b_corrector_ablation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments_to_run = []
    if args.type == 'both':
        experiments_to_run = ['synthetic', 'groundtruth']
    else:
        experiments_to_run = [args.type]
    
    results = {}
    for exp_type in experiments_to_run:
        result = analyze_experiment(exp_type, output_dir)
        if result is not None:
            results[exp_type] = result
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Experiments analyzed: {', '.join(results.keys())}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
