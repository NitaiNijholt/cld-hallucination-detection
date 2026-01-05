#!/usr/bin/env python3
"""
RQ2 Ensemble Performance Table Generator

Generates ensemble_performance_table.tex from Phase 4, 5, 6 JSON outputs.
This table shows cross-phase validation analysis for hallucination detection classifiers.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Output directory / analyses directory (optional overrides via env)
from rq2_paths import rq2_dirs
PHASE_RESULTS_DIR, OUTPUT_DIR = rq2_dirs()


def load_latest_phase_results(phase_pattern: str, preferred_file: str = None) -> dict:
    """Load the most recent results JSON for a given phase."""
    phase_dirs = list(PHASE_RESULTS_DIR.glob(phase_pattern))
    if not phase_dirs:
        raise FileNotFoundError(f"No results found matching {phase_pattern}")
    
    latest_dir = max(phase_dirs, key=lambda p: p.stat().st_mtime)
    
    # Find the JSON results file
    json_files = list(latest_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {latest_dir}")
    
    # If preferred file specified, try that first
    if preferred_file:
        matches = [f for f in json_files if f.name == preferred_file]
        if matches:
            with open(matches[0], 'r') as f:
                return json.load(f), latest_dir
    
    # Pick the main results file
    for pattern in ["phase4_rfe_results.json", "phase5_ensemble_results.json", 
                    "phase6_leave_one_out_all_classifiers.json", "phase6_summary_stats.json"]:
        matches = [f for f in json_files if f.name == pattern]
        if matches:
            with open(matches[0], 'r') as f:
                return json.load(f), latest_dir
    
    # Fallback to first JSON
    with open(json_files[0], 'r') as f:
        return json.load(f), latest_dir


def generate_ensemble_table():
    """Generate the ensemble performance LaTeX table."""
    print("="*80)
    print("GENERATING ENSEMBLE PERFORMANCE TABLE")
    print("="*80)
    
    # Load Phase 4 results
    print("\nLoading Phase 4 (RFE) results...")
    phase4_data, phase4_dir = load_latest_phase_results("rq2_phase4_rfe_*")
    print(f"  Loaded from: {phase4_dir.name}")
    
    # Load Phase 5 results
    print("Loading Phase 5 (Ensemble) results...")
    phase5_data, phase5_dir = load_latest_phase_results("rq2_phase5_ensemble_*")
    print(f"  Loaded from: {phase5_dir.name}")
    
    # Load Phase 6 results
    print("Loading Phase 6 (Cross-CLD) results...")
    phase6_data, phase6_dir = load_latest_phase_results(
        "rq2_phase6_cross_cld_*", 
        preferred_file="phase6_leave_one_out_all_classifiers.json"
    )
    print(f"  Loaded from: {phase6_dir.name}")
    
    # Get CV method info
    cv_method = phase4_data.get('cv_method', 'GroupKFold (block = CLD × run)')
    n_blocks = phase4_data.get('n_blocks', 9)
    n_folds = phase4_data.get('n_folds', 3)
    
    print(f"\nCV Method: {cv_method}")
    print(f"N blocks: {n_blocks}, N folds: {n_folds}")
    
    # Classifiers to include
    classifiers = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network']
    
    # Build table data
    table_rows = []
    
    for clf in classifiers:
        row = {'classifier': clf}
        
        # Phase 4 data (RFE) - Neural Network not in RFE
        if clf in phase4_data.get('all_classifiers', {}):
            p4 = phase4_data['all_classifiers'][clf]
            row['p4_mean'] = p4['mean_auc']
            row['p4_min'] = p4['min_auc']
            row['p4_max'] = p4['max_auc']
        else:
            row['p4_mean'] = None
            row['p4_min'] = None
            row['p4_max'] = None
        
        # Phase 5 data
        if clf in phase5_data.get('classifiers', {}):
            p5 = phase5_data['classifiers'][clf]
            row['p5_cv_mean'] = p5['cv_auc_mean']
            row['p5_cv_min'] = p5['cv_auc_min']
            row['p5_cv_max'] = p5['cv_auc_max']
            row['p5_test'] = p5['test_auc']
        else:
            row['p5_cv_mean'] = None
            row['p5_cv_min'] = None
            row['p5_cv_max'] = None
            row['p5_test'] = None
        
        # Phase 6 data (from leave_one_out_all_classifiers or summary_stats)
        if clf in phase6_data:
            # phase6_data is keyed by classifier name with list of per-CLD results
            p6_results = phase6_data[clf]
            aucs = [r['auc'] for r in p6_results]
            row['p6_mean'] = np.mean(aucs)
            row['p6_min'] = np.min(aucs)
            row['p6_max'] = np.max(aucs)
        else:
            row['p6_mean'] = None
            row['p6_min'] = None
            row['p6_max'] = None
        
        # Performance drop
        if row['p5_test'] is not None and row['p6_mean'] is not None:
            row['drop'] = row['p5_test'] - row['p6_mean']
        else:
            row['drop'] = None
        
        table_rows.append(row)
    
    # Generate LaTeX
    latex = generate_latex(table_rows, n_folds, n_blocks, cv_method)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ensemble_performance_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\n✅ Table saved: {output_path}")
    
    # Also save JSON for reference
    json_path = OUTPUT_DIR / "ensemble_performance_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'cv_method': cv_method,
            'n_blocks': n_blocks,
            'n_folds': n_folds,
            'phase4_source': phase4_dir.name,
            'phase5_source': phase5_dir.name,
            'phase6_source': phase6_dir.name,
            'classifiers': table_rows
        }, f, indent=2)
    print(f"✅ JSON saved: {json_path}")
    
    return table_rows


def generate_latex(rows: list, n_folds: int, n_blocks: int, cv_method: str) -> str:
    """Generate the LaTeX table code."""
    
    def fmt_auc(mean, min_val, max_val):
        if mean is None:
            return "N/A"
        return f"{mean:.3f} [{min_val:.3f}, {max_val:.3f}]"
    
    def fmt_test(val):
        if val is None:
            return "N/A"
        return f"{val:.3f}"
    
    def fmt_drop(val):
        if val is None:
            return "---"
        return f"{val:.3f}"
    
    latex = r"""\begin{table}[H]
\centering
\small
\caption{Ensemble Classifier Performance Across Validation Phases (Block-Level Cross-Validation)}
\label{tab:rq2_ensemble_performance}
\begin{threeparttable}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Classifier} & \textbf{Phase 4:} & \textbf{Phase 5:} & \textbf{Phase 5:} & \textbf{Phase 6:} & \textbf{Performance} \\
 & \textbf{RFE CV AUC} & \textbf{CV AUC} & \textbf{Test AUC} & \textbf{Cross-CLD AUC} & \textbf{Drop (5$\rightarrow$6)} \\
 & \textbf{[min, max]} & \textbf{[min, max]} & & \textbf{[min, max]} & \\
\midrule
"""
    
    for i, row in enumerate(rows):
        latex += f"{row['classifier']} & "
        latex += f"{fmt_auc(row['p4_mean'], row['p4_min'], row['p4_max'])} & "
        latex += f"{fmt_auc(row['p5_cv_mean'], row['p5_cv_min'], row['p5_cv_max'])} & "
        latex += f"{fmt_test(row['p5_test'])} & "
        latex += f"{fmt_auc(row['p6_mean'], row['p6_min'], row['p6_max'])} & "
        latex += f"{fmt_drop(row['drop'])} \\\\\n"
        if i < len(rows) - 1:
            latex += r"\midrule" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}}
\begin{tablenotes}
\small
"""
    
    latex += rf"""\item \textit{{Note.}} Block-level cross-validation using {n_folds}-fold GroupKFold with blocks = (CLD$\times$run, $N={n_blocks}$). Entire blocks stay together in train OR test to prevent pseudo-replication.
\textbf{{Phase 4 (RFE CV AUC):}} {n_folds}-fold block-level CV on all pooled edges; measures feature selection performance.
\textbf{{Phase 5 (CV AUC):}} {n_folds}-fold block-level CV on training blocks ($\sim$67\% of blocks); measures in-distribution performance.
\textbf{{Phase 5 (Test AUC):}} Single evaluation on held-out test blocks ($\sim$33\% of blocks); validates generalization within same distribution.
\textbf{{Phase 6 (Cross-CLD AUC):}} Leave-one-CLD-out CV---train on 2 CLDs, test on held-out 3rd CLD, averaged across all 3 CLDs; measures cross-domain generalization.
Phase 4 evaluates all 4 UQ metrics via RFE for each classifier; the best-performing classifier (Random Forest) selected all 4 features, which are then used in Phases 5--6. 
\textbf{{Uncertainty:}} All phases show mean [min, max] across CV folds or CLDs, per the uncertainty reporting rule (Methods Section~\ref{{sec:uncertainty_rule}}). For $n={n_folds}$ (Phases 4--5) or $n=3$ (Phase 6), min-max ranges are more appropriate than t-distribution CIs.
Performance Drop = Phase 5 Test AUC $-$ Phase 6 Mean AUC.
"""
    
    latex += r"""\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    return latex


def main():
    print("\n" + "="*80)
    print("RQ2 ENSEMBLE PERFORMANCE TABLE GENERATOR")
    print("="*80)
    print("Generates ensemble_performance_table.tex from Phase 4-6 results")
    print("="*80 + "\n")
    
    try:
        rows = generate_ensemble_table()
        
        print("\n" + "="*80)
        print("TABLE SUMMARY")
        print("="*80)
        for row in rows:
            p5_test = row['p5_test'] if row['p5_test'] else 0
            p6_mean = row['p6_mean'] if row['p6_mean'] else 0
            drop = row['drop'] if row['drop'] else 0
            print(f"{row['classifier']:25s}: Phase 5 Test={p5_test:.3f}, Phase 6={p6_mean:.3f}, Drop={drop:.3f}")
        
        print("\n✅ ENSEMBLE TABLE GENERATION COMPLETE!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

