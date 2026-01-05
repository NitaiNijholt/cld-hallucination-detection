#!/usr/bin/env python3
"""
RQ2 Master Report Generator (Version 2)

Creates a comprehensive master report synthesizing all RQ2 analyses
following the new logical pipeline order:
  Phase 1-3: Individual analysis and aggregation
  Phase 4: RFE Feature Selection
  Phase 5: Ensemble with optimal features
  Phase 6: Cross-CLD generalization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

from rq2_paths import rq2_dirs


def load_latest_results(analyses_dir: Path):
    """Load results from all analysis stages."""
    results = {}
    
    print("Loading results from all analysis phases...")
    
    # 1. Load file inventory
    inventory_files = list(analyses_dir.glob("rq2_file_inventory_*.json"))
    if inventory_files:
        latest_inventory = max(inventory_files, key=lambda p: p.stat().st_mtime)
        with open(latest_inventory, 'r') as f:
            results['inventory'] = json.load(f)
        print(f"  ✅ Inventory: {latest_inventory.name}")
    
    # 2. Load batch results
    batch_dirs = list(analyses_dir.glob("rq2_batch_*"))
    if batch_dirs:
        latest_batch = max(batch_dirs, key=lambda p: p.stat().st_mtime)
        results['batch_dir'] = latest_batch
        print(f"  ✅ Batch: {latest_batch.name}")
    
    # 3. Load aggregates
    aggregate_dirs = list(analyses_dir.glob("rq2_aggregates_*"))
    if aggregate_dirs:
        latest_aggregate = max(aggregate_dirs, key=lambda p: p.stat().st_mtime)
        results['aggregate_dir'] = latest_aggregate
        print(f"  ✅ Aggregates: {latest_aggregate.name}")
    
    # 4. Load grand aggregate
    grand_dirs = list(analyses_dir.glob("rq2_grand_aggregate_*"))
    if grand_dirs:
        latest_grand = max(grand_dirs, key=lambda p: p.stat().st_mtime)
        
        grand_file = latest_grand / "grand_meta_analysis.xlsx"
        if grand_file.exists():
            results['grand_aggregate'] = pd.read_excel(grand_file, engine='openpyxl')
        
        results['grand_dir'] = latest_grand
        print(f"  ✅ Grand Aggregate: {latest_grand.name}")
    
    # 5. Load Phase 4: RFE Feature Selection
    phase4_dirs = list(analyses_dir.glob("rq2_phase4_rfe_*"))
    if phase4_dirs:
        latest_phase4 = max(phase4_dirs, key=lambda p: p.stat().st_mtime)
        
        phase4_results_file = latest_phase4 / "phase4_rfe_results.json"
        if phase4_results_file.exists():
            with open(phase4_results_file, 'r') as f:
                results['phase4_rfe'] = json.load(f)
        
        phase4_report_file = latest_phase4 / "phase4_rfe_report.md"
        if phase4_report_file.exists():
            with open(phase4_report_file, 'r') as f:
                results['phase4_report_content'] = f.read()
        
        results['phase4_dir'] = latest_phase4
        print(f"  ✅ Phase 4 (RFE): {latest_phase4.name}")
    
    # 6. Load Phase 5: Ensemble with optimal features
    phase5_dirs = list(analyses_dir.glob("rq2_phase5_ensemble_*"))
    if phase5_dirs:
        latest_phase5 = max(phase5_dirs, key=lambda p: p.stat().st_mtime)
        
        phase5_results_file = latest_phase5 / "phase5_ensemble_results.json"
        if phase5_results_file.exists():
            with open(phase5_results_file, 'r') as f:
                results['phase5_ensemble'] = json.load(f)
        
        phase5_report_file = latest_phase5 / "phase5_ensemble_report.md"
        if phase5_report_file.exists():
            with open(phase5_report_file, 'r') as f:
                results['phase5_report_content'] = f.read()
        
        results['phase5_dir'] = latest_phase5
        print(f"  ✅ Phase 5 (Ensemble): {latest_phase5.name}")
    
    # 7. Load Phase 6: Cross-CLD generalization
    phase6_dirs = list(analyses_dir.glob("rq2_phase6_cross_cld_*"))
    if phase6_dirs:
        latest_phase6 = max(phase6_dirs, key=lambda p: p.stat().st_mtime)
        
        phase6_results_file = latest_phase6 / "phase6_leave_one_out_results.json"
        if phase6_results_file.exists():
            with open(phase6_results_file, 'r') as f:
                results['phase6_cross_cld'] = json.load(f)
        
        phase6_report_file = latest_phase6 / "phase6_cross_cld_report.md"
        if phase6_report_file.exists():
            with open(phase6_report_file, 'r') as f:
                results['phase6_report_content'] = f.read()
        
        results['phase6_dir'] = latest_phase6
        print(f"  ✅ Phase 6 (Cross-CLD): {latest_phase6.name}")
    
    print()
    return results


def generate_master_report(results: dict, output_path: Path):
    """Generate comprehensive master report."""
    
    with open(output_path, 'w') as f:
        # Header
        f.write("# RQ2: Context-Insensitive Metrics for Hallucination Detection\n\n")
        f.write("## Master Analysis Report (Version 2)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Research Question:** Can context-insensitive (CI) metrics detect hallucinations in LLM-generated causal loop diagrams?\n\n")
        
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        if 'phase4_rfe' in results and 'phase5_ensemble' in results and 'phase6_cross_cld' in results:
            phase4 = results['phase4_rfe']
            phase5 = results['phase5_ensemble']
            phase6_results = results['phase6_cross_cld']
            
            phase6_mean_auc = np.mean([r['auc'] for r in phase6_results])
            
            f.write(f"### Key Findings\n\n")
            f.write(f"1. **Optimal Feature Set (Phase 4):** {', '.join(phase4['optimal_features'])}\n")
            f.write(f"   - Selected via RFE with {phase4['best_classifier']}\n")
            f.write(f"   - Cross-validation AUC: {phase4['best_auc']:.3f} ± {phase4['best_auc_std']:.3f}\n\n")
            
            f.write(f"2. **Best Classifier (Phase 5):** {phase5['best_classifier']}\n")
            f.write(f"   - Test Set AUC: {phase5['best_test_auc']:.3f}\n")
            f.write(f"   - Using {len(phase5['features_used'])} optimized features\n\n")
            
            f.write(f"3. **Cross-Domain Generalization (Phase 6):** \n")
            f.write(f"   - Leave-one-CLD-out AUC: {phase6_mean_auc:.3f}\n")
            
            if phase6_mean_auc >= 0.8:
                f.write(f"   - ✅ **Excellent** generalization across causal domains\n\n")
            elif phase6_mean_auc >= 0.7:
                f.write(f"   - ⚠️  **Good** generalization (calibration recommended)\n\n")
            else:
                f.write(f"   - ⚠️  **Moderate** generalization (fine-tuning needed)\n\n")
            
            f.write(f"### Bottom Line\n\n")
            f.write(f"**Answer to RQ2:** ")
            
            if phase5['best_test_auc'] > 0.9:
                f.write(f"**YES** - CI metrics can **excellently** detect hallucinations (AUC={phase5['best_test_auc']:.3f})\n\n")
            elif phase5['best_test_auc'] > 0.8:
                f.write(f"**YES** - CI metrics can **reliably** detect hallucinations (AUC={phase5['best_test_auc']:.3f})\n\n")
            elif phase5['best_test_auc'] > 0.7:
                f.write(f"**PARTIALLY** - CI metrics show **moderate** detection ability (AUC={phase5['best_test_auc']:.3f})\n\n")
            else:
                f.write(f"**LIMITED** - CI metrics show only **weak** detection ability (AUC={phase5['best_test_auc']:.3f})\n\n")
            
            f.write(f"**Production Readiness:** ")
            if phase6_mean_auc >= 0.8 and phase5['best_test_auc'] > 0.9:
                f.write(f"Ready for deployment\n\n")
            elif phase6_mean_auc >= 0.7:
                f.write(f"Requires calibration for new domains\n\n")
            else:
                f.write(f"Requires domain-specific fine-tuning\n\n")
        
        f.write("---\n\n")
        
        # Pipeline Overview
        f.write("## Analysis Pipeline Overview\n\n")
        f.write("This analysis follows a rigorous methodology:\n\n")
        f.write("1. **Phases 1-3:** Individual file analysis and aggregation\n")
        f.write("   - Analyze each experiment file separately\n")
        f.write("   - Aggregate by experiment type (citation vs correctness)\n")
        f.write("   - Meta-analysis across all experiments\n\n")
        
        f.write("2. **Phase 4:** Feature Selection via RFE\n")
        f.write("   - Test multiple classifiers with Recursive Feature Elimination\n")
        f.write("   - Identify optimal feature subset\n")
        f.write("   - Select best-performing classifier\n\n")
        
        f.write("3. **Phase 5:** Ensemble Comparison with Optimal Features\n")
        f.write("   - Compare all classifiers using features from Phase 4\n")
        f.write("   - Evaluate on held-out test set\n")
        f.write("   - Identify final winner for deployment\n\n")
        
        f.write("4. **Phase 6:** Cross-Domain Generalization\n")
        f.write("   - Test winner from Phase 5 on unseen CLDs\n")
        f.write("   - Leave-one-CLD-out cross-validation\n")
        f.write("   - Assess production deployment strategy\n\n")
        
        f.write("---\n\n")
        
        # Detailed Results for Each Phase
        f.write("## Phase 4: RFE Feature Selection\n\n")
        if 'phase4_report_content' in results:
            f.write(results['phase4_report_content'])
        else:
            f.write("No Phase 4 results found.\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("## Phase 5: Ensemble Comparison with Optimal Features\n\n")
        if 'phase5_report_content' in results:
            f.write(results['phase5_report_content'])
        else:
            f.write("No Phase 5 results found.\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("## Phase 6: Cross-Domain Generalization\n\n")
        if 'phase6_report_content' in results:
            f.write(results['phase6_report_content'])
        else:
            f.write("No Phase 6 results found.\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Cross-Phase Synthesis
        f.write("## Cross-Phase Synthesis\n\n")
        
        if all(k in results for k in ['phase4_rfe', 'phase5_ensemble', 'phase6_cross_cld']):
            phase4 = results['phase4_rfe']
            phase5 = results['phase5_ensemble']
            phase6_results = results['phase6_cross_cld']
            
            phase6_mean_auc = np.mean([r['auc'] for r in phase6_results])
            phase6_std_auc = np.std([r['auc'] for r in phase6_results])
            
            f.write("### Classifier Journey Through Phases\n\n")
            f.write("| Phase | Method | Best Classifier | AUC | Features Used |\n")
            f.write("|-------|--------|-----------------|-----|---------------|\n")
            f.write(f"| 4 | RFE with CV | {phase4['best_classifier']} | {phase4['best_auc']:.3f} | {', '.join(phase4['optimal_features'])} |\n")
            f.write(f"| 5 | Held-out test | {phase5['best_classifier']} | {phase5['best_test_auc']:.3f} | {', '.join(phase5['features_used'])} |\n")
            f.write(f"| 6 | Leave-one-CLD-out | {phase5['best_classifier']} | {phase6_mean_auc:.3f} ± {phase6_std_auc:.3f} | {', '.join(phase5['features_used'])} |\n")
            
            f.write("\n### Performance Consistency\n\n")
            
            # Check if same classifier won in all phases
            p4_clf = phase4['best_classifier']
            p5_clf = phase5['best_classifier']
            
            if p4_clf == p5_clf:
                f.write(f"✅ **Consistent winner:** {p5_clf} performed best in both Phase 4 (feature selection) and Phase 5 (final evaluation)\n\n")
            else:
                f.write(f"⚠️  **Different winners:** Phase 4 identified {p4_clf}, but Phase 5 selected {p5_clf}\n")
                f.write(f"   - This suggests feature-classifier interaction effects\n\n")
            
            # Performance degradation from Phase 5 to Phase 6
            degradation = phase5['best_test_auc'] - phase6_mean_auc
            
            f.write(f"**Generalization Gap:** {degradation:+.3f} (Phase 5 test → Phase 6 cross-CLD)\n\n")
            
            if abs(degradation) < 0.05:
                f.write("✅ Minimal generalization gap - model transfers well across domains\n\n")
            elif degradation > 0:
                f.write("⚠️  Performance drops on unseen CLDs - domain-specific patterns exist\n\n")
            else:
                f.write("✅ Performance improves on cross-CLD! This suggests the held-out test set was challenging.\n\n")
            
            f.write("### Feature Importance Stability\n\n")
            
            if phase4['optimal_features'] == phase5['features_used']:
                f.write(f"✅ **Stable feature set:** Phase 4 RFE features were used unchanged in Phases 5 & 6\n")
                f.write(f"   - Features: {', '.join(phase5['features_used'])}\n\n")
            else:
                f.write(f"⚠️  **Feature set changed:** \n")
                f.write(f"   - Phase 4 RFE: {', '.join(phase4['optimal_features'])}\n")
                f.write(f"   - Phase 5 used: {', '.join(phase5['features_used'])}\n\n")
        
        else:
            f.write("Incomplete phase results - cannot perform cross-phase synthesis.\n\n")
        
        f.write("---\n\n")
        
        # Production Deployment Guide
        f.write("## Production Deployment Recommendations\n\n")
        
        if 'phase5_ensemble' in results and 'phase6_cross_cld' in results:
            phase5 = results['phase5_ensemble']
            phase6_results = results['phase6_cross_cld']
            phase6_mean_auc = np.mean([r['auc'] for r in phase6_results])
            
            f.write("### Recommended Configuration\n\n")
            f.write(f"**Classifier:** {phase5['best_classifier']}\n\n")
            f.write(f"**Features:** {', '.join(phase5['features_used'])}\n\n")
            f.write(f"**Expected Performance:**\n")
            f.write(f"- In-distribution (same CLDs): AUC ≈ {phase5['best_test_auc']:.3f}\n")
            f.write(f"- Out-of-distribution (new CLDs): AUC ≈ {phase6_mean_auc:.3f}\n\n")
            
            f.write("### Deployment Strategy\n\n")
            
            if phase6_mean_auc >= 0.8:
                f.write("**Strategy: Direct Deployment**\n\n")
                f.write("- Strong cross-domain performance\n")
                f.write("- Can be applied to new CLDs without modification\n")
                f.write("- Monitor performance and recalibrate if necessary\n\n")
            elif phase6_mean_auc >= 0.7:
                f.write("**Strategy: Deploy with Calibration**\n\n")
                f.write("- Good but not excellent cross-domain performance\n")
                f.write("- Recommended: Platt scaling or isotonic regression\n")
                f.write("- Collect small labeled sample from new CLDs for calibration\n")
                f.write("- Monitor and adjust thresholds per domain if needed\n\n")
            else:
                f.write("**Strategy: Domain-Specific Fine-Tuning**\n\n")
                f.write("- Moderate cross-domain performance indicates domain shift\n")
                f.write("- Recommended: Collect labeled samples from new CLDs\n")
                f.write("- Fine-tune classifier on new domain data\n")
                f.write("- Maintain domain-specific models or use domain adaptation\n\n")
            
            f.write("### Implementation Notes\n\n")
            f.write("1. **Feature Computation:** Ensure all CI metrics are computed consistently\n")
            f.write("2. **Preprocessing:** Apply same scaling/normalization as training\n")
            f.write("3. **Threshold Selection:** Optimize for your specific precision/recall trade-off\n")
            f.write("4. **Monitoring:** Track performance metrics and retrain periodically\n\n")
        
        f.write("---\n\n")
        
        # Appendix
        f.write("## Appendix: Output Locations\n\n")
        
        if 'phase4_dir' in results:
            f.write(f"- **Phase 4 (RFE):** `{results['phase4_dir']}/`\n")
        if 'phase5_dir' in results:
            f.write(f"- **Phase 5 (Ensemble):** `{results['phase5_dir']}/`\n")
        if 'phase6_dir' in results:
            f.write(f"- **Phase 6 (Cross-CLD):** `{results['phase6_dir']}/`\n")
        if 'grand_dir' in results:
            f.write(f"- **Grand Aggregate (Phases 1-3):** `{results['grand_dir']}/`\n")
        
        f.write("\n---\n\n")
        f.write("## Methodology References\n\n")
        f.write("- **RFE:** Guyon, I., et al. (2002). Gene selection for cancer classification using support vector machines. Machine Learning, 46(1-3), 389-422.\n")
        f.write("- **Nested CV:** Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics, 7(1), 91.\n")
        f.write("- **Cross-Domain Generalization:** Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. CVPR 2011.\n\n")
    
    print(f"\n✅ Master report saved: {output_path}")


def main():
    print("\n" + "="*80)
    print("RQ2 MASTER REPORT GENERATOR (VERSION 2)")
    print("="*80 + "\n")
    
    analyses_dir, _unused_output = rq2_dirs()
    
    if not analyses_dir.exists():
        print("❌ Analysis directory not found!")
        return 1
    
    # Load all results
    results = load_latest_results(analyses_dir)
    
    # Generate master report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analyses_dir / f"rq2_master_report_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "RQ2_MASTER_REPORT_V2.md"
    generate_master_report(results, report_path)
    
    print("\n" + "="*80)
    print("✅ MASTER REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📄 Report: {report_path.name}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

