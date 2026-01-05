"""
RQ1 Master Pipeline

Orchestrates complete RQ1 analysis:
- Can LLM-as-a-judge detect hallucinations induced by the corruptor?
- Comparison of serial vs parallel judging strategies

This is the single entry point for complete RQ1 analysis.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import logging

# Import all RQ1 analysis modules
from rq1_data_loader import RQ1DataLoader
from rq1_classification_analysis import RQ1ClassificationAnalysis
from rq1_comparison_analysis import RQ1ComparisonAnalysis
from rq1_statistical_validation import RQ1StatisticalValidation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RQ1MasterPipeline:
    """
    Master pipeline for RQ1 analysis.
    
    Orchestrates:
    1. Data loading and preparation
    2. Classification performance analysis
    3. Serial vs parallel strategy comparison
    4. Statistical validation
    5. Comprehensive reporting
    """
    
    def __init__(self, experiment_id: str, output_base_dir: str = None):
        """
        Initialize RQ1 master pipeline.
        
        Args:
            experiment_id: Experiment ID to analyze
            output_base_dir: Base directory for outputs. If None, creates timestamped folder.
        """
        self.experiment_id = experiment_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_base_dir is None:
            # Create timestamped analysis folder
            base = Path(__file__).parent.parent / 'rq1_analyses'
            self.output_dir = base / f'rq1_analysis_{self.timestamp}_{experiment_id}'
        else:
            self.output_dir = Path(output_base_dir)
        
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures'
        
        # Create directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
        logger.info(f"RQ1 Master Pipeline initialized for experiment: {experiment_id}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_full_pipeline(
        self,
        combo_number: int = None,
        prompt_set: str = None,
        cld_name: str = None,
        run_number: str = None,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000
    ):
        """
        Run complete RQ1 analysis pipeline.
        
        Args:
            combo_number: Optional filter for parameter combo
            prompt_set: Optional filter for prompt set
            cld_name: Optional filter for CLD name
            run_number: Optional filter for run number
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutation samples
        """
        
        print("\n" + "="*80)
        print(" "*30 + "RQ1 MASTER PIPELINE")
        print("="*80)
        print(f"\nExperiment ID: {self.experiment_id}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Output directory: {self.output_dir}")
        print("\n")
        
        # Step 1: Load and prepare data
        print("="*80)
        print("STEP 1/5: DATA LOADING AND PREPARATION")
        print("="*80)
        
        loader = RQ1DataLoader()
        
        try:
            print(f"\n[1.1] Loading experiment data...")
            df_raw = loader.load_experiment_data(
                experiment_id=self.experiment_id,
                combo_number=combo_number,
                prompt_set=prompt_set,
                cld_name=cld_name,
                run_number=run_number
            )
            print(f"✓ Loaded {len(df_raw)} edges")
            
            print(f"\n[1.2] Preparing data for RQ1 analysis...")
            df_prepared = loader.prepare_rq1_data(df_raw)
            
            # Export prepared data
            loader.export_prepared_data(df_prepared, str(self.tables_dir), self.experiment_id)
            
            # Store data summary
            halluc_rate = df_prepared['is_hallucination'].mean() * 100
            print(f"\n✓ Data Summary:")
            print(f"  • Total edges: {len(df_prepared)}")
            print(f"  • Hallucinations: {df_prepared['is_hallucination'].sum()} ({halluc_rate:.1f}%)")
            print(f"  • Genuine edges: {(~df_prepared['is_hallucination']).sum()} ({100-halluc_rate:.1f}%)")
            
            self.results['data'] = {
                'n_edges': len(df_prepared),
                'n_hallucinations': int(df_prepared['is_hallucination'].sum()),
                'n_genuine': int((~df_prepared['is_hallucination']).sum()),
                'hallucination_rate': float(halluc_rate)
            }
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        # Step 2: Classification analysis
        print("\n" + "="*80)
        print("STEP 2/5: CLASSIFICATION ANALYSIS")
        print("="*80)
        
        classifier = RQ1ClassificationAnalysis(df_prepared)
        
        print("\n[2.1] Computing confusion matrices...")
        conf_matrix = classifier.compute_confusion_matrix()
        print("\nConfusion Matrix:")
        print(conf_matrix.to_string(index=False))
        
        print("\n[2.2] Computing classification metrics...")
        metrics = classifier.compute_classification_metrics()
        print("\nClassification Metrics:")
        print(metrics[['group', 'precision', 'recall', 'f1', 'accuracy']].to_string(index=False))
        
        # Check if we have corruption rate data
        if 'corruption_rate' in df_prepared.columns:
            print("\n[2.3] Computing metrics by corruption rate...")
            metrics_by_rate = classifier.compute_metrics_by_corruption_rate()
            if not metrics_by_rate.empty:
                print("\nMetrics by Corruption Rate:")
                print(metrics_by_rate[['group', 'precision', 'recall', 'f1', 'accuracy']].to_string(index=False))
        
        print("\n[2.4] Generating classification visualizations...")
        classifier.visualize_confusion_matrix(str(self.figures_dir / 'rq1_confusion_matrix.png'))
        classifier.visualize_metrics_comparison(str(self.figures_dir / 'rq1_metrics_comparison.png'))
        if 'corruption_rate' in df_prepared.columns:
            classifier.visualize_performance_by_corruption_rate(
                str(self.figures_dir / 'rq1_performance_by_corruption_rate.png')
            )
        
        # Save results
        conf_matrix.to_csv(self.tables_dir / 'rq1_confusion_matrix.csv', index=False)
        metrics.to_csv(self.tables_dir / 'rq1_metrics.csv', index=False)
        
        self.results['classification'] = {
            'confusion_matrix': conf_matrix.to_dict('records'),
            'metrics': metrics.to_dict('records')
        }
        
        # Step 3: Strategy comparison (serial vs parallel)
        print("\n" + "="*80)
        print("STEP 3/5: STRATEGY COMPARISON (Serial vs Parallel)")
        print("="*80)
        
        if 'judging_strategy' in df_prepared.columns or 'num_judges' in df_prepared.columns:
            comparator = RQ1ComparisonAnalysis(df_prepared)
            
            print("\n[3.1] Comparing judging strategies...")
            strategy_comp = comparator.compare_strategies()
            
            if not strategy_comp.empty:
                print("\nStrategy Comparison:")
                print(strategy_comp[['strategy', 'n_edges', 'precision', 'recall', 'f1', 'accuracy']].to_string(index=False))
                
                print("\n[3.2] Statistical comparison of strategies...")
                stat_comp = comparator.statistical_comparison()
                if not stat_comp.empty:
                    print("\nStatistical Comparison:")
                    print(stat_comp[['strategy_1', 'strategy_2', 'diff', 'p_value', 'significant']].to_string(index=False))
                
                print("\n[3.3] Generating strategy comparison visualizations...")
                comparator.visualize_strategy_comparison(str(self.figures_dir / 'rq1_strategy_comparison.png'))
                comparator.visualize_confusion_matrices_comparison(
                    str(self.figures_dir / 'rq1_confusion_matrices_by_strategy.png')
                )
                
                # Save results
                strategy_comp.to_csv(self.tables_dir / 'rq1_strategy_comparison.csv', index=False)
                if not stat_comp.empty:
                    stat_comp.to_csv(self.tables_dir / 'rq1_statistical_comparison.csv', index=False)
                
                self.results['strategy_comparison'] = {
                    'comparison': strategy_comp.to_dict('records'),
                    'statistical_tests': stat_comp.to_dict('records') if not stat_comp.empty else []
                }
            else:
                print("\n⚠️ No multiple strategies found, skipping comparison")
        else:
            print("\n⚠️ No judging strategy information available, skipping comparison")
        
        # Step 4: Statistical validation
        print("\n" + "="*80)
        print("STEP 4/5: STATISTICAL VALIDATION")
        print("="*80)
        
        validator = RQ1StatisticalValidation(df_prepared, n_bootstrap=n_bootstrap, n_permutations=n_permutations)
        
        print(f"\n[4.1] Computing bootstrap confidence intervals (n={n_bootstrap})...")
        bootstrap_ci = validator.bootstrap_confidence_intervals()
        print("\nBootstrap 95% Confidence Intervals:")
        print(bootstrap_ci[['group', 'precision_observed', 'precision_ci_lower', 'precision_ci_upper',
                           'recall_observed', 'recall_ci_lower', 'recall_ci_upper',
                           'f1_observed', 'f1_ci_lower', 'f1_ci_upper']].to_string(index=False))
        
        print(f"\n[4.2] Running permutation tests (n={n_permutations})...")
        perm_tests = validator.permutation_test()
        print("\nPermutation Test Results:")
        print(perm_tests[['group', 'observed_accuracy', 'perm_accuracy_mean', 'p_value_accuracy', 'significant_accuracy',
                         'observed_f1', 'perm_f1_mean', 'p_value_f1', 'significant_f1']].to_string(index=False))
        
        print("\n[4.3] Generating statistical validation visualizations...")
        validator.visualize_bootstrap_distributions(str(self.figures_dir / 'rq1_bootstrap_distributions.png'))
        validator.visualize_permutation_distributions(str(self.figures_dir / 'rq1_permutation_distributions.png'))
        
        # Save results
        bootstrap_ci.to_csv(self.tables_dir / 'rq1_bootstrap_ci.csv', index=False)
        perm_tests.to_csv(self.tables_dir / 'rq1_permutation_tests.csv', index=False)
        
        self.results['statistical_validation'] = {
            'bootstrap_ci': bootstrap_ci.to_dict('records'),
            'permutation_tests': perm_tests.to_dict('records')
        }
        
        # Step 5: Generate summary report
        print("\n" + "="*80)
        print("STEP 5/5: GENERATING SUMMARY REPORT")
        print("="*80)
        
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("✅ RQ1 MASTER PIPELINE COMPLETE!")
        print("="*80)
        
        print(f"\n📁 Outputs saved to: {self.output_dir}")
        print(f"  • Tables: {self.tables_dir}")
        print(f"  • Figures: {self.figures_dir}")
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        
        # Extract key findings
        findings = self._extract_key_findings()
        
        # Create summary dictionary
        summary = {
            'metadata': {
                'experiment_id': self.experiment_id,
                'timestamp': self.timestamp,
                'output_directory': str(self.output_dir)
            },
            'research_question': "RQ1: Can LLM-as-a-judge detect hallucinations induced by the corruptor?",
            'data_summary': self.results['data'],
            'key_findings': findings,
            'detailed_results': self.results
        }
        
        # Save as JSON
        json_path = self.tables_dir / f'rq1_complete_summary_{self.timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved JSON summary: {json_path.name}")
        
        # Generate markdown report
        md_report = self._generate_markdown_report(summary)
        md_path = self.tables_dir / f'rq1_summary_report_{self.timestamp}.md'
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        print(f"✓ Saved Markdown report: {md_path.name}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("📊 RQ1 ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n🎯 ANSWER TO RQ1: {findings['answer']}")
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   • Precision: {findings['precision']:.3f} (95% CI: [{findings['precision_ci'][0]:.3f}, {findings['precision_ci'][1]:.3f}])")
        print(f"   • Recall: {findings['recall']:.3f} (95% CI: [{findings['recall_ci'][0]:.3f}, {findings['recall_ci'][1]:.3f}])")
        print(f"   • F1 Score: {findings['f1']:.3f} (95% CI: [{findings['f1_ci'][0]:.3f}, {findings['f1_ci'][1]:.3f}])")
        print(f"   • Accuracy: {findings['accuracy']:.3f}")
        
        if findings.get('statistical_significance'):
            print(f"\n   ✅ Performance is statistically significant (p < 0.05)")
        else:
            print(f"\n   ⚠️ Performance is NOT statistically significant")
        
        if 'strategy_comparison' in findings:
            print(f"\n📊 STRATEGY COMPARISON:")
            print(f"   • {findings['strategy_comparison']}")
        
        print(f"\n📊 DATA:")
        print(f"   • Total edges: {self.results['data']['n_edges']}")
        print(f"   • Hallucination rate: {self.results['data']['hallucination_rate']:.1f}%")
        
        print("\n" + "="*80)
    
    def _extract_key_findings(self):
        """Extract key findings for summary."""
        
        # Get overall metrics
        metrics = self.results['classification']['metrics'][0]  # First row is overall
        
        # Get bootstrap CIs
        bootstrap = self.results['statistical_validation']['bootstrap_ci'][0]
        
        # Get permutation test results
        perm = self.results['statistical_validation']['permutation_tests'][0]
        
        # Determine statistical significance
        significant = perm['significant_accuracy'] and perm['significant_f1']
        
        # Determine answer to RQ1
        if metrics['f1'] >= 0.7 and significant:
            answer = "YES - LLM-as-a-judge can effectively detect hallucinations with high accuracy"
        elif metrics['f1'] >= 0.5 and significant:
            answer = "PARTIAL - LLM-as-a-judge shows moderate hallucination detection ability"
        elif metrics['f1'] >= 0.5:
            answer = "WEAK - LLM-as-a-judge shows detection ability but lacks statistical significance"
        else:
            answer = "NO - LLM-as-a-judge cannot reliably detect hallucinations"
        
        findings = {
            'answer': answer,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'precision_ci': [bootstrap['precision_ci_lower'], bootstrap['precision_ci_upper']],
            'recall_ci': [bootstrap['recall_ci_lower'], bootstrap['recall_ci_upper']],
            'f1_ci': [bootstrap['f1_ci_lower'], bootstrap['f1_ci_upper']],
            'statistical_significance': significant,
            'p_value_f1': perm['p_value_f1']
        }
        
        # Add strategy comparison if available
        if 'strategy_comparison' in self.results:
            comp = self.results['strategy_comparison']['comparison']
            if len(comp) > 1:
                # Find best strategy
                best_idx = max(range(len(comp)), key=lambda i: comp[i]['f1'])
                best_strategy = comp[best_idx]['strategy']
                best_f1 = comp[best_idx]['f1']
                findings['strategy_comparison'] = f"Best strategy: {best_strategy} (F1={best_f1:.3f})"
        
        return findings
    
    def _generate_markdown_report(self, summary):
        """Generate markdown report."""
        
        findings = summary['key_findings']
        data = summary['data_summary']
        
        md = f"""# RQ1 Analysis Summary Report

**Generated:** {summary['metadata']['timestamp']}  
**Experiment ID:** `{summary['metadata']['experiment_id']}`  
**Research Question:** {summary['research_question']}

---

## 🎯 Answer to RQ1

**{findings['answer']}**

---

## 📊 Key Findings

### Overall Judge Performance

- **Precision:** {findings['precision']:.3f} (95% CI: [{findings['precision_ci'][0]:.3f}, {findings['precision_ci'][1]:.3f}])
- **Recall:** {findings['recall']:.3f} (95% CI: [{findings['recall_ci'][0]:.3f}, {findings['recall_ci'][1]:.3f}])
- **F1 Score:** {findings['f1']:.3f} (95% CI: [{findings['f1_ci'][0]:.3f}, {findings['f1_ci'][1]:.3f}])
- **Accuracy:** {findings['accuracy']:.3f}
- **Statistical Significance:** {'✅ Yes' if findings['statistical_significance'] else '❌ No'} (p={findings['p_value_f1']:.4f})

"""
        
        if 'strategy_comparison' in findings:
            md += f"\n### Strategy Comparison\n\n{findings['strategy_comparison']}\n"
        
        md += f"""
---

## 📈 Data Summary

- **Total Edges:** {data['n_edges']}
- **Hallucinations:** {data['n_hallucinations']} ({data['hallucination_rate']:.1f}%)
- **Genuine Edges:** {data['n_genuine']} ({100-data['hallucination_rate']:.1f}%)

---

## 📁 Output Files

### Tables

1. `rq1_prepared_data_{self.experiment_id}.csv` - Prepared edge data
2. `rq1_confusion_matrix.csv` - Confusion matrix counts
3. `rq1_metrics.csv` - Classification metrics
4. `rq1_bootstrap_ci.csv` - Bootstrap confidence intervals
5. `rq1_permutation_tests.csv` - Permutation test results
6. `rq1_strategy_comparison.csv` - Strategy comparison (if applicable)

### Figures

1. `rq1_confusion_matrix.png` - Confusion matrix visualization
2. `rq1_metrics_comparison.png` - Metrics comparison
3. `rq1_performance_by_corruption_rate.png` - Performance vs corruption rate (if applicable)
4. `rq1_bootstrap_distributions.png` - Bootstrap distributions
5. `rq1_permutation_distributions.png` - Permutation test null distributions
6. `rq1_strategy_comparison.png` - Strategy comparison (if applicable)

---

*Report generated by RQ1 Master Pipeline*
"""
        
        return md


def main():
    """Run RQ1 master pipeline."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RQ1 analysis pipeline')
    parser.add_argument('experiment_id', type=str, help='Experiment ID to analyze')
    parser.add_argument('--combo', type=int, help='Filter by parameter combo number')
    parser.add_argument('--prompt', type=str, help='Filter by prompt set name')
    parser.add_argument('--cld', type=str, help='Filter by CLD name')
    parser.add_argument('--run', type=str, help='Filter by run number')
    parser.add_argument('--bootstrap', type=int, default=10000, help='Number of bootstrap samples')
    parser.add_argument('--permutations', type=int, default=10000, help='Number of permutation samples')
    parser.add_argument('--output', type=str, help='Output directory (optional)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = RQ1MasterPipeline(
        experiment_id=args.experiment_id,
        output_base_dir=args.output
    )
    
    # Run analysis
    results = pipeline.run_full_pipeline(
        combo_number=args.combo,
        prompt_set=args.prompt,
        cld_name=args.cld,
        run_number=args.run,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations
    )
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
