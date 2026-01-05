"""
RQ2 Master Pipeline

Orchestrates all RQ2 analyses and generates a comprehensive summary report.
This is the single entry point for complete RQ2 analysis.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys

# Import all analysis modules
from rq2_data_loader import RQ2DataLoader
from rq2_correlation_analysis import RQ2CorrelationAnalysis
from rq2_classification_analysis import RQ2ClassificationAnalysis
from rq2_statistical_validation import RQ2StatisticalValidation


class RQ2MasterPipeline:
    """
    Master pipeline that orchestrates all RQ2 analyses.
    
    Runs:
    1. Data loading and validation
    2. Correlation analysis
    3. Classification analysis (ROC/thresholds)
    4. Statistical validation (permutation/bootstrap)
    5. Generate comprehensive summary report
    """
    
    def __init__(self, output_base_dir: str = None):
        """
        Initialize master pipeline.
        
        Args:
            output_base_dir: Base directory for all outputs. If None, uses default.
        """
        if output_base_dir is None:
            self.output_base_dir = Path(__file__).parent.parent
        else:
            self.output_base_dir = Path(output_base_dir)
        
        self.tables_dir = self.output_base_dir / 'tables'
        self.figures_dir = self.output_base_dir / 'figures'
        self.results_dir = self.output_base_dir / 'results'
        self.configs_dir = self.output_base_dir / 'configs'
        
        # Create directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def extract_data_provenance(self, df: pd.DataFrame) -> dict:
        """
        Extract comprehensive metadata about the data sources and experimental configuration.
        
        Args:
            df: DataFrame with loaded experiment data
            
        Returns:
            Dictionary with data provenance information
        """
        provenance = {}
        
        # Extract experiment IDs
        experiment_ids = df['experiment_id'].unique().tolist()
        provenance['experiment_ids'] = experiment_ids
        provenance['n_experiments'] = len(experiment_ids)
        
        # Extract CLD information
        clds = df['cld_name'].unique().tolist()
        provenance['clds'] = clds
        provenance['n_clds'] = len(clds)
        
        # Count runs and samples per experiment and CLD
        runs_info = []
        for exp_id in experiment_ids:
            exp_df = df[df['experiment_id'] == exp_id]
            for cld in exp_df['cld_name'].unique():
                cld_df = exp_df[exp_df['cld_name'] == cld]
                n_edges = len(cld_df)
                n_tp = len(cld_df[cld_df['classification'] == 'TP'])
                n_fp = len(cld_df[cld_df['classification'] == 'FP'])
                n_fn = len(cld_df[cld_df['classification'] == 'FN'])
                n_tn = len(cld_df[cld_df['classification'] == 'TN'])
                
                runs_info.append({
                    'experiment_id': exp_id,
                    'cld_name': cld,
                    'n_edges': n_edges,
                    'n_tp': n_tp,
                    'n_fp': n_fp,
                    'n_fn': n_fn,
                    'n_tn': n_tn
                })
        
        provenance['runs_info'] = runs_info
        provenance['total_edges'] = len(df)
        
        # Get actual parameters from param_combo_mapping for precise matching
        experiment_params = {}
        mapping_info = []
        for exp_id in experiment_ids:
            mapping_file = self.results_dir / f"param_combo_mapping_{exp_id}.csv"
            if mapping_file.exists():
                try:
                    mapping_df = pd.read_csv(mapping_file)
                    mapping_info.append({
                        'experiment_id': exp_id,
                        'n_runs_in_mapping': len(mapping_df),
                        'prompts': mapping_df['prompt'].unique().tolist(),
                        'clds': mapping_df['cld_prefix'].unique().tolist(),
                        'n_combos': mapping_df['combo_number'].nunique()
                    })
                    
                    # Extract actual parameters used
                    if not mapping_df.empty and 'parameters' in mapping_df.columns:
                        params_json = mapping_df.iloc[0]['parameters']
                        params = json.loads(params_json)
                        experiment_params[exp_id] = params
                except Exception as e:
                    print(f"Warning: Could not load mapping for {exp_id}: {e}")
        
        provenance['mapping_info'] = mapping_info
        
        # Find matching config YAML files (primary source of truth)
        # Match based on both CLD and actual parameters
        yaml_configs = []
        if self.configs_dir.exists():
            for config_file in self.configs_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config_yaml = yaml.safe_load(f)
                    
                    # Check if excel_files match our CLDs
                    yaml_excel_files = config_yaml.get('excel_files', [])
                    param_grid = config_yaml.get('param_grid', {})
                    
                    if yaml_excel_files:
                        for cld in clds:
                            if any(cld in ef for ef in yaml_excel_files):
                                # Additional check: match parameters if we have them
                                matches_params = True
                                for exp_id, params in experiment_params.items():
                                    # Check key parameters match
                                    if 'generator_config' in params and 'generator_config' in param_grid:
                                        gen_model = params['generator_config'].get('model')
                                        yaml_gen_models = [g.get('model') if isinstance(g, dict) else g 
                                                          for g in param_grid['generator_config']]
                                        if gen_model and gen_model not in yaml_gen_models:
                                            matches_params = False
                                            break
                                    
                                    if 'judge_config' in params and 'judge_config' in param_grid:
                                        judge_model = params['judge_config'].get('model')
                                        yaml_judge_models = [j.get('model') if isinstance(j, dict) else j 
                                                            for j in param_grid['judge_config']]
                                        if judge_model and judge_model not in yaml_judge_models:
                                            matches_params = False
                                            break
                                    
                                    if 'corruption_rate' in params and 'corruption_rate' in param_grid:
                                        if params['corruption_rate'] not in param_grid['corruption_rate']:
                                            matches_params = False
                                            break
                                    
                                    if 'embedding_enable' in params and 'embedding_enable' in param_grid:
                                        if params['embedding_enable'] not in param_grid['embedding_enable']:
                                            matches_params = False
                                            break
                                
                                if matches_params:
                                    yaml_configs.append({
                                        'file': config_file.name,
                                        'path': str(config_file),
                                        'param_grid': param_grid,
                                        'runs': config_yaml.get('runs', 1),
                                        'excel_files': yaml_excel_files,
                                        'prompt_files': config_yaml.get('prompt_files', []),
                                        'full_config': config_yaml
                                    })
                                    break
                except Exception as e:
                    pass
        
        provenance['config_yamls'] = yaml_configs
        
        # Determine root directory for results
        if experiment_ids:
            exp_dirs = [self.results_dir / exp_id for exp_id in experiment_ids]
            provenance['result_directories'] = [str(d) for d in exp_dirs if d.exists()]
        
        return provenance
        
    def run_full_pipeline(self, n_permutations: int = 10000, n_bootstrap: int = 10000):
        """
        Run the complete RQ2 analysis pipeline.
        
        Args:
            n_permutations: Number of permutations for permutation tests
            n_bootstrap: Number of bootstrap samples for CI estimation
        """
        
        print("\n" + "="*80)
        print(" "*25 + "RQ2 MASTER PIPELINE")
        print("="*80)
        print(f"\nTimestamp: {self.timestamp}")
        print(f"Output directory: {self.output_base_dir}")
        print("\n")
        
        # Step 1: Load data
        print("="*80)
        print("STEP 1/4: DATA LOADING")
        print("="*80)
        
        data_file = self.tables_dir / 'rq2_standalone_data.csv'
        if not data_file.exists():
            print(f"\n❌ ERROR: Data file not found: {data_file}")
            print("Please run RQ2 data loader first to generate the data.")
            sys.exit(1)
        
        df = pd.read_csv(data_file)
        print(f"\n✓ Loaded {len(df)} edges from: {data_file.name}")
        
        # Extract data provenance
        print("\n[1.1] Extracting data provenance...")
        provenance = self.extract_data_provenance(df)
        
        print(f"\n📊 Data Provenance:")
        print(f"  • Experiments: {provenance['n_experiments']}")
        for exp_id in provenance['experiment_ids']:
            print(f"    - {exp_id}")
        print(f"  • CLDs: {provenance['n_clds']}")
        for cld in provenance['clds']:
            print(f"    - {cld}")
        if provenance['config_yamls']:
            print(f"  • Configuration YAML(s):")
            for cfg in provenance['config_yamls']:
                print(f"    - File: {cfg['file']}")
                print(f"      Runs per combo: {cfg['runs']}")
                param_grid = cfg['param_grid']
                if 'generator_config' in param_grid:
                    gen_models = [g.get('model', 'unknown') if isinstance(g, dict) else str(g) for g in param_grid['generator_config']]
                    print(f"      Generator models: {gen_models}")
                if 'judge_config' in param_grid:
                    judge_models = [j.get('model', 'unknown') if isinstance(j, dict) else str(j) for j in param_grid['judge_config']]
                    print(f"      Judge models: {judge_models}")
                if 'corruption_rate' in param_grid:
                    print(f"      Corruption rates: {param_grid['corruption_rate']}")
        if provenance['mapping_info']:
            for info in provenance['mapping_info']:
                print(f"  • Experiment {info['experiment_id']}: {info['n_runs_in_mapping']} runs, {info['n_combos']} parameter combos")
        
        # Basic data stats
        halluc_rate = df['is_hallucination'].mean() * 100
        print(f"\n  • Hallucination rate: {halluc_rate:.1f}%")
        print(f"  • Non-hallucinations: {(~df['is_hallucination']).sum()}")
        print(f"  • Hallucinations: {df['is_hallucination'].sum()}")
        
        self.results['data'] = {
            'n_edges': len(df),
            'n_hallucinations': int(df['is_hallucination'].sum()),
            'n_non_hallucinations': int((~df['is_hallucination']).sum()),
            'hallucination_rate': float(halluc_rate)
        }
        self.results['provenance'] = provenance
        
        # Step 2: Correlation Analysis
        print("\n" + "="*80)
        print("STEP 2/4: CORRELATION ANALYSIS")
        print("="*80)
        
        corr_analyzer = RQ2CorrelationAnalysis(df)
        
        # Compute correlations
        print("\n[2.1] Computing correlations...")
        corr_results = corr_analyzer.compute_correlations()
        print("\nCorrelation Results:")
        print(corr_results[['metric', 'pearson_r', 'pearson_p', 'significant']].to_string(index=False))
        
        # Group comparison
        print("\n[2.2] Computing group comparisons...")
        group_results = corr_analyzer.group_comparison()
        print("\nGroup Comparison (t-tests):")
        print(group_results[['metric', 'halluc_mean', 'non_halluc_mean', 
                            't_p_value', 'significant']].to_string(index=False))
        
        # Generate visualizations
        print("\n[2.3] Generating visualizations...")
        corr_analyzer.visualize_distributions(str(self.figures_dir))
        corr_analyzer.visualize_scatter_plots(str(self.figures_dir))
        
        # Save results
        corr_results.to_csv(self.tables_dir / 'rq2_correlations.csv', index=False)
        group_results.to_csv(self.tables_dir / 'rq2_group_comparison.csv', index=False)
        
        self.results['correlation'] = {
            'correlations': corr_results.to_dict('records'),
            'group_comparison': group_results.to_dict('records')
        }
        
        # Step 3: Classification Analysis
        print("\n" + "="*80)
        print("STEP 3/4: CLASSIFICATION ANALYSIS")
        print("="*80)
        
        class_analyzer = RQ2ClassificationAnalysis(df)
        
        # Compute ROC curves
        print("\n[3.1] Computing ROC curves...")
        roc_results = class_analyzer.compute_roc_curves()
        
        print("\nROC AUC Scores:")
        for metric, results in roc_results.items():
            print(f"  {metric:20s}: AUC = {results['auc']:.3f}")
        
        # Find optimal thresholds
        print("\n[3.2] Finding optimal thresholds (F1 criterion)...")
        optimal = class_analyzer.find_optimal_thresholds(criterion='f1')
        print("\nOptimal Thresholds:")
        print(optimal[['metric', 'optimal_threshold', 'f1', 'precision', 'recall']].to_string(index=False))
        
        # Generate visualizations
        print("\n[3.3] Generating visualizations...")
        class_analyzer.visualize_roc_curves(str(self.figures_dir))
        class_analyzer.visualize_confusion_matrices(str(self.figures_dir))
        
        # Save results
        roc_summary = pd.DataFrame([
            {'metric': metric, 'roc_auc': results['auc']}
            for metric, results in roc_results.items()
        ])
        roc_summary.to_csv(self.tables_dir / 'rq2_roc_auc.csv', index=False)
        optimal.to_csv(self.tables_dir / 'rq2_optimal_thresholds.csv', index=False)
        
        # Find best metric
        best_metric = roc_summary.loc[roc_summary['roc_auc'].idxmax()]
        
        self.results['classification'] = {
            'roc_auc_scores': roc_summary.to_dict('records'),
            'optimal_thresholds': optimal.to_dict('records'),
            'best_metric': best_metric['metric'],
            'best_auc': float(best_metric['roc_auc'])
        }
        
        # Step 4: Statistical Validation
        print("\n" + "="*80)
        print("STEP 4/4: STATISTICAL VALIDATION")
        print("="*80)
        
        validator = RQ2StatisticalValidation(df, n_permutations=n_permutations, 
                                            n_bootstrap=n_bootstrap)
        
        # Permutation tests
        print("\n[4.1] Running permutation tests...")
        perm_results = validator.run_all_permutation_tests()
        print("\nPermutation Test Results (uncorrected):")
        print(perm_results[['metric', 'observed_r', 'permutation_p_value', 'significant']].to_string(index=False))
        
        # Multiple comparison correction
        print("\n[4.2] Applying Holm-Bonferroni correction...")
        perm_results_corrected = validator.correct_permutation_pvalues(method='holm')
        print("\nCorrected p-values:")
        print(perm_results_corrected[['metric', 'permutation_p_value', 
                                      'corrected_p_value', 'significant_corrected']].to_string(index=False))
        
        # Bootstrap CI
        print("\n[4.3] Computing bootstrap confidence intervals...")
        boot_results = validator.run_all_bootstrap_ci()
        print("\nBootstrap 95% CI for AUC:")
        print(boot_results[['metric', 'observed_auc', 'ci_lower', 'ci_upper']].to_string(index=False))
        
        # Generate visualizations
        print("\n[4.4] Generating visualizations...")
        validator.visualize_permutation_distributions(str(self.figures_dir))
        validator.visualize_bootstrap_distributions(str(self.figures_dir))
        
        # Save results
        perm_results_corrected.to_csv(self.tables_dir / 'rq2_permutation_tests.csv', index=False)
        boot_results.to_csv(self.tables_dir / 'rq2_bootstrap_ci.csv', index=False)
        
        self.results['statistical_validation'] = {
            'permutation_tests': perm_results_corrected.to_dict('records'),
            'bootstrap_ci': boot_results.to_dict('records'),
            'n_significant_uncorrected': int(perm_results['significant'].sum()),
            'n_significant_corrected': int(perm_results_corrected['significant_corrected'].sum())
        }
        
        # Generate summary report
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("✅ RQ2 MASTER PIPELINE COMPLETE!")
        print("="*80)
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report in multiple formats."""
        
        # Create summary dictionary
        summary = {
            'metadata': {
                'timestamp': self.timestamp,
                'output_directory': str(self.output_base_dir),
                'analysis_version': '1.0'
            },
            'research_question': "RQ2: Can context-insensitive (CI) metrics predict hallucinations?",
            'data_summary': self.results['data'],
            'key_findings': self._extract_key_findings(),
            'detailed_results': self.results
        }
        
        # Save as JSON
        json_path = self.tables_dir / f'rq2_complete_summary_{self.timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved JSON summary: {json_path.name}")
        
        # Generate markdown report
        md_report = self._generate_markdown_report(summary)
        md_path = self.tables_dir / f'rq2_summary_report_{self.timestamp}.md'
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        print(f"✓ Saved Markdown report: {md_path.name}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("📊 RQ2 ANALYSIS SUMMARY")
        print("="*80)
        
        findings = summary['key_findings']
        
        print(f"\n🎯 ANSWER TO RQ2: {findings['answer']}")
        print(f"\n📈 BEST METRIC: {findings['best_metric']}")
        print(f"   • AUC: {findings['best_auc']:.3f} (95% CI: [{findings['best_ci_lower']:.3f}, {findings['best_ci_upper']:.3f}])")
        print(f"   • Optimal threshold: {findings['best_threshold']:.3f}")
        print(f"   • F1 score at threshold: {findings['best_f1']:.3f}")
        print(f"   • Statistical significance: {'Yes (p=' + str(findings['best_p_value']) + ')' if findings['best_significant'] else 'No'}")
        
        print(f"\n📊 DATA:")
        print(f"   • Total edges: {self.results['data']['n_edges']}")
        print(f"   • Hallucination rate: {self.results['data']['hallucination_rate']:.1f}%")
        
        print(f"\n📁 OUTPUTS GENERATED:")
        print(f"   • 9 CSV tables in: {self.tables_dir}")
        print(f"   • 6 PNG figures (300 dpi) in: {self.figures_dir}")
        
        print("\n" + "="*80)
        
    def _extract_key_findings(self):
        """Extract key findings for summary."""
        
        # Get best metric info
        best_metric = self.results['classification']['best_metric']
        best_auc = self.results['classification']['best_auc']
        
        # Find corresponding bootstrap CI
        boot_results = self.results['statistical_validation']['bootstrap_ci']
        best_boot = next((r for r in boot_results if r['metric'] == best_metric), None)
        
        # Find corresponding optimal threshold
        opt_thresholds = self.results['classification']['optimal_thresholds']
        best_opt = next((r for r in opt_thresholds if r['metric'] == best_metric), None)
        
        # Find corresponding permutation test
        perm_results = self.results['statistical_validation']['permutation_tests']
        best_perm = next((r for r in perm_results if r['metric'] == best_metric), None)
        
        # Determine answer to RQ2
        if best_auc > 0.7 and best_boot and best_boot['ci_lower'] > 0.5:
            if best_perm and best_perm['significant_corrected']:
                answer = "YES - CI metrics can predict hallucinations with good accuracy and statistical significance"
            else:
                answer = "PARTIAL - CI metrics show predictive ability but lack statistical significance after correction"
        elif best_auc > 0.6:
            answer = "WEAK - CI metrics show modest predictive ability"
        else:
            answer = "NO - CI metrics cannot reliably predict hallucinations"
        
        return {
            'answer': answer,
            'best_metric': best_metric,
            'best_auc': best_auc,
            'best_ci_lower': best_boot['ci_lower'] if best_boot else None,
            'best_ci_upper': best_boot['ci_upper'] if best_boot else None,
            'best_threshold': best_opt['optimal_threshold'] if best_opt else None,
            'best_f1': best_opt['f1'] if best_opt else None,
            'best_significant': best_perm['significant_corrected'] if best_perm else False,
            'best_p_value': best_perm['corrected_p_value'] if best_perm else None
        }
    
    def _generate_markdown_report(self, summary):
        """Generate markdown format report."""
        
        findings = summary['key_findings']
        data = summary['data_summary']
        provenance = summary['detailed_results'].get('provenance', {})
        
        md = f"""# RQ2 Analysis Summary Report

**Generated:** {summary['metadata']['timestamp']}  
**Research Question:** {summary['research_question']}

---

## 📂 Data Provenance

### Experiments Analyzed

"""
        
        # Add experiment IDs
        for exp_id in provenance.get('experiment_ids', []):
            md += f"- `{exp_id}`\n"
        
        # Add ground truth CLDs
        md += "\n### Ground Truth CLD(s)\n\n"
        for cld in provenance.get('clds', []):
            md += f"- **{cld}**\n"
        
        # Add dataset breakdown
        if provenance.get('runs_info'):
            md += "\n### Dataset Breakdown\n\n"
            for run_info in provenance['runs_info']:
                md += f"**Experiment:** `{run_info['experiment_id']}`  \n"
                md += f"**CLD:** {run_info['cld_name']}  \n"
                md += f"- Total Edges: {run_info['n_edges']}\n"
                md += f"  - True Positives (TP): {run_info['n_tp']}\n"
                md += f"  - False Positives (FP): {run_info['n_fp']}\n"
                md += f"  - False Negatives (FN): {run_info['n_fn']}\n"
                md += f"  - True Negatives (TN): {run_info['n_tn']}\n\n"
        
        # Add configuration from YAML
        if provenance.get('config_yamls'):
            md += "### Configuration (from YAML)\n\n"
            for yaml_cfg in provenance['config_yamls']:
                md += f"**Config File:** `{yaml_cfg['file']}`  \n"
                md += f"**Path:** `{yaml_cfg['path']}`\n\n"
                md += f"- Runs per combo: **{yaml_cfg['runs']}**\n"
                md += f"- CLDs: {', '.join(yaml_cfg['excel_files'])}\n"
                if yaml_cfg.get('prompt_files'):
                    md += f"- Prompts: {', '.join(yaml_cfg['prompt_files'])}\n"
                md += "\n**Parameter Grid:**\n"
                param_grid = yaml_cfg['param_grid']
                if 'generator_config' in param_grid:
                    gen_models = [g.get('model', str(g)) if isinstance(g, dict) else str(g) for g in param_grid['generator_config']]
                    md += f"- Generator: {', '.join(f'`{m}`' for m in gen_models)}\n"
                if 'judge_config' in param_grid:
                    judge_models = [j.get('model', str(j)) if isinstance(j, dict) else str(j) for j in param_grid['judge_config']]
                    md += f"- Judge: {', '.join(f'`{m}`' for m in judge_models)}\n"
                if 'corruption_rate' in param_grid:
                    md += f"- Corruption rate: {param_grid['corruption_rate']}\n"
                if 'embedding_enable' in param_grid:
                    md += f"- CI metrics enabled: {param_grid['embedding_enable']}\n"
                md += "\n"
        
        # Add run statistics
        if provenance.get('mapping_info'):
            md += "### Run Statistics\n\n"
            for info in provenance['mapping_info']:
                md += f"**Experiment:** `{info['experiment_id']}`\n"
                md += f"- Total runs: {info['n_runs_in_mapping']}, Parameter combos: {info['n_combos']}\n\n"
        
        if provenance.get('result_directories'):
            md += "### Result Directories\n\n"
            for dir_path in provenance['result_directories']:
                md += f"- `{dir_path}`\n"
            md += "\n"
        
        md += "---\n\n## 🎯 Answer to RQ2\n\n**" + findings['answer'] + "**\n\n---\n\n"
        md += f"## 📊 Key Findings\n\n"
        md += f"### Best Performing Metric: `{findings['best_metric']}`\n\n"
        md += f"- **AUC Score:** {findings['best_auc']:.3f}\n"
        md += f"- **95% Confidence Interval:** [{findings['best_ci_lower']:.3f}, {findings['best_ci_upper']:.3f}]\n"
        md += f"- **Optimal Threshold:** {findings['best_threshold']:.3f}\n"
        md += f"- **F1 Score at Threshold:** {findings['best_f1']:.3f}\n"
        sig_text = f"✅ Yes (p={findings['best_p_value']})" if findings['best_significant'] else "❌ No"
        md += f"- **Statistical Significance:** {sig_text}\n\n"
        md += "---\n\n"
        md += f"## 📈 Data Summary\n\n"
        md += f"- **Total Edges:** {data['n_edges']}\n"
        md += f"- **Hallucinations:** {data['n_hallucinations']} ({data['hallucination_rate']:.1f}%)\n"
        md += f"- **Non-Hallucinations:** {data['n_non_hallucinations']} ({100-data['hallucination_rate']:.1f}%)\n\n"
        md += "---\n\n"
        md += "## 📋 Detailed Results\n\n"
        md += "### Correlation Analysis\n\n"
        
        # Add correlation results
        for corr in self.results['correlation']['correlations']:
            md += f"- **{corr['metric']}**: r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.4f}"
            md += f" {'✅' if corr['significant'] else '❌'}\n"
        
        md += "\n### Classification Performance\n\n"
        
        # Add ROC results
        for roc in self.results['classification']['roc_auc_scores']:
            md += f"- **{roc['metric']}**: AUC = {roc['roc_auc']:.3f}\n"
        
        md += "\n### Statistical Validation\n\n"
        
        # Add permutation test results
        md += "**Permutation Tests (with Holm-Bonferroni correction):**\n\n"
        for perm in self.results['statistical_validation']['permutation_tests']:
            md += f"- **{perm['metric']}**: p={perm['corrected_p_value']:.4f}"
            md += f" {'✅ Significant' if perm['significant_corrected'] else '❌ Not significant'}\n"
        
        md += "\n---\n\n## 📁 Output Files\n\n"
        md += "### Tables\n\n"
        md += "1. `rq2_correlations.csv` - Correlation coefficients\n"
        md += "2. `rq2_group_comparison.csv` - t-test results\n"
        md += "3. `rq2_roc_auc.csv` - ROC AUC scores\n"
        md += "4. `rq2_optimal_thresholds.csv` - **Critical for RQ3**\n"
        md += "5. `rq2_permutation_tests.csv` - Permutation test results\n"
        md += "6. `rq2_bootstrap_ci.csv` - Bootstrap confidence intervals\n"
        
        md += "\n### Figures\n\n"
        md += "1. `rq2_distributions.png` - Distribution comparisons\n"
        md += "2. `rq2_scatter_plots.png` - Scatter plots with regression\n"
        md += "3. `rq2_roc_curves.png` - ROC curves\n"
        md += "4. `rq2_confusion_matrices.png` - Confusion matrices\n"
        md += "5. `rq2_permutation_distributions.png` - Null distributions\n"
        md += "6. `rq2_bootstrap_distributions.png` - Bootstrap distributions\n"
        
        md += "\n---\n\n"
        md += "*Report generated by RQ2 Master Pipeline v1.0*\n"
        
        return md


def main():
    """Run the complete RQ2 analysis pipeline."""
    
    pipeline = RQ2MasterPipeline()
    
    # Run full analysis
    results = pipeline.run_full_pipeline(n_permutations=10000, n_bootstrap=10000)
    
    print("\n✅ All analyses complete!")
    print(f"\nResults saved to:")
    print(f"  • Tables: {pipeline.tables_dir}")
    print(f"  • Figures: {pipeline.figures_dir}")
    

if __name__ == "__main__":
    main()