#!/usr/bin/env python3
"""
Node Ablation Aggregate Analysis

Aggregates node generation performance across prompts, CLDs, and runs.
Generates statistical comparisons and visualizations for the ablation study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from scipy import stats
import argparse


class NodeAblationAggregator:
    """
    Aggregates and analyzes node generation performance across ablation study.
    
    Performs:
    1. Meta-statistics (mean, std, CI) by prompt variant
    2. Statistical comparisons (ANOVA, pairwise t-tests)
    3. Effect size calculations (Cohen's d)
    4. Visualizations (bar charts, box plots, heatmaps)
    5. Comprehensive report generation
    """
    
    def __init__(self, data_file: Path, output_dir: Path = None):
        """
        Initialize aggregator.
        
        Args:
            data_file: Path to CSV file with node ablation data
            output_dir: Output directory for results (auto-generated if None)
        """
        self.data_file = Path(data_file)
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Load data
        self.df = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.df)} records from {self.data_file.name}")
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = self.df['experiment_id'].iloc[0] if 'experiment_id' in self.df.columns else 'unknown'
            output_dir = self.data_file.parent / f'node_ablation_aggregate_{timestamp}_{experiment_id}'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def compute_meta_statistics(self, metric_col: str = 'hybrid_f1', 
                                  group_by: str = 'prompt') -> pd.DataFrame:
        """
        Compute meta-statistics (mean, std, CI) for a metric grouped by a variable.
        
        Args:
            metric_col: Column name for the metric to analyze
            group_by: Column to group by (e.g., 'prompt', 'cld_name')
            
        Returns:
            DataFrame with meta-statistics
        """
        meta_stats = self.df.groupby(group_by)[metric_col].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('n', 'count')
        ]).reset_index()
        
        # Compute 95% CI
        meta_stats['ci_lower'] = meta_stats['mean'] - 1.96 * meta_stats['std'] / np.sqrt(meta_stats['n'])
        meta_stats['ci_upper'] = meta_stats['mean'] + 1.96 * meta_stats['std'] / np.sqrt(meta_stats['n'])
        
        # Add SEM
        meta_stats['sem'] = meta_stats['std'] / np.sqrt(meta_stats['n'])
        
        return meta_stats
    
    def run_anova(self, metric_col: str = 'hybrid_f1', group_col: str = 'prompt') -> Dict:
        """
        Run one-way ANOVA to test for significant differences between groups.
        
        Args:
            metric_col: Metric to test
            group_col: Grouping variable
            
        Returns:
            Dictionary with ANOVA results
        """
        groups = [group[metric_col].values for name, group in self.df.groupby(group_col)]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'metric': metric_col,
            'group_by': group_col,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_groups': len(groups)
        }
    
    def run_pairwise_tests(self, metric_col: str = 'hybrid_f1', 
                           group_col: str = 'prompt') -> pd.DataFrame:
        """
        Run pairwise t-tests with Bonferroni correction.
        
        Args:
            metric_col: Metric to test
            group_col: Grouping variable
            
        Returns:
            DataFrame with pairwise comparison results
        """
        groups = self.df.groupby(group_col)[metric_col]
        group_names = list(groups.groups.keys())
        
        results = []
        
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                g1_name = group_names[i]
                g2_name = group_names[j]
                
                g1_data = groups.get_group(g1_name).values
                g2_data = groups.get_group(g2_name).values
                
                # Two-sided t-test
                t_stat, p_value = stats.ttest_ind(g1_data, g2_data)
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt(((len(g1_data)-1)*np.std(g1_data, ddof=1)**2 + 
                                      (len(g2_data)-1)*np.std(g2_data, ddof=1)**2) / 
                                     (len(g1_data) + len(g2_data) - 2))
                cohens_d = (np.mean(g1_data) - np.mean(g2_data)) / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'group1': g1_name,
                    'group2': g2_name,
                    'mean1': np.mean(g1_data),
                    'mean2': np.mean(g2_data),
                    'mean_diff': np.mean(g1_data) - np.mean(g2_data),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'n1': len(g1_data),
                    'n2': len(g2_data)
                })
        
        df_results = pd.DataFrame(results)
        
        # Bonferroni correction
        n_comparisons = len(df_results)
        df_results['p_value_bonferroni'] = df_results['p_value'] * n_comparisons
        df_results['p_value_bonferroni'] = df_results['p_value_bonferroni'].clip(upper=1.0)
        df_results['significant'] = df_results['p_value'] < 0.05
        df_results['significant_bonferroni'] = df_results['p_value_bonferroni'] < 0.05
        
        return df_results.sort_values('p_value')
    
    def compute_effect_sizes(self, metric_col: str = 'hybrid_f1',
                             baseline_prompt: str = 'V0_Minimal') -> pd.DataFrame:
        """
        Compute effect sizes (Cohen's d) relative to baseline prompt.
        
        Args:
            metric_col: Metric to analyze
            baseline_prompt: Baseline prompt to compare against
            
        Returns:
            DataFrame with effect sizes
        """
        if baseline_prompt not in self.df['prompt'].values:
            print(f"Warning: Baseline prompt '{baseline_prompt}' not found in data")
            return pd.DataFrame()
        
        baseline_data = self.df[self.df['prompt'] == baseline_prompt][metric_col].values
        baseline_mean = np.mean(baseline_data)
        baseline_std = np.std(baseline_data, ddof=1)
        
        results = []
        
        for prompt in self.df['prompt'].unique():
            if prompt == baseline_prompt:
                continue
            
            prompt_data = self.df[self.df['prompt'] == prompt][metric_col].values
            prompt_mean = np.mean(prompt_data)
            prompt_std = np.std(prompt_data, ddof=1)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(baseline_data)-1)*baseline_std**2 + 
                                  (len(prompt_data)-1)*prompt_std**2) / 
                                 (len(baseline_data) + len(prompt_data) - 2))
            
            # Cohen's d
            cohens_d = (prompt_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'prompt': prompt,
                'mean': prompt_mean,
                'baseline_mean': baseline_mean,
                'difference': prompt_mean - baseline_mean,
                'cohens_d': cohens_d,
                'effect_magnitude': self._interpret_cohens_d(cohens_d)
            })
        
        return pd.DataFrame(results).sort_values('cohens_d', ascending=False)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def plot_performance_by_prompt(self, metric_col: str = 'hybrid_f1',
                                   metric_label: str = 'Hybrid F1 Score'):
        """Generate bar plot with error bars for each prompt variant."""
        meta_stats = self.compute_meta_statistics(metric_col, 'prompt')
        meta_stats = meta_stats.sort_values('mean', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(meta_stats))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(meta_stats)))
        
        ax.barh(y_pos, meta_stats['mean'], color=colors, alpha=0.8)
        ax.errorbar(meta_stats['mean'], y_pos,
                    xerr=[meta_stats['mean'] - meta_stats['ci_lower'],
                          meta_stats['ci_upper'] - meta_stats['mean']],
                    fmt='none', color='black', capsize=5, linewidth=2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(meta_stats['prompt'], fontsize=11)
        ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} by Prompt Variant\\n(Mean ± 95% CI)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (idx, row) in enumerate(meta_stats.iterrows()):
            ax.text(row['mean'], i, f" {row['mean']:.3f}", 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / f'performance_by_prompt_{metric_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file.name}")
    
    def plot_performance_by_cld(self, metric_col: str = 'hybrid_f1',
                               metric_label: str = 'Hybrid F1 Score'):
        """Generate bar plot for each CLD."""
        meta_stats = self.compute_meta_statistics(metric_col, 'cld_name')
        meta_stats = meta_stats.sort_values('mean', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(meta_stats))
        ax.barh(y_pos, meta_stats['mean'], color='coral', alpha=0.7)
        ax.errorbar(meta_stats['mean'], y_pos,
                    xerr=[meta_stats['mean'] - meta_stats['ci_lower'],
                          meta_stats['ci_upper'] - meta_stats['mean']],
                    fmt='none', color='black', capsize=5)
        
        ax.set_yticks(y_pos)
        # Truncate long CLD names
        labels = [name[:50] + '...' if len(name) > 50 else name 
                  for name in meta_stats['cld_name']]
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} by CLD\\n(Mean ± 95% CI)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'performance_by_cld_{metric_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file.name}")
    
    def plot_heatmap(self, metric_col: str = 'hybrid_f1',
                    metric_label: str = 'Hybrid F1'):
        """Generate heatmap of performance across prompts and CLDs."""
        pivot = self.df.pivot_table(values=metric_col, index='prompt', 
                                     columns='cld_name', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.4, vmax=0.9, center=0.65,
                   cbar_kws={'label': metric_label},
                   linewidths=0.5, linecolor='gray',
                   ax=ax)
        
        ax.set_title(f'{metric_label} Heatmap\\nPrompts × CLDs', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Ground Truth CLD', fontsize=11, fontweight='bold')
        ax.set_ylabel('Prompt Variant', fontsize=11, fontweight='bold')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = self.output_dir / f'heatmap_{metric_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file.name}")
    
    def plot_boxplots(self, metric_col: str = 'hybrid_f1',
                     metric_label: str = 'Hybrid F1 Score'):
        """Generate box plots for distribution visualization."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort prompts by median performance
        prompt_order = self.df.groupby('prompt')[metric_col].median().sort_values().index
        
        sns.boxplot(data=self.df, y='prompt', x=metric_col, order=prompt_order,
                   hue='prompt', palette='Set2', legend=False, ax=ax)
        sns.stripplot(data=self.df, y='prompt', x=metric_col, order=prompt_order,
                     color='black', alpha=0.5, size=5, ax=ax)
        
        ax.set_ylabel('Prompt Variant', fontsize=12, fontweight='bold')
        ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} Distribution by Prompt\\n(Box plot + individual runs)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_file = self.output_dir / f'boxplot_{metric_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file.name}")
    
    def generate_report(self, baseline_prompt: str = 'V0_Minimal'):
        """Generate comprehensive markdown report."""
        report_lines = []
        
        # Header
        report_lines.append("# Node Generation Prompt Ablation Study - Aggregate Analysis")
        report_lines.append("")
        report_lines.append(f"**Generated:** {self.timestamp}")
        report_lines.append(f"**Experiment:** {self.df['experiment_id'].iloc[0]}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Data summary
        report_lines.append("## Data Summary")
        report_lines.append("")
        report_lines.append(f"- **Total Records:** {len(self.df)}")
        report_lines.append(f"- **Prompt Variants:** {self.df['prompt'].nunique()}")
        for prompt in sorted(self.df['prompt'].unique()):
            count = len(self.df[self.df['prompt'] == prompt])
            report_lines.append(f"  - `{prompt}`: {count} runs")
        report_lines.append(f"- **CLDs:** {self.df['cld_name'].nunique()}")
        for cld in sorted(self.df['cld_name'].unique()):
            count = len(self.df[self.df['cld_name'] == cld])
            report_lines.append(f"  - {cld}: {count} runs")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Meta-statistics by prompt
        report_lines.append("## Performance by Prompt Variant")
        report_lines.append("")
        report_lines.append("### Hybrid F1 Score (Primary Metric)")
        report_lines.append("")
        meta_stats = self.compute_meta_statistics('hybrid_f1', 'prompt').sort_values('mean', ascending=False)
        report_lines.append("| Prompt | Mean | Std | Min | Max | 95% CI | N |")
        report_lines.append("|--------|------|-----|-----|-----|--------|---|")
        for _, row in meta_stats.iterrows():
            report_lines.append(
                f"| **{row['prompt']}** | {row['mean']:.4f} | {row['std']:.4f} | "
                f"{row['min']:.4f} | {row['max']:.4f} | "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {int(row['n'])} |"
            )
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # ANOVA
        report_lines.append("## Statistical Analysis")
        report_lines.append("")
        report_lines.append("### One-Way ANOVA")
        report_lines.append("")
        anova_result = self.run_anova('hybrid_f1', 'prompt')
        report_lines.append(f"- **F-statistic:** {anova_result['f_statistic']:.4f}")
        report_lines.append(f"- **p-value:** {anova_result['p_value']:.6f}")
        report_lines.append(f"- **Significant:** {'✅ Yes' if anova_result['significant'] else '❌ No'}")
        report_lines.append(f"- **Interpretation:** " + (
            "There ARE statistically significant differences between prompt variants." 
            if anova_result['significant'] else 
            "There are NO statistically significant differences between prompt variants."
        ))
        report_lines.append("")
        report_lines.append("### Pairwise Comparisons (t-tests with Bonferroni correction)")
        report_lines.append("")
        pairwise = self.run_pairwise_tests('hybrid_f1', 'prompt')
        report_lines.append("| Comparison | Mean Diff | t-stat | p-value | Bonferroni p | Cohen's d | Sig | Sig (Bonf) |")
        report_lines.append("|------------|-----------|--------|---------|--------------|-----------|-----|------------|")
        for _, row in pairwise.iterrows():
            sig_marker = '✅' if row['significant'] else '❌'
            sig_bonf_marker = '✅' if row['significant_bonferroni'] else '❌'
            report_lines.append(
                f"| {row['group1']} vs {row['group2']} | {row['mean_diff']:+.4f} | "
                f"{row['t_statistic']:.3f} | {row['p_value']:.4f} | {row['p_value_bonferroni']:.4f} | "
                f"{row['cohens_d']:.3f} | {sig_marker} | {sig_bonf_marker} |"
            )
        report_lines.append("")
        report_lines.append("### Effect Sizes (Cohen's d)")
        report_lines.append("")
        report_lines.append(f"**Baseline:** `{baseline_prompt}`")
        report_lines.append("")
        effect_sizes = self.compute_effect_sizes('hybrid_f1', baseline_prompt)
        if not effect_sizes.empty:
            report_lines.append("| Prompt | Mean | Difference | Cohen's d | Magnitude |")
            report_lines.append("|--------|------|------------|-----------|-----------|")
            for _, row in effect_sizes.iterrows():
                report_lines.append(
                    f"| **{row['prompt']}** | {row['mean']:.4f} | {row['difference']:+.4f} | "
                    f"{row['cohens_d']:+.3f} | {row['effect_magnitude']} |"
                )
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Performance by CLD
        report_lines.append("## Performance by CLD")
        report_lines.append("")
        cld_stats = self.compute_meta_statistics('hybrid_f1', 'cld_name').sort_values('mean', ascending=False)
        report_lines.append("| CLD | Mean | Std | Min | Max | N |")
        report_lines.append("|-----|------|-----|-----|-----|---|")
        for _, row in cld_stats.iterrows():
            cld_short = row['cld_name'][:60] + '...' if len(row['cld_name']) > 60 else row['cld_name']
            report_lines.append(
                f"| {cld_short} | {row['mean']:.4f} | {row['std']:.4f} | "
                f"{row['min']:.4f} | {row['max']:.4f} | {int(row['n'])} |"
            )
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Key findings
        report_lines.append("## Key Findings")
        report_lines.append("")
        best_prompt = meta_stats.iloc[0]
        worst_prompt = meta_stats.iloc[-1]
        report_lines.append(f"1. **Best Performing Prompt:** `{best_prompt['prompt']}` (Mean F1 = {best_prompt['mean']:.4f})")
        report_lines.append(f"2. **Worst Performing Prompt:** `{worst_prompt['prompt']}` (Mean F1 = {worst_prompt['mean']:.4f})")
        report_lines.append(f"3. **Performance Range:** {worst_prompt['mean']:.4f} to {best_prompt['mean']:.4f} (Δ = {best_prompt['mean'] - worst_prompt['mean']:.4f})")
        report_lines.append(f"4. **ANOVA Significant:** {'Yes' if anova_result['significant'] else 'No'} (p = {anova_result['p_value']:.4f})")
        
        easiest_cld = cld_stats.iloc[0]
        hardest_cld = cld_stats.iloc[-1]
        report_lines.append(f"5. **Easiest CLD:** {easiest_cld['cld_name'][:50]} (Mean F1 = {easiest_cld['mean']:.4f})")
        report_lines.append(f"6. **Hardest CLD:** {hardest_cld['cld_name'][:50]} (Mean F1 = {hardest_cld['mean']:.4f})")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Output files
        report_lines.append("## Generated Files")
        report_lines.append("")
        report_lines.append("### Data Tables")
        report_lines.append("")
        report_lines.append("- `aggregate_meta_statistics.xlsx` - Meta-statistics by prompt")
        report_lines.append("- `aggregate_anova.csv` - ANOVA results")
        report_lines.append("- `aggregate_pairwise.csv` - Pairwise t-test results")
        report_lines.append("- `aggregate_effect_sizes.csv` - Effect sizes vs baseline")
        report_lines.append("- `aggregate_by_cld.csv` - Performance by CLD")
        report_lines.append("")
        report_lines.append("### Visualizations")
        report_lines.append("")
        report_lines.append("- `performance_by_prompt_hybrid_f1.png` - Bar chart with CI")
        report_lines.append("- `performance_by_cld_hybrid_f1.png` - Performance by CLD")
        report_lines.append("- `heatmap_hybrid_f1.png` - Prompts × CLDs heatmap")
        report_lines.append("- `boxplot_hybrid_f1.png` - Distribution box plots")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report generated by Node Ablation Aggregator*")
        
        # Write report
        report_path = self.output_dir / f'node_ablation_aggregate_report_{self.timestamp}.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Saved: {report_path.name}")
        
        return report_path
    
    def run_full_analysis(self, baseline_prompt: str = 'V0_Minimal'):
        """Run complete aggregate analysis pipeline."""
        print("\\n" + "="*80)
        print(" "*20 + "NODE ABLATION AGGREGATE ANALYSIS")
        print("="*80)
        print(f"\\nData: {self.data_file.name}")
        print(f"Output: {self.output_dir}")
        print()
        
        # Compute meta-statistics
        print("[1/6] Computing meta-statistics...")
        meta_stats_prompt = self.compute_meta_statistics('hybrid_f1', 'prompt')
        meta_stats_cld = self.compute_meta_statistics('hybrid_f1', 'cld_name')
        
        # Statistical tests
        print("[2/6] Running statistical tests (ANOVA, pairwise t-tests)...")
        anova_result = self.run_anova('hybrid_f1', 'prompt')
        pairwise_results = self.run_pairwise_tests('hybrid_f1', 'prompt')
        effect_sizes = self.compute_effect_sizes('hybrid_f1', baseline_prompt)
        
        # Save tables
        print("[3/6] Saving data tables...")
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(self.output_dir / 'aggregate_meta_statistics.xlsx') as writer:
            meta_stats_prompt.to_excel(writer, sheet_name='By_Prompt', index=False)
            meta_stats_cld.to_excel(writer, sheet_name='By_CLD', index=False)
            pairwise_results.to_excel(writer, sheet_name='Pairwise_Tests', index=False)
            if not effect_sizes.empty:
                effect_sizes.to_excel(writer, sheet_name='Effect_Sizes', index=False)
        
        pd.DataFrame([anova_result]).to_csv(self.output_dir / 'aggregate_anova.csv', index=False)
        pairwise_results.to_csv(self.output_dir / 'aggregate_pairwise.csv', index=False)
        if not effect_sizes.empty:
            effect_sizes.to_csv(self.output_dir / 'aggregate_effect_sizes.csv', index=False)
        meta_stats_cld.to_csv(self.output_dir / 'aggregate_by_cld.csv', index=False)
        
        # Generate visualizations
        print("[4/6] Generating visualizations...")
        self.plot_performance_by_prompt('hybrid_f1', 'Hybrid F1 Score')
        self.plot_performance_by_cld('hybrid_f1', 'Hybrid F1 Score')
        self.plot_heatmap('hybrid_f1', 'Hybrid F1')
        self.plot_boxplots('hybrid_f1', 'Hybrid F1 Score')
        
        # Generate report
        print("[5/6] Generating comprehensive report...")
        report_path = self.generate_report(baseline_prompt)
        
        # Print summary
        print("\\n[6/6] Analysis complete!")
        print("\\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        best_prompt = meta_stats_prompt.sort_values('mean', ascending=False).iloc[0]
        print(f"\\n🏆 Best Prompt: {best_prompt['prompt']} (Mean F1 = {best_prompt['mean']:.4f})")
        print(f"\\n📊 ANOVA: F={anova_result['f_statistic']:.3f}, p={anova_result['p_value']:.6f}")
        print(f"   → {'Significant differences found!' if anova_result['significant'] else 'No significant differences.'}")
        print(f"\\n📁 Results saved to: {self.output_dir}")
        print("="*80)


def main():
    """CLI for aggregate analysis."""
    parser = argparse.ArgumentParser(
        description="Aggregate node ablation results across prompts and CLDs"
    )
    parser.add_argument(
        "data_file",
        help="Path to CSV file with node ablation data (from data loader)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--baseline",
        default="V0_Minimal",
        help="Baseline prompt for effect size calculations (default: V0_Minimal)"
    )
    
    args = parser.parse_args()
    
    aggregator = NodeAblationAggregator(args.data_file, args.output_dir)
    aggregator.run_full_analysis(baseline_prompt=args.baseline)


if __name__ == "__main__":
    main()
