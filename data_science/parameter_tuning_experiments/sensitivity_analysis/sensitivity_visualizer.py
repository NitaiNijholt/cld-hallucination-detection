"""
Sensitivity Analysis Visualizer

Creates visualizations for Sobol sensitivity analysis results.
This is the VIEW layer - handles only visualization logic.

Architecture:
- Takes analysis results from SobolAnalyzer
- Creates publication-quality figures
- Saves to specified output directories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class SensitivityVisualizer:
    """
    Visualization module for Sobol sensitivity analysis.
    
    Creates:
    - Bar plots of S1 and ST indices
    - Comparison plots
    - Heatmaps of second-order interactions
    - Convergence plots
    - Parameter importance rankings
    """
    
    def __init__(
        self,
        output_dir: str = "parameter_tuning_experiments/sensitivity_analysis/visualizations",
        style: str = "whitegrid"
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style(style)
        
        logger.info(f"Initialized SensitivityVisualizer")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def plot_sobol_indices(
        self,
        results: Dict[str, pd.DataFrame],
        outcome_name: str = "outcome",
        figsize: Tuple[int, int] = (14, 10),
        show_confidence: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive Sobol indices visualization.
        
        Creates 4 subplots:
        1. First-order indices (S1) - bar plot
        2. Total-effect indices (ST) - bar plot
        3. S1 vs ST comparison - grouped bar plot
        4. Second-order interactions (S2) - heatmap
        
        Args:
            results: Results dict from SobolAnalyzer.analyze()
            outcome_name: Name of outcome variable
            figsize: Figure size
            show_confidence: Whether to show confidence intervals
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Sobol Sensitivity Indices - {outcome_name}', fontsize=16, y=0.995)
        
        # Plot 1: First-order indices (S1)
        self._plot_s1_indices(axes[0, 0], results['S1'], show_confidence)
        
        # Plot 2: Total-effect indices (ST)
        self._plot_st_indices(axes[0, 1], results['ST'], show_confidence)
        
        # Plot 3: S1 vs ST comparison
        self._plot_s1_vs_st(axes[1, 0], results)
        
        # Plot 4: Second-order interactions (if available)
        if 'S2' in results and len(results['S2']) > 0:
            self._plot_s2_heatmap(axes[1, 1], results['S2'], results['S1']['Parameter'].tolist())
        else:
            axes[1, 1].text(0.5, 0.5, 'Second-order\nindices not\ncalculated',
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / f"sobol_indices_{outcome_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Sobol indices plot: {output_file}")
        
        return fig
    
    def _plot_s1_indices(self, ax, s1_df: pd.DataFrame, show_confidence: bool):
        """Plot first-order indices."""
        if show_confidence:
            ax.barh(s1_df['Parameter'], s1_df['S1'], xerr=s1_df['S1_conf'],
                   capsize=5, color='steelblue', alpha=0.8)
        else:
            ax.barh(s1_df['Parameter'], s1_df['S1'], color='steelblue', alpha=0.8)
        
        ax.set_xlabel('First-Order Index (S1)')
        ax.set_title('Direct Effects')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=min(0, s1_df['S1'].min() - 0.1))
    
    def _plot_st_indices(self, ax, st_df: pd.DataFrame, show_confidence: bool):
        """Plot total-effect indices."""
        if show_confidence:
            ax.barh(st_df['Parameter'], st_df['ST'], xerr=st_df['ST_conf'],
                   capsize=5, color='coral', alpha=0.8)
        else:
            ax.barh(st_df['Parameter'], st_df['ST'], color='coral', alpha=0.8)
        
        ax.set_xlabel('Total-Effect Index (ST)')
        ax.set_title('Direct + Interaction Effects')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=min(0, st_df['ST'].min() - 0.1))
    
    def _plot_s1_vs_st(self, ax, results: Dict[str, pd.DataFrame]):
        """Plot S1 vs ST comparison."""
        merged = pd.merge(
            results['S1'][['Parameter', 'S1']],
            results['ST'][['Parameter', 'ST']],
            on='Parameter'
        )
        
        x = np.arange(len(merged))
        width = 0.35
        
        ax.bar(x - width/2, merged['S1'], width, label='S1 (Direct)', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, merged['ST'], width, label='ST (Total)', alpha=0.8, color='coral')
        
        ax.set_xticks(x)
        ax.set_xticklabels(merged['Parameter'], rotation=45, ha='right')
        ax.set_ylabel('Sensitivity Index')
        ax.set_title('S1 vs ST Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    def _plot_s2_heatmap(self, ax, s2_df: pd.DataFrame, param_names: List[str]):
        """Plot second-order interactions heatmap."""
        n_vars = len(param_names)
        s2_matrix = np.zeros((n_vars, n_vars))
        
        # Fill matrix
        for _, row in s2_df.iterrows():
            i = param_names.index(row['Param1'])
            j = param_names.index(row['Param2'])
            s2_matrix[i, j] = row['S2']
            s2_matrix[j, i] = row['S2']
        
        # Plot
        im = ax.imshow(s2_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticklabels(param_names)
        ax.set_title('Second-Order Interactions (S2)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('S2 Index', rotation=270, labelpad=15)
        
        # Add text annotations for significant interactions
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and s2_matrix[i, j] > 0.05:
                    ax.text(j, i, f'{s2_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    def plot_convergence(
        self,
        convergence_df: pd.DataFrame,
        outcome_name: str = "outcome",
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot convergence of Sobol indices with increasing sample size.
        
        Args:
            convergence_df: DataFrame from SobolAnalyzer.compute_convergence_metrics()
            outcome_name: Name of outcome variable
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Sobol Index Convergence - {outcome_name}', fontsize=14)
        
        # Plot S1 convergence
        for param in convergence_df['parameter'].unique():
            param_data = convergence_df[convergence_df['parameter'] == param]
            ax1.plot(param_data['n'], param_data['S1'], marker='o', label=param, linewidth=2)
        
        ax1.set_xlabel('N (base samples)')
        ax1.set_ylabel('First-Order Index (S1)')
        ax1.set_title('S1 Convergence')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(alpha=0.3)
        
        # Plot ST convergence
        for param in convergence_df['parameter'].unique():
            param_data = convergence_df[convergence_df['parameter'] == param]
            ax2.plot(param_data['n'], param_data['ST'], marker='o', label=param, linewidth=2)
        
        ax2.set_xlabel('N (base samples)')
        ax2.set_ylabel('Total-Effect Index (ST)')
        ax2.set_title('ST Convergence')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / f"sobol_convergence_{outcome_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved convergence plot: {output_file}")
        
        return fig
    
    def plot_parameter_ranking(
        self,
        results: Dict[str, pd.DataFrame],
        outcome_name: str = "outcome",
        figsize: Tuple[int, int] = (10, 6),
        top_n: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot parameter importance ranking.
        
        Args:
            results: Results dict from SobolAnalyzer.analyze()
            outcome_name: Name of outcome variable
            figsize: Figure size
            top_n: Show only top N parameters (None = all)
        
        Returns:
            Matplotlib figure
        """
        # Merge S1 and ST
        merged = pd.merge(
            results['S1'][['Parameter', 'S1', 'S1_conf']],
            results['ST'][['Parameter', 'ST', 'ST_conf']],
            on='Parameter'
        )
        
        # Sort by ST (most important first)
        merged = merged.sort_values('ST', ascending=True)
        
        if top_n:
            merged = merged.tail(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(merged))
        
        # Plot ST bars
        ax.barh(y_pos, merged['ST'], xerr=merged['ST_conf'],
               capsize=5, alpha=0.7, label='Total Effect (ST)', color='coral')
        
        # Plot S1 bars on top
        ax.barh(y_pos, merged['S1'], xerr=merged['S1_conf'],
               capsize=5, alpha=0.9, label='Direct Effect (S1)', color='steelblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['Parameter'])
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(f'Parameter Importance Ranking - {outcome_name}')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / f"parameter_ranking_{outcome_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved parameter ranking plot: {output_file}")
        
        return fig
    
    def plot_multi_outcome_comparison(
        self,
        results_dict: Dict[str, Dict[str, pd.DataFrame]],
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Compare Sobol indices across multiple outcomes.
        
        Args:
            results_dict: Dict mapping outcome_name -> results_dict
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        outcomes = list(results_dict.keys())
        n_outcomes = len(outcomes)
        
        # Get all parameters
        first_outcome = outcomes[0]
        param_names = results_dict[first_outcome]['S1']['Parameter'].tolist()
        n_params = len(param_names)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, n_outcomes, figsize=figsize)
        if n_outcomes == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle('Sobol Indices Across Multiple Outcomes', fontsize=16)
        
        for idx, outcome in enumerate(outcomes):
            results = results_dict[outcome]
            
            # S1 plot
            ax = axes[0, idx]
            s1_df = results['S1']
            ax.barh(s1_df['Parameter'], s1_df['S1'], color='steelblue', alpha=0.8)
            ax.set_xlabel('S1')
            ax.set_title(f'{outcome}')
            ax.grid(axis='x', alpha=0.3)
            if idx == 0:
                ax.set_ylabel('Parameter')
            else:
                ax.set_yticklabels([])
            
            # ST plot
            ax = axes[1, idx]
            st_df = results['ST']
            ax.barh(st_df['Parameter'], st_df['ST'], color='coral', alpha=0.8)
            ax.set_xlabel('ST')
            ax.grid(axis='x', alpha=0.3)
            if idx == 0:
                ax.set_ylabel('Parameter')
            else:
                ax.set_yticklabels([])
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / "multi_outcome_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-outcome comparison plot: {output_file}")
        
        return fig





