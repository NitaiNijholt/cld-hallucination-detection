#!/usr/bin/env python3
"""
Node Ablation Analysis Pipeline

Unified tool for analyzing node generation prompt ablation studies.
Automatically runs data loading and aggregate analysis.

Usage:
  # Analyze single experiment
  python run_node_ablation_analysis.py parameter_tuning_experiments/results/exp_20251009_163247_a49a01a2/
  
  # With custom baseline for effect size calculations
  python run_node_ablation_analysis.py exp_dir/ --baseline V5_Nitai_C
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import our modules
from node_ablation_data_loader import NodeAblationDataLoader
from node_ablation_aggregate import NodeAblationAggregator


def main():
    parser = argparse.ArgumentParser(
        description="Node Ablation Analysis Pipeline - End-to-end analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze test experiment
  python run_node_ablation_analysis.py parameter_tuning_experiments/results/exp_20251009_163247_a49a01a2/
  
  # Full ablation study with 7 prompts
  python run_node_ablation_analysis.py parameter_tuning_experiments/results/exp_20251009_XXXXXX_full_ablation/
  
  # With custom baseline
  python run_node_ablation_analysis.py exp_dir/ --baseline V1_Role_Only
        """
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory (e.g., parameter_tuning_experiments/results/exp_*/)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for all analyses (auto-generated if not specified)"
    )
    parser.add_argument(
        "--baseline",
        default="Node_V0_Minimal",
        help="Baseline prompt variant for effect size calculations (default: Node_V0_Minimal)"
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregate analysis (only extract data)"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return 1
    
    if not experiment_dir.is_dir():
        print(f"❌ Path is not a directory: {experiment_dir}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path("parameter_tuning_experiments/node_ablation_analyses")
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(" "*20 + "NODE ABLATION ANALYSIS PIPELINE")
    print("="*80)
    print(f"\n📁 Experiment: {experiment_dir.name}")
    print(f"📂 Output base: {output_base}")
    print(f"⚖️  Baseline: {args.baseline}")
    print()
    
    # ============================================================================
    # STEP 1: DATA LOADING
    # ============================================================================
    print("="*80)
    print("STEP 1/2: DATA LOADING")
    print("="*80)
    print()
    
    try:
        loader = NodeAblationDataLoader(experiment_dir)
        df = loader.load_all_metrics()
        
        # Save data
        data_file = loader.save_data(df, output_base)
        
        print()
        print("✅ Data loading complete!")
        print(f"   • Extracted {len(df)} records")
        print(f"   • Prompts: {df['prompt'].nunique()}")
        print(f"   • CLDs: {df['cld_name'].nunique()}")
        print(f"   • Data file: {data_file}")
        
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if args.skip_aggregate:
        print("\n⏭️  Skipping aggregate analysis (--skip-aggregate flag)")
        return 0
    
    # ============================================================================
    # STEP 2: AGGREGATE ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2/2: AGGREGATE ANALYSIS")
    print("="*80)
    print()
    
    try:
        aggregator = NodeAblationAggregator(data_file)
        aggregator.run_full_analysis(baseline_prompt=args.baseline)
        
        print()
        print("="*80)
        print("✅ NODE ABLATION ANALYSIS COMPLETE!")
        print("="*80)
        print()
        print(f"📊 Data extracted: {len(df)} records")
        print(f"📈 Statistical analysis: Complete")
        print(f"📉 Visualizations: Generated")
        print(f"📝 Report: Created")
        print()
        print(f"📁 All outputs saved to:")
        print(f"   {aggregator.output_dir}")
        print()
        print("Key files:")
        print("  • aggregate_meta_statistics.xlsx - Summary statistics")
        print("  • aggregate_anova.csv - ANOVA results")
        print("  • aggregate_pairwise.csv - Pairwise comparisons")
        print("  • node_ablation_aggregate_report_*.md - Full report")
        print("  • *.png - Visualizations (4 figures)")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Aggregate analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
