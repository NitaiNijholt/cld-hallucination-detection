#!/usr/bin/env python3
"""
Review Experiments for RQ2 Suitability

This script analyzes all available experiments to determine which ones
have the necessary data for RQ2 analysis (CI metrics + judge verdicts).

Usage:
    python review_experiments_for_rq2.py
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from rq2_data_loader import RQ2DataLoader


def review_experiments():
    """
    Review all experiments and identify which are suitable for RQ2 analysis.
    """
    
    print("=" * 80)
    print("RQ2 Experiment Data Review")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    loader = RQ2DataLoader()
    experiments = loader.find_all_experiments()
    
    print(f"Total experiments found: {len(experiments)}\n")
    print("Analyzing each experiment for RQ2 suitability...\n")
    
    suitable_experiments = []
    unsuitable_experiments = []
    
    for i, exp_id in enumerate(experiments, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(experiments)} experiments analyzed...")
        
        try:
            # Try to load the experiment
            df = loader.load_experiment(exp_id, max_files_per_combo=None)
            
            if df.empty:
                unsuitable_experiments.append({
                    'exp_id': exp_id,
                    'reason': 'No data loaded',
                    'n_edges': 0
                })
                continue
            
            # Check for CI metrics
            ci_metrics = ['perplexity', 'min_prob', 'max_window_entropy', 'cosine_similarity']
            has_ci_metrics = any(col in df.columns for col in ci_metrics)
            
            # Check for judge verdicts
            has_judge = 'aggregate_score' in df.columns or 'judge_verdict' in df.columns
            
            # Count edges with CI metrics
            if has_ci_metrics:
                ci_cols_present = [col for col in ci_metrics if col in df.columns]
                n_edges_with_ci = df[ci_cols_present].notna().any(axis=1).sum()
            else:
                n_edges_with_ci = 0
            
            # Get CLDs
            clds = df['cld_name'].unique() if 'cld_name' in df.columns else []
            
            # Determine suitability
            if has_ci_metrics and has_judge and n_edges_with_ci > 0:
                suitable_experiments.append({
                    'exp_id': exp_id,
                    'n_edges': len(df),
                    'n_edges_with_ci': n_edges_with_ci,
                    'ci_coverage': n_edges_with_ci / len(df) * 100,
                    'has_judge': has_judge,
                    'n_clds': len(clds),
                    'clds': ', '.join(clds),
                    'ci_metrics_available': ', '.join(ci_cols_present) if has_ci_metrics else 'None'
                })
            else:
                reasons = []
                if not has_ci_metrics:
                    reasons.append('No CI metrics')
                if not has_judge:
                    reasons.append('No judge verdicts')
                if n_edges_with_ci == 0:
                    reasons.append('CI metrics all null')
                
                unsuitable_experiments.append({
                    'exp_id': exp_id,
                    'reason': '; '.join(reasons),
                    'n_edges': len(df),
                    'n_edges_with_ci': n_edges_with_ci
                })
        
        except Exception as e:
            unsuitable_experiments.append({
                'exp_id': exp_id,
                'reason': f'Error: {str(e)[:50]}',
                'n_edges': 0
            })
    
    print(f"\n{'=' * 80}")
    print("Analysis Complete!")
    print("=" * 80)
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  ✓ Suitable for RQ2: {len(suitable_experiments)}")
    print(f"  ✗ Not suitable: {len(unsuitable_experiments)}")
    print(f"  Suitability rate: {len(suitable_experiments)/len(experiments)*100:.1f}%")
    
    if suitable_experiments:
        total_edges = sum(exp['n_edges'] for exp in suitable_experiments)
        total_edges_with_ci = sum(exp['n_edges_with_ci'] for exp in suitable_experiments)
        
        print(f"\n📈 Data Available for RQ2:")
        print(f"  Total edges across suitable experiments: {total_edges:,}")
        print(f"  Edges with CI metrics: {total_edges_with_ci:,}")
        print(f"  Average CI coverage: {total_edges_with_ci/total_edges*100:.1f}%")
        
        # Top 10 experiments by data volume
        print(f"\n🏆 Top 10 Experiments by Data Volume:")
        sorted_exps = sorted(suitable_experiments, key=lambda x: x['n_edges_with_ci'], reverse=True)
        for i, exp in enumerate(sorted_exps[:10], 1):
            print(f"  {i:2d}. {exp['exp_id']}")
            print(f"      Edges: {exp['n_edges']:,} ({exp['n_edges_with_ci']:,} with CI, {exp['ci_coverage']:.1f}%)")
            print(f"      CLDs: {exp['clds'][:60]}{'...' if len(exp['clds']) > 60 else ''}")
    
    # Save detailed reports
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'analysis_reports'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save suitable experiments
    if suitable_experiments:
        df_suitable = pd.DataFrame(suitable_experiments)
        df_suitable = df_suitable.sort_values('n_edges_with_ci', ascending=False)
        csv_file = output_dir / f'rq2_suitable_experiments_{timestamp}.csv'
        df_suitable.to_csv(csv_file, index=False)
        print(f"\n✓ Saved suitable experiments: {csv_file}")
        print(f"  ({len(suitable_experiments)} experiments)")
    
    # Save unsuitable experiments
    if unsuitable_experiments:
        df_unsuitable = pd.DataFrame(unsuitable_experiments)
        csv_file = output_dir / f'rq2_unsuitable_experiments_{timestamp}.csv'
        df_unsuitable.to_csv(csv_file, index=False)
        print(f"✓ Saved unsuitable experiments: {csv_file}")
        print(f"  ({len(unsuitable_experiments)} experiments)")
    
    # Generate summary report
    summary = {
        'timestamp': timestamp,
        'total_experiments': len(experiments),
        'suitable_experiments': len(suitable_experiments),
        'unsuitable_experiments': len(unsuitable_experiments),
        'suitability_rate': len(suitable_experiments) / len(experiments) * 100 if experiments else 0,
        'total_edges_available': sum(exp['n_edges_with_ci'] for exp in suitable_experiments) if suitable_experiments else 0,
        'top_experiments': sorted_exps[:10] if suitable_experiments else []
    }
    
    json_file = output_dir / f'rq2_experiment_review_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary report: {json_file}")
    
    # Recommendations
    print(f"\n{'-' * 80}")
    print("💡 Recommendations:")
    print(f"{'-' * 80}")
    
    if len(suitable_experiments) == 0:
        print("⚠️  No experiments with required data found!")
        print("   Action: Run experiments with CI metrics enabled (embedding_enable: true)")
        print("   and judge_edges: true in experiment configs")
    elif len(suitable_experiments) < 5:
        print(f"⚠️  Only {len(suitable_experiments)} suitable experiment(s) found")
        print("   Recommendation: Run more experiments for robust RQ2 analysis")
        print("   Target: At least 5-10 experiments across different CLDs")
    else:
        print(f"✓ Good: {len(suitable_experiments)} suitable experiments available")
        
        if suitable_experiments:
            avg_coverage = sum(exp['ci_coverage'] for exp in suitable_experiments) / len(suitable_experiments)
            if avg_coverage < 50:
                print(f"⚠️  Low CI coverage: {avg_coverage:.1f}% average")
                print("   Many edges missing CI metrics")
            else:
                print(f"✓ Good CI coverage: {avg_coverage:.1f}% average")
            
            # Check CLD diversity
            all_clds = set()
            for exp in suitable_experiments:
                all_clds.update(exp['clds'].split(', '))
            
            if len(all_clds) < 3:
                print(f"⚠️  Limited CLD diversity: only {len(all_clds)} CLDs")
                print("   Recommendation: Run experiments on more diverse CLDs")
            else:
                print(f"✓ Good CLD diversity: {len(all_clds)} different CLDs")
    
    print(f"\n{'-' * 80}")
    print("Next Steps:")
    print(f"{'-' * 80}")
    if suitable_experiments:
        print("1. Review the suitable experiments CSV file")
        print("2. Select experiments for RQ2 analysis")
        print("3. Run: python analysis/rq2_master_pipeline.py")
        print("4. Or load specific experiments using the data loader")
    else:
        print("1. Check configs/ for experiments with embedding_enable: true")
        print("2. Run experiments with judge_edges: true")
        print("3. Re-run this review script")
    
    print()


if __name__ == '__main__':
    review_experiments()

