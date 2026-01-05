#!/usr/bin/env python3
"""
Simple analysis of score-verdict correlation (no scipy dependency).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_judge_data(excel_path: Path) -> pd.DataFrame:
    """Extract aggregate scores and verdicts from Excel."""
    try:
        df = pd.read_excel(excel_path, sheet_name='All Edges')
        df = df[df['Relationship Type'] != 'NONE'].copy()
        
        scores = []
        for _, row in df.iterrows():
            verdict = row.get('Judge Verdict', '')
            aggregate_score = row.get('Aggregate Score', np.nan)
            
            if pd.isna(aggregate_score):
                judge_msg = row.get('Judge Message', '')
                try:
                    judge_data = json.loads(judge_msg)
                    aggregate_score = judge_data.get('aggregate_score', np.nan)
                except:
                    pass
            
            scores.append({
                'Source': row['Source'],
                'Target': row['Target'],
                'Verdict': verdict,
                'Aggregate_Score': aggregate_score,
                'Is_Corrupted': row.get('Is Corrupted', False),
                'Classification': row.get('Classification', '')
            })
        
        return pd.DataFrame(scores)
    
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        return pd.DataFrame()


def analyze_score_verdict(df: pd.DataFrame, cld_name: str):
    """Analyze relationship between scores and verdicts."""
    
    print(f"\n{'='*80}")
    print(f"SCORE-VERDICT ANALYSIS: {cld_name}")
    print(f"{'='*80}")
    
    df_clean = df[df['Aggregate_Score'].notna()].copy()
    
    if len(df_clean) == 0:
        print("❌ No aggregate scores found")
        return None
    
    print(f"\nTotal edges: {len(df_clean)}")
    
    # Statistics by verdict
    print(f"\n📊 SCORE STATISTICS BY VERDICT:")
    print("-" * 80)
    
    verdict_order = ['INCORRECT', 'PARTIALLY_CORRECT', 'CORRECT']
    available_verdicts = [v for v in verdict_order if v in df_clean['Verdict'].unique()]
    
    for verdict in available_verdicts:
        subset = df_clean[df_clean['Verdict'] == verdict]
        if len(subset) > 0:
            mean_score = subset['Aggregate_Score'].mean()
            median_score = subset['Aggregate_Score'].median()
            std_score = subset['Aggregate_Score'].std()
            count = len(subset)
            
            print(f"{verdict:20s}: n={count:3d}, Mean={mean_score:.3f}, Median={median_score:.3f}, Std={std_score:.3f}")
    
    # Check for perfect consistency
    print(f"\n📈 CONSISTENCY CHECK:")
    print("-" * 80)
    
    # Expected scores for each verdict
    expected_scores = {
        'INCORRECT': 0.0,
        'PARTIALLY_CORRECT': 0.5,
        'CORRECT': 1.0
    }
    
    perfect_consistency = True
    for verdict in available_verdicts:
        subset = df_clean[df_clean['Verdict'] == verdict]
        if len(subset) > 0:
            actual_mean = subset['Aggregate_Score'].mean()
            expected = expected_scores.get(verdict, None)
            
            if expected is not None:
                diff = abs(actual_mean - expected)
                if diff < 0.01:
                    print(f"  ✅ {verdict}: Perfect (expected={expected:.1f}, actual={actual_mean:.3f})")
                else:
                    print(f"  ⚠️  {verdict}: Deviation (expected={expected:.1f}, actual={actual_mean:.3f}, diff={diff:.3f})")
                    perfect_consistency = False
    
    if perfect_consistency:
        print(f"\n✅ PERFECT CONSISTENCY: Scores exactly match verdict categories")
        print(f"   This indicates a deterministic mapping: verdict → score")
    else:
        print(f"\n⚠️  IMPERFECT CONSISTENCY: Scores deviate from expected values")
    
    # Calculate correlation manually (Spearman's rho approximation)
    print(f"\n📊 ORDINAL CORRELATION:")
    print("-" * 80)
    
    verdict_mapping = {'INCORRECT': 0, 'PARTIALLY_CORRECT': 1, 'CORRECT': 2}
    df_clean['Verdict_Ordinal'] = df_clean['Verdict'].map(verdict_mapping)
    
    # Simple correlation calculation
    if len(df_clean) > 1:
        x = df_clean['Verdict_Ordinal'].values
        y = df_clean['Aggregate_Score'].values
        
        # Pearson correlation
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
        
        if denominator > 0:
            correlation = numerator / denominator
            print(f"  Pearson correlation: {correlation:.4f}")
            
            if correlation > 0.9:
                print(f"  ✅ Very strong positive correlation")
            elif correlation > 0.7:
                print(f"  ✅ Strong positive correlation")
            elif correlation > 0.5:
                print(f"  ⚠️  Moderate positive correlation")
            else:
                print(f"  ⚠️  Weak correlation")
    
    # Breakdown by corruption status
    if 'Is_Corrupted' in df_clean.columns:
        print(f"\n📊 SCORES BY CORRUPTION STATUS:")
        print("-" * 80)
        
        for is_corrupted in [True, False]:
            subset = df_clean[df_clean['Is_Corrupted'] == is_corrupted]
            if len(subset) > 0:
                label = "CORRUPTED" if is_corrupted else "CLEAN"
                print(f"\n{label} edges (n={len(subset)}):")
                
                for verdict in available_verdicts:
                    verdict_subset = subset[subset['Verdict'] == verdict]
                    if len(verdict_subset) > 0:
                        mean_score = verdict_subset['Aggregate_Score'].mean()
                        count = len(verdict_subset)
                        pct = 100 * count / len(subset)
                        print(f"  {verdict:20s}: {count:3d} ({pct:5.1f}%) - Mean score: {mean_score:.3f}")
    
    return {
        'cld_name': cld_name,
        'n_edges': len(df_clean),
        'perfect_consistency': perfect_consistency,
        'verdict_counts': df_clean['Verdict'].value_counts().to_dict()
    }


def main():
    """Main execution."""
    
    base_dir = Path("parameter_tuning_experiments/results/rq1_multi_cld_20251105_205040")
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    with open(base_dir / "experiment_summary.json", 'r') as f:
        summary = json.load(f)
    
    print("="*80)
    print("SCORE-VERDICT CORRELATION ANALYSIS")
    print("="*80)
    print("Testing: Are categorical verdicts consistent with continuous scores?")
    print()
    
    all_results = []
    
    for cld_result in summary['results']:
        cld_name = cld_result['cld_name']
        output_dir = Path(cld_result['output_dir'])
        
        judged_files = list(output_dir.glob("judged_no_correction_*.xlsx"))
        
        if not judged_files:
            print(f"❌ No judged file found for {cld_name}")
            continue
        
        df = load_judge_data(judged_files[0])
        
        if df.empty:
            continue
        
        result = analyze_score_verdict(df, cld_name)
        if result:
            all_results.append(result)
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        for result in all_results:
            status = "✅ Perfect" if result['perfect_consistency'] else "⚠️  Imperfect"
            print(f"{result['cld_name']:25s}: {status} (n={result['n_edges']} edges)")
        
        all_perfect = all(r['perfect_consistency'] for r in all_results)
        
        if all_perfect:
            print(f"\n✅ ALL CLDs show perfect score-verdict consistency")
            print(f"   Interpretation: Aggregate scores are deterministically derived from verdicts")
            print(f"   This validates the scoring system coherence")
        else:
            print(f"\n⚠️  Some CLDs show score-verdict deviations")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()



