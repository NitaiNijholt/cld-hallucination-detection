#!/usr/bin/env python3
"""
Compare Jina AI vs ContentScraper citation fetching results
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Result files
RESULTS_DIR = Path(__file__).parent / "results"
JINA_FILE = RESULTS_DIR / "judge_citation_providers_full_20251113_163644.json"
SCRAPER_FILE = RESULTS_DIR / "judge_citation_providers_full_20251113_154250.json"
OUTPUT_FILE = RESULTS_DIR / "jina_vs_scraper_comparison_brave.xlsx"

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[0]  # First element contains the results

def analyze_results(results, name):
    """Analyze results and return statistics"""
    edges = results['judged_edges']
    
    # Count edges with citations fetched (not NO_CITATION or None)
    # Note: Some edges may have verdict='None' (string) or None (null) or empty string
    edges_with_citations = sum(1 for e in edges 
                              if e.get('aggregate_verdict') 
                              and e.get('aggregate_verdict') not in ['NO_CITATION', 'None', ''])
    citation_coverage = edges_with_citations / len(edges) * 100 if edges else 0
    
    # Get scores (excluding NO_CITATION/None)
    scores = [e.get('aggregate_score') for e in edges 
             if e.get('aggregate_score') is not None 
             and e.get('aggregate_verdict') 
             and e.get('aggregate_verdict') not in ['NO_CITATION', 'None', '']]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Get verdicts (excluding NO_CITATION/None)
    verdicts = [e.get('aggregate_verdict') for e in edges 
               if e.get('aggregate_verdict') 
               and e.get('aggregate_verdict') not in ['NO_CITATION', 'None', '']]
    verdict_counts = Counter(verdicts)
    
    # Count NO_CITATION (including None and empty)
    no_citation_count = sum(1 for e in edges 
                           if not e.get('aggregate_verdict') 
                           or e.get('aggregate_verdict') in ['NO_CITATION', 'None', ''])
    
    stats = {
        'name': name,
        'total_edges': len(edges),
        'edges_with_citations': edges_with_citations,
        'citation_coverage_pct': citation_coverage,
        'no_citation_count': no_citation_count,
        'avg_score': avg_score,
        'duration_seconds': results['duration_seconds'],
        'verdict_counts': verdict_counts,
        'scores': scores,
        'edges': edges
    }
    
    return stats

def create_comparison(jina_stats, scraper_stats):
    """Create comparison tables"""
    
    # Overall comparison
    overall = pd.DataFrame({
        'Metric': [
            'Total Edges',
            'Edges with Citations',
            'Citation Coverage %',
            'NO_CITATION Count',
            'Average Score',
            'Duration (seconds)',
            'Duration (minutes)'
        ],
        'Jina AI': [
            jina_stats['total_edges'],
            jina_stats['edges_with_citations'],
            f"{jina_stats['citation_coverage_pct']:.1f}%",
            jina_stats['no_citation_count'],
            f"{jina_stats['avg_score']:.3f}",
            f"{jina_stats['duration_seconds']:.1f}",
            f"{jina_stats['duration_seconds']/60:.1f}"
        ],
        'ContentScraper': [
            scraper_stats['total_edges'],
            scraper_stats['edges_with_citations'],
            f"{scraper_stats['citation_coverage_pct']:.1f}%",
            scraper_stats['no_citation_count'],
            f"{scraper_stats['avg_score']:.3f}",
            f"{scraper_stats['duration_seconds']:.1f}",
            f"{scraper_stats['duration_seconds']/60:.1f}"
        ]
    })
    
    # Verdict comparison
    all_verdicts = set(list(jina_stats['verdict_counts'].keys()) + list(scraper_stats['verdict_counts'].keys()))
    verdict_data = []
    for verdict in sorted(all_verdicts):
        jina_count = jina_stats['verdict_counts'].get(verdict, 0)
        scraper_count = scraper_stats['verdict_counts'].get(verdict, 0)
        jina_pct = jina_count / jina_stats['edges_with_citations'] * 100 if jina_stats['edges_with_citations'] else 0
        scraper_pct = scraper_count / scraper_stats['edges_with_citations'] * 100 if scraper_stats['edges_with_citations'] else 0
        
        verdict_data.append({
            'Verdict': verdict,
            'Jina AI Count': jina_count,
            'Jina AI %': f"{jina_pct:.1f}%",
            'ContentScraper Count': scraper_count,
            'ContentScraper %': f"{scraper_pct:.1f}%"
        })
    
    verdicts_df = pd.DataFrame(verdict_data)
    
    # Edge-by-edge comparison
    edge_comparison = []
    for i, (jina_edge, scraper_edge) in enumerate(zip(jina_stats['edges'], scraper_stats['edges'])):
        edge_comparison.append({
            'Edge #': i + 1,
            'Source': jina_edge.get('source', ''),
            'Polarity': jina_edge.get('polarity', ''),
            'Target': jina_edge.get('target', ''),
            'Jina Verdict': jina_edge.get('aggregate_verdict', ''),
            'Jina Score': jina_edge.get('aggregate_score', ''),
            'Scraper Verdict': scraper_edge.get('aggregate_verdict', ''),
            'Scraper Score': scraper_edge.get('aggregate_score', ''),
            'Same Verdict': jina_edge.get('aggregate_verdict') == scraper_edge.get('aggregate_verdict'),
            'Score Diff': abs((jina_edge.get('aggregate_score') or 0) - (scraper_edge.get('aggregate_score') or 0))
        })
    
    edges_df = pd.DataFrame(edge_comparison)
    
    return overall, verdicts_df, edges_df

def main():
    print("Loading results...")
    jina_results = load_results(JINA_FILE)
    scraper_results = load_results(SCRAPER_FILE)
    
    print("Analyzing Jina AI results...")
    jina_stats = analyze_results(jina_results, "Jina AI")
    
    print("Analyzing ContentScraper results...")
    scraper_stats = analyze_results(scraper_results, "ContentScraper")
    
    print("Creating comparison...")
    overall_df, verdicts_df, edges_df = create_comparison(jina_stats, scraper_stats)
    
    print("\n" + "="*80)
    print("JINA AI vs CONTENTSCRAPER COMPARISON (Brave Fixed Provider)")
    print("="*80)
    
    print("\n📊 OVERALL COMPARISON:")
    print(overall_df.to_string(index=False))
    
    print("\n📋 VERDICT BREAKDOWN:")
    print(verdicts_df.to_string(index=False))
    
    # Calculate agreement
    same_verdict = edges_df['Same Verdict'].sum()
    agreement_pct = same_verdict / len(edges_df) * 100
    print(f"\n🤝 AGREEMENT: {same_verdict}/{len(edges_df)} edges ({agreement_pct:.1f}%) have the same verdict")
    
    # Save to Excel
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        overall_df.to_excel(writer, sheet_name='Overall Comparison', index=False)
        verdicts_df.to_excel(writer, sheet_name='Verdict Breakdown', index=False)
        edges_df.to_excel(writer, sheet_name='Edge-by-Edge Comparison', index=False)
    
    print(f"✅ Comparison saved to: {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    main()

