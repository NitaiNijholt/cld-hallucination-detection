#!/usr/bin/env python3
"""
Recalculate summary statistics from existing judge results
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results"

# Load the latest full results
results_file = OUTPUT_DIR / "judge_citation_providers_full_20251106_061449.json"

print(f"Loading: {results_file}")
with open(results_file, 'r') as f:
    results = json.load(f)

print(f"\n{'='*80}")
print("RECALCULATING SUMMARIES")
print('='*80)

# Recalculate for each provider
summary_data = []
for result in results:
    provider = result['provider']
    session_id = result['session_id']
    judged_edges = result.get('judged_edges', [])
    
    print(f"\n{provider}:")
    print(f"  Session: {session_id}")
    print(f"  Total edges: {len(judged_edges)}")
    
    # Extract aggregate scores (skip NO_CITATION edges)
    aggregate_scores = []
    verdicts = []
    
    for edge in judged_edges:
        score = edge.get("aggregate_score")
        if score is not None:  # Only include edges with scores (excludes NO_CITATION)
            aggregate_scores.append(score)
            verdicts.append(edge.get("aggregate_verdict", "unknown"))
    
    # Calculate stats
    avg_score = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0
    verdict_counts = Counter(verdicts)
    
    print(f"  Edges with citations: {len(aggregate_scores)}")
    print(f"  Average score: {avg_score:.3f}")
    print(f"  Verdict breakdown:")
    for verdict, count in verdict_counts.most_common():
        pct = count / len(verdicts) * 100 if verdicts else 0
        print(f"    - {verdict}: {count} ({pct:.1f}%)")
    
    # Update result
    result['average_aggregate_score'] = avg_score
    result['verdict_counts'] = dict(verdict_counts)
    
    # Add to summary
    verdict_breakdown = " | ".join([f"{k}: {v}" for k, v in verdict_counts.items()])
    summary_data.append({
        "Provider": provider,
        "Session ID": session_id[:8] + "...",
        "Edges Judged": len(judged_edges),
        "Edges w/ Citations": len(aggregate_scores),
        "Avg Score": f"{avg_score:.3f}",
        "Duration (s)": f"{result['duration_seconds']:.1f}",
        "Verdict Breakdown": verdict_breakdown
    })

# Save updated results
output_file = OUTPUT_DIR / "judge_citation_providers_full_CORRECTED.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Updated full results saved to: {output_file}")

# Save summary
df = pd.DataFrame(summary_data)
summary_file = OUTPUT_DIR / "judge_citation_providers_summary_CORRECTED.csv"
df.to_csv(summary_file, index=False)
print(f"💾 Summary saved to: {summary_file}")

print(f"\n{'='*80}")
print("CORRECTED SUMMARY")
print('='*80)
print()
print(df.to_string(index=False))
print('='*80)



