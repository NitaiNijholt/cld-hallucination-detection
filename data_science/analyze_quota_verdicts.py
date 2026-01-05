#!/usr/bin/env python3
"""
Analyze what happened to edges that encountered quota errors
"""
import re

log_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/logs/rq1a_emergency_cot_mechanistic_20251118_051013.log"

# Track edges that had quota errors
edges_with_quota_errors = set()
edges_completed = set()
edges_updated = set()

with open(log_file, 'r') as f:
    current_edge = None
    lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Track which edge is being processed
        if "CITATION JUDGING: Edge" in line:
            match = re.search(r'Edge (\d+)/1190', line)
            if match:
                current_edge = int(match.group(1))
        
        # Check if quota error occurred
        if "insufficient_quota" in line and current_edge:
            edges_with_quota_errors.add(current_edge)
        
        # Check if edge was completed
        if "✓ Completed edge" in line:
            match = re.search(r'edge (\d+)/1190', line)
            if match:
                edges_completed.add(int(match.group(1)))
        
        # Check if edge was updated with verdict
        if "UPDATED EDGE:" in line:
            # Look back a few lines to find which edge number this was
            for j in range(max(0, i-20), i):
                prev_line = lines[j]
                if "CITATION JUDGING: Edge" in prev_line:
                    match = re.search(r'Edge (\d+)/1190', prev_line)
                    if match:
                        edges_updated.add(int(match.group(1)))
                        break

print("=" * 80)
print("QUOTA ERROR IMPACT ANALYSIS")
print("=" * 80)

print(f"\nTotal edges with quota errors: {len(edges_with_quota_errors)}")
print(f"Total edges completed: {len(edges_completed)}")
print(f"Total edges updated: {len(edges_updated)}")

# Find edges that had quota errors but were still completed
quota_but_completed = edges_with_quota_errors & edges_completed
quota_but_updated = edges_with_quota_errors & edges_updated

print(f"\nEdges with quota errors that were still completed: {len(quota_but_completed)}")
print(f"Edges with quota errors that were still updated: {len(quota_but_updated)}")

# Find edges that had quota errors and were NOT completed
quota_and_not_completed = edges_with_quota_errors - edges_completed
quota_and_not_updated = edges_with_quota_errors - edges_updated

print(f"\n⚠️  Edges with quota errors that were NOT completed: {len(quota_and_not_completed)}")
print(f"⚠️  Edges with quota errors that were NOT updated: {len(quota_and_not_updated)}")

if quota_and_not_completed:
    print(f"\nEdge numbers NOT completed: {sorted(list(quota_and_not_completed))[:20]} ...")
if quota_and_not_updated:
    print(f"Edge numbers NOT updated: {sorted(list(quota_and_not_updated))[:20]} ...")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)

if len(quota_and_not_completed) > 0:
    print("❌ Some edges encountered quota errors and were NEVER completed")
    print("   These edges likely have NO_CITATION or null verdicts in Neo4j")
else:
    print("✅ All edges that encountered quota errors were eventually completed")
    print("   The system likely has retry logic or the quota was temporarily exhausted")
    print("   but then refreshed (rate limit vs hard quota)")

print("=" * 80)
