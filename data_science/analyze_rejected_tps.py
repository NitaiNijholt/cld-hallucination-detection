#!/usr/bin/env python3
"""Analyze why TP edges were rejected by the Deep Research system."""

import json

with open('deep_research_test_results.json', 'r') as f:
    results = json.load(f)

# Find rejected TPs
rejected_tps = [r for r in results if r['classification'] == 'TP' and r['verdict'] == 'unsupported']

print("="*100)
print("ANALYSIS: WHY WERE THESE TRUE POSITIVE EDGES REJECTED?")
print("="*100)

for i, edge in enumerate(rejected_tps, 1):
    print(f"\n{'#'*100}")
    print(f"REJECTED TP #{i}: {edge['source']} → {edge['target']}")
    print(f"{'#'*100}")
    
    print(f"\n📋 BASIC INFO:")
    print(f"  Verdict: {edge['verdict'].upper()}")
    print(f"  Confidence: {edge['confidence']}/10")
    print(f"  Direct Causal Found: {edge.get('direct_causal_found', False)}")
    print(f"  Expected: {edge.get('expected', 'N/A')}")
    print(f"  Duration: {edge.get('duration', 'N/A')}")
    
    print(f"\n🔬 STRONGEST CAUSAL EVIDENCE ANALYSIS:")
    print(f"  Confidence: {edge.get('strongest_causal_evidence_confidence', 'N/A')}/10")
    print(f"  Study Type: {edge.get('strongest_causal_evidence_study_type', 'N/A')}")
    print(f"  URL: {edge.get('strongest_causal_evidence_url', 'N/A')}")
    
    print(f"\n  📝 Cited Passage:")
    passage = edge.get('strongest_causal_evidence_passage', 'N/A')
    for line in passage.split('. '):
        print(f"    {line}.")
    
    print(f"\n  💡 Summary:")
    summary = edge.get('strongest_causal_evidence_summary', 'N/A')
    # Wrap at 90 chars
    words = summary.split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 94:
            print(line)
            line = "    " + word
        else:
            line += " " + word if line != "    " else word
    if line.strip():
        print(line)
    
    print(f"\n🧠 FULL JUDGE REASONING:")
    print(f"{'─'*100}")
    reasoning = edge.get('reasoning', 'No reasoning available')
    # Word wrap at 100 chars
    words = reasoning.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 100:
            print(line)
            line = word
        else:
            line += " " + word if line else word
    if line:
        print(line)
    print(f"{'─'*100}")
    
    print(f"\n📚 EVIDENCE SOURCES ({edge.get('evidence_count', 0)} total):")
    for j, url in enumerate(edge.get('evidence_urls', [])[:8], 1):
        print(f"  {j}. {url}")
    if edge.get('evidence_count', 0) > 8:
        print(f"  ... and {edge.get('evidence_count', 0) - 8} more")
    
    print(f"\n{'='*100}")
    print(f"KEY INSIGHT for TP #{i}:")
    print(f"{'='*100}")
    
    # Analyze what went wrong
    if "REVERSE" in summary.upper() or "reverse" in reasoning.lower():
        print("🔴 PROBLEM: System found evidence for REVERSE causality (Target → Source)")
        print("   The claim may be stated backwards, or the system is confusing bidirectional relationships.")
    
    if "compositional" in reasoning.lower() or "mathematical" in reasoning.lower():
        print("🔴 PROBLEM: System rejected compositional/definitional relationships as 'not causal'")
        print("   The system may be too strict - refusing to accept that components cause wholes.")
    
    if "NOT associated" in passage or "no association" in passage.lower():
        print("🔴 PROBLEM: System found studies showing NO association")
        print("   The system may have found confounding or null results that weakened the claim.")
    
    if edge.get('direct_causal_found') == False:
        print("🔴 PROBLEM: No direct causal evidence detected")
        print("   The causal evidence detector didn't find RCTs/experiments for this relationship.")

print("\n" + "="*100)
print("SUMMARY & RECOMMENDATIONS")
print("="*100)
print("\n1. The system appears to be CORRECTLY identifying that some claimed relationships")
print("   may actually operate in reverse (e.g., body composition → BMR, not BMR → TDEE).")
print("\n2. The system is being VERY STRICT about what counts as 'causal evidence',")
print("   potentially rejecting compositional/definitional relationships.")
print("\n3. Consider whether the TEST EDGES themselves are correctly specified:")
print("   - Are they truly causal or just correlational/compositional?")
print("   - Is the causal direction correctly specified?")
print("\n4. The system's reasoning appears SOUND - it's finding contradictory evidence")
print("   and correctly identifying when causation flows the opposite direction.")

