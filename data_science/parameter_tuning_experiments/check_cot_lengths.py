#!/usr/bin/env python3
"""
Check actual CoT response lengths to understand truncation issue
"""
import pandas as pd
import json

df = pd.read_excel('final_runs/RQ1a_judging_correctness_GT_depresive/run_1/judged_Depressive_symptoms_in_response_to_a_stressor_chain_of_thought_20251114_001142.xlsx')

# Get successful judgments (non-ERROR)
successful = df[df['Judge Verdict'] != 'ERROR']

print("=" * 80)
print(f"ANALYZING {len(successful)} SUCCESSFUL CoT JUDGMENTS")
print("=" * 80)

# Extract response lengths from Judge Message
response_lengths = []
for idx, row in successful.head(10).iterrows():
    msg_str = row['Judge Message']
    try:
        msg = json.loads(msg_str)
        reason = msg['judge_results'][0]['reason']
        response_lengths.append(len(reason))
        print(f"\nEdge: {row['Source'][:30]}... -> {row['Target'][:30]}...")
        print(f"Response length: {len(reason)} chars")
        print(f"Verdict: {msg['judge_results'][0]['verdict']}")
        # Show first 200 chars of reason
        print(f"Reason preview: {reason[:200]}...")
    except Exception as e:
        print(f"Error parsing: {e}")

if response_lengths:
    print(f"\n" + "=" * 80)
    print(f"RESPONSE LENGTH STATISTICS (successful CoT)")
    print("=" * 80)
    print(f"Min: {min(response_lengths)} chars")
    print(f"Max: {max(response_lengths)} chars")
    print(f"Avg: {sum(response_lengths)/len(response_lengths):.0f} chars")
    print(f"\nNote: ~4 chars per token, so {max(response_lengths)} chars ≈ {max(response_lengths)/4:.0f} tokens")
