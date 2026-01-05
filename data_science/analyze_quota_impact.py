#!/usr/bin/env python3
"""
Analyze the impact of OpenAI quota errors on judging completeness
"""
import re
from collections import defaultdict

log_file = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/logs/rq1a_emergency_cot_mechanistic_20251118_051013.log"

# Track sessions by looking for session cloning
sessions = []
current_session = None
current_prompt = None

# Track edge outcomes
edge_outcomes = defaultdict(lambda: {"completed": 0, "no_citation": 0, "failed": 0})

with open(log_file, 'r') as f:
    for line in f:
        # Track prompts
        if "PROMPT 1/2: CITATION COT" in line:
            current_prompt = "cot"
            print(f"Found CoT prompt section")
        elif "PROMPT 2/2: CITATION MECHANISTIC" in line:
            current_prompt = "mechanistic"
            print(f"Found Mechanistic prompt section")
        
        # Track session IDs
        if "Cloned:" in line and "->" in line:
            match = re.search(r'Cloned:.*→\s+([a-f0-9]{8})', line)
            if match:
                session_id = match.group(1)
                if current_session != session_id:
                    if current_session:
                        print(f"  Session {current_session} finished with prompt {current_prompt}")
                    current_session = session_id
                    sessions.append({"id": session_id, "prompt": current_prompt})
                    print(f"New session detected: {session_id} (prompt: {current_prompt})")
        
        # Track successfully completed edges
        if "✓ Completed edge" in line and current_session:
            edge_outcomes[current_session]["completed"] += 1
        
        # Track NO_CITATION edges
        if 'verdict=\'NO_CITATION\'' in line and current_session:
            edge_outcomes[current_session]["no_citation"] += 1
        
        # Track failed edges  
        if "Error single-citation" in line or "Error aggregator" in line:
            if current_session:
                edge_outcomes[current_session]["failed"] += 1

print("\n" + "=" * 80)
print("SESSION ANALYSIS")
print("=" * 80)

for i, sess in enumerate(sessions, 1):
    sid = sess["id"]
    prompt = sess["prompt"]
    outcomes = edge_outcomes[sid]
    total = outcomes["completed"] + outcomes["no_citation"]
    
    print(f"\n{i}. Session: {sid}...")
    print(f"   Prompt: {prompt}")
    print(f"   Successfully judged: {outcomes['completed']}")
    print(f"   NO_CITATION (no refs found): {outcomes['no_citation']}")
    print(f"   Failed (quota errors): {outcomes['failed']}")
    print(f"   Total processed: {total}/1190 ({total/1190*100:.1f}%)")
    
    if total < 1190:
        print(f"   ⚠️  INCOMPLETE: {1190 - total} edges not judged")
    else:
        print(f"   ✅ ALL 1190 EDGES PROCESSED")

print("\n" + "=" * 80)
print(f"Total sessions found: {len(sessions)}")
print("=" * 80)
