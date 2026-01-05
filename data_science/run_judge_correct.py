#!/usr/bin/env python3
"""
Run Judge -> Correct Pipeline (Single Session)

This script performs a complete judge -> correct cycle on a specific session ID.
1. Judges all edges in the session using the specified model
2. Runs the corrector on the judged edges
3. Reports detailed outcomes

Usage:
    python run_judge_correct.py <session_id> [model]

Example:
    python run_judge_correct.py 1234-5678-90ab gpt-4o
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent))

from modules import CausalDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_judge_correct.py <session_id> [model]")
        return 1
    
    session_id = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1"
    
    print("="*80)
    print(f"🧪 RUNNING JUDGE -> CORRECT ON SESSION: {session_id}")
    print(f"🤖 Model: {model}")
    print("="*80)
    
    # 1. Initialize CausalDiscovery
    # Using mechanistic prompt as default for judging
    prompts_path = Path("data_science/parameter_tuning_experiments/alternative_prompts/prompts_citation_mechanistic_lit.yaml")
    if not prompts_path.exists():
        # Fallback for testing if file moved
        prompts_path = Path("parameter_tuning_experiments/alternative_prompts/prompts_citation_mechanistic_lit.yaml")
        
    try:
        discovery = CausalDiscovery(
            target_variable="Unknown", 
            temporal_scale="Unknown", 
            spatial_scale="Unknown",
            yaml_path=str(prompts_path),
            dev_mode=False,
            # Set models via config dicts
            generator_config={"provider": "openai", "model": model},
            corruptor_config={"provider": "openai", "model": model},
            judge_config={"provider": "openai", "model": model}
        )
        
        # Set session ID manually
        discovery.session_id = session_id
        
        # 2. Judge All Edges
        print("\n🔍 STARTING JUDGING PHASE...")
        # Using basic judging first to ensure we get verdicts
        # Note: In a real run, you might use judge_all_edges_with_citations_serial
        judged_edges = discovery.judge_all_edges_serial(
            judge_models=[model],
            num_judges=1
        )
        print(f"✅ Judged {len(judged_edges)} edges")
        
        # 3. Run Correction
        print("\n🚀 STARTING CORRECTION PHASE...")
        # This will pick up the edges we just judged as INCORRECT
        outcomes = discovery.correct_edges_serial(
            corrector_models=[model],
            max_rounds=1,
            action_order=("revise", "recite", "remove"),
            rejudge_after=False, # Just want to see corrections
            use_simple_corrector=False,
            corrector_prompt_variant="original" 
        )
        
        # 4. Output Results
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print("="*80)
        
        corrected_count = sum(1 for o in outcomes if o.get("corrected"))
        print(f"Total actions attempted: {len(outcomes)}")
        print(f"Successfully corrected edges: {corrected_count}")
        
        # Save detailed log
        output_file = f"judge_correct_outcomes_{session_id}.json"
        with open(output_file, "w") as f:
            json.dump(outcomes, f, indent=2)
        print(f"Detailed outcomes saved to: {output_file}")
        
        # Print breakdown
        actions = {}
        for o in outcomes:
            act = o.get("action")
            actions[act] = actions.get(act, 0) + 1
            
        print("\nAction Breakdown:")
        for act, count in actions.items():
            print(f"  - {act}: {count}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
