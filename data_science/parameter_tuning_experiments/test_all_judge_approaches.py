#!/usr/bin/env python3
"""
Comprehensive test to verify inference time tracking works for all judging approaches.
Tests all 4 approaches: correctness, all_citations_at_once, per_citation_aggregate, one_causal_source_found
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment
load_dotenv(Path(__file__).parent.parent.parent / ".env.dev")

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# Test parameters - using small test CLD
SESSION_ID = "6efe8a11-920e-41c1-83c6-e1c4bbfb03e0"  # test CLD
EXCEL_FILE = "test_CLD/CLD_test.xlsx"

# Config - we don't need generator/corruptor, just judge
DUMMY_CONFIG = {"provider": "openai", "model": "gpt-4o-mini"}
JUDGE_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",  # Fast model for testing
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": 1500
}

# All judging approaches to test
APPROACHES = [
    "correctness",
    "all_citations_at_once",
    "per_citation_aggregate",
    "one_causal_source_found"
]

def test_approach(approach_name: str) -> dict:
    """Test a single judging approach and return stats."""
    print(f"\n{'='*80}")
    print(f"TESTING APPROACH: {approach_name}")
    print(f"{'='*80}")
    
    try:
        # Run experiment with this approach
        result = run_discovery_experiment(
            retrieved_session_id=SESSION_ID,
            excel_path=EXCEL_FILE,
            yaml_path="../../backend/configs/prompts.yaml",
            generator_config=DUMMY_CONFIG,
            corruptor_config=DUMMY_CONFIG,
            judge_config=JUDGE_CONFIG,
            judge_edges=True,
            judge_approach=approach_name,
            judge_models=[JUDGE_CONFIG['model']],
            num_judges=1,
            judge_parallel=False,
            judge_enable_web_search=True,
            run_correction=False
        )
        
        # Extract judge stats
        judge_stats = result['lm_stats']['judge']
        inference_time = judge_stats.get('inference_time_total', 0.0)
        inference_calls = judge_stats.get('inference_call_count', 0)
        token_totals = judge_stats.get('token_totals', {})
        status_counts = judge_stats.get('status_counts', {})
        
        success = inference_time > 0 and inference_calls > 0
        
        return {
            'approach': approach_name,
            'success': success,
            'inference_time': inference_time,
            'inference_calls': inference_calls,
            'total_tokens': token_totals.get('total_tokens', 0),
            'status_codes': status_counts,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            'approach': approach_name,
            'success': False,
            'inference_time': 0.0,
            'inference_calls': 0,
            'total_tokens': 0,
            'status_codes': {},
            'error': str(e)
        }

def main():
    print("="*80)
    print("INFERENCE TIME TRACKING VERIFICATION - ALL JUDGING APPROACHES")
    print("="*80)
    print(f"Session ID: {SESSION_ID}")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Judge model: {JUDGE_CONFIG['model']}")
    print(f"Approaches to test: {len(APPROACHES)}")
    print("="*80)
    
    # Test each approach
    results = []
    for approach in APPROACHES:
        result = test_approach(approach)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL APPROACHES")
    print("="*80)
    
    print(f"\n{'Approach':<30} {'Status':<10} {'Time (s)':<12} {'Calls':<8} {'Tokens':<10}")
    print("-" * 80)
    
    all_passed = True
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        if not r['success']:
            all_passed = False
        
        time_str = f"{r['inference_time']:.2f}" if r['inference_time'] > 0 else "0.00"
        
        print(f"{r['approach']:<30} {status:<10} {time_str:<12} {r['inference_calls']:<8} {r['total_tokens']:<10}")
        
        if r['error']:
            print(f"  Error: {r['error']}")
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ SUCCESS: All approaches track inference time correctly!")
    else:
        print("❌ FAILURE: Some approaches failed to track inference time")
        failed = [r['approach'] for r in results if not r['success']]
        print(f"Failed approaches: {', '.join(failed)}")
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
