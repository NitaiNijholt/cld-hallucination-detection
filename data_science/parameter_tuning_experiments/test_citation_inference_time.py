#!/usr/bin/env python3
"""
Quick test to verify inference time tracking in citation judging.
Tests with just a few edges from social_norms CLD.
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
    level=logging.INFO,
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

print("="*80)
print("CITATION JUDGING INFERENCE TIME TEST")
print("="*80)
print(f"Session ID: {SESSION_ID}")
print(f"Excel file: {EXCEL_FILE}")
print(f"Judge model: {JUDGE_CONFIG['model']}")
print("="*80)

# Run experiment with citation judging
result = run_discovery_experiment(
    retrieved_session_id=SESSION_ID,
    excel_path=EXCEL_FILE,
    yaml_path="../../backend/configs/prompts.yaml",  # Correct path from test location
    generator_config=DUMMY_CONFIG,  # Not used, but required
    corruptor_config=DUMMY_CONFIG,  # Not used, but required
    judge_config=JUDGE_CONFIG,
    judge_edges=True,  # Enable judging
    judge_approach="per_citation_aggregate",  # Use per citation aggregate judging
    judge_models=[JUDGE_CONFIG['model']],  # Single judge for speed
    num_judges=1,
    judge_parallel=False,  # Sequential for easier debugging
    judge_enable_web_search=True,
    run_correction=False
)

print("\n" + "="*80)
print("TEST COMPLETE - RESULTS")
print("="*80)

# Extract and display judge stats
judge_stats = result['lm_stats']['judge']
print(f"\nJudge Statistics:")
print(f"  Status codes: {judge_stats.get('status_counts', {})}")
print(f"  Token totals: {judge_stats.get('token_totals', {})}")
print(f"  Inference time: {judge_stats.get('inference_time_total', 0.0):.2f} seconds")
print(f"  Inference calls: {judge_stats.get('inference_call_count', 0)}")

# Check if inference time is zero (the bug)
inference_time = judge_stats.get('inference_time_total', 0.0)
if inference_time == 0.0:
    print("\n❌ BUG CONFIRMED: Inference time is 0.0!")
    print("   Check the DEBUG output above to see where stats are lost.")
else:
    print(f"\n✅ SUCCESS: Inference time tracked correctly: {inference_time:.2f}s")

print("\n" + "="*80)
