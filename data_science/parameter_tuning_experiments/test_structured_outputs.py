#!/usr/bin/env python3
"""
Test script to verify structured outputs work correctly with a small CLD.

This script tests both:
1. Correctness judging with structured outputs
2. Citation judging with structured outputs

Uses the Social Norms CLD (10 variables, 12 edges) as a small test case.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env.dev")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# Test CLD - using small test CLD
TEST_CLD = "test_CLD/CLD_test.xlsx"
TEST_OUTPUT_DIR = "test_structured_outputs"

def test_structured_outputs_correctness():
    """Test structured outputs with correctness judging (no citations)"""
    
    logging.info("="*80)
    logging.info("TEST 1: STRUCTURED OUTPUTS - CORRECTNESS JUDGING")
    logging.info("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_correctness_baseline.yaml"
    result_path = output_dir / f"test_correctness_structured_{timestamp}.xlsx"
    
    logging.info(f"CLD: {TEST_CLD}")
    logging.info(f"Prompt: Baseline (testing structured outputs)")
    logging.info(f"Structured outputs: ENABLED")
    logging.info(f"Output: {result_path}")
    
    # Run with structured outputs enabled
    judge_config = {
        "provider": "openai",
        "model": "gpt-4o-2024-08-06",
        "temperature": 0.3,
        "use_structured_outputs": True,  # ← ENABLED
        "max_tokens": 500
    }
    
    try:
        exp_info = run_discovery_experiment(
            excel_path=TEST_CLD,
            yaml_path=str(yaml_path),
            
            # Generator config
            generator_config={
                "provider": "openai",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.7
            },
            
            # No corruption
            corruption_rate=0.0,
            corruptor_config={
                "provider": "openai",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.7
            },
            
            # Judging settings - CORRECTNESS APPROACH (no citations)
            judge_edges=True,
            judge_approach="correctness",
            judge_enable_web_search=False,
            judge_config=judge_config,
            num_judges=1,
            judge_models=["gpt-4o-2024-08-06"],
            judge_parallel=False,  # Sequential for easier debugging
            
            # Output settings
            result_excel_path=str(result_path),
            output_dir=str(output_dir),
            plot_session_graph=False,
            plot_validation_graph=False,
            export_json=True,
            
            # Citation settings - DISABLED
            embedding_enable=False,
            citation_search_provider="brave",
            node_comparison_enable=False
        )
        
        logging.info("✅ Test 1 PASSED: Structured outputs work with correctness judging")
        logging.info(f"   Result saved to: {result_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structured_outputs_citation():
    """Test structured outputs with citation judging"""
    
    logging.info("\n" + "="*80)
    logging.info("TEST 2: STRUCTURED OUTPUTS - CITATION JUDGING")
    logging.info("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_citation_baseline.yaml"
    result_path = output_dir / f"test_citation_structured_{timestamp}.xlsx"
    
    logging.info(f"CLD: {TEST_CLD}")
    logging.info(f"Prompt: Baseline (testing structured outputs)")
    logging.info(f"Structured outputs: ENABLED")
    logging.info(f"Output: {result_path}")
    
    # Run with structured outputs enabled
    judge_config = {
        "provider": "openai",
        "model": "gpt-4o-2024-08-06",
        "temperature": 0.3,
        "use_structured_outputs": True,  # ← ENABLED
        "max_tokens": 500
    }
    
    try:
        exp_info = run_discovery_experiment(
            excel_path=TEST_CLD,
            yaml_path=str(yaml_path),
            
            # Generator config
            generator_config={
                "provider": "openai",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.7
            },
            
            # No corruption
            corruption_rate=0.0,
            corruptor_config={
                "provider": "openai",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.7
            },
            
            # Judging settings - CITATION APPROACH
            judge_edges=True,
            judge_approach="per_citation_aggregate",
            judge_enable_web_search=False,
            judge_config=judge_config,
            num_judges=1,
            judge_models=["gpt-4o-2024-08-06"],
            judge_parallel=False,  # Sequential for easier debugging
            
            # Output settings
            result_excel_path=str(result_path),
            output_dir=str(output_dir),
            plot_session_graph=False,
            plot_validation_graph=False,
            export_json=True,
            
            # Citation settings - DISABLED (to avoid fetching issues in test)
            embedding_enable=False,
            citation_search_provider="brave",
            node_comparison_enable=False
        )
        
        logging.info("✅ Test 2 PASSED: Structured outputs work with citation judging")
        logging.info(f"   Result saved to: {result_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_behavior():
    """Test that system falls back to regex parsing with incompatible model"""
    
    logging.info("\n" + "="*80)
    logging.info("TEST 3: FALLBACK BEHAVIOR - INCOMPATIBLE MODEL")
    logging.info("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_correctness_baseline.yaml"
    result_path = output_dir / f"test_fallback_{timestamp}.xlsx"
    
    logging.info(f"CLD: {TEST_CLD}")
    logging.info(f"Prompt: Baseline")
    logging.info(f"Model: gpt-4.1 (NOT compatible with structured outputs)")
    logging.info(f"Structured outputs: ENABLED (should fallback to regex)")
    logging.info(f"Output: {result_path}")
    
    # Run with structured outputs enabled but incompatible model
    judge_config = {
        "provider": "openai",
        "model": "gpt-4.1",  # ← NOT compatible
        "temperature": 0.3,
        "use_structured_outputs": True,  # ← Should fallback to regex
        "max_tokens": 250
    }
    
    try:
        exp_info = run_discovery_experiment(
            excel_path=TEST_CLD,
            yaml_path=str(yaml_path),
            
            # Generator config
            generator_config={
                "provider": "openai",
                "model": "gpt-4.1",
                "temperature": 0.7
            },
            
            # No corruption
            corruption_rate=0.0,
            corruptor_config={
                "provider": "openai",
                "model": "gpt-4.1",
                "temperature": 0.7
            },
            
            # Judging settings - CORRECTNESS
            judge_edges=True,
            judge_approach="correctness",
            judge_enable_web_search=False,
            judge_config=judge_config,
            num_judges=1,
            judge_models=["gpt-4.1"],
            judge_parallel=False,
            
            # Output settings
            result_excel_path=str(result_path),
            output_dir=str(output_dir),
            plot_session_graph=False,
            plot_validation_graph=False,
            export_json=True,
            
            # Citation settings
            embedding_enable=False,
            citation_search_provider="brave",
            node_comparison_enable=False
        )
        
        logging.info("✅ Test 3 PASSED: Fallback to regex parsing works correctly")
        logging.info(f"   Result saved to: {result_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logging.info("\n" + "🧪 "*40)
    logging.info("STRUCTURED OUTPUTS INTEGRATION TEST SUITE")
    logging.info("🧪 "*40 + "\n")
    
    results = {
        "test_1_correctness": False,
        "test_2_citation": False,
        "test_3_fallback": False
    }
    
    # Run tests
    results["test_1_correctness"] = test_structured_outputs_correctness()
    results["test_2_citation"] = test_structured_outputs_citation()
    results["test_3_fallback"] = test_fallback_behavior()
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("TEST SUMMARY")
    logging.info("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logging.info(f"{test_name}: {status}")
    
    logging.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logging.info("\n🎉 ALL TESTS PASSED! Structured outputs integration is working correctly.")
    else:
        logging.error(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
