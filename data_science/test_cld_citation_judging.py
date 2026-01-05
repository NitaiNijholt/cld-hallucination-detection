#!/usr/bin/env python3
"""
Test CLD Citation Judging - Following Ulemans Methodology

This script tests the 3 ground truth CLDs (Social Norms, Depressive Symptoms, 
Emergency Department) using citation-based judging similar to the Ulemans 
Alzheimer's CLD approach.

The process:
1. Load ground truth CLD from Excel
2. Create Neo4j session with edges from the Excel file
3. Fill citations using Brave Search
4. Run citation-based judging with per_citation_aggregate approach
5. Save results for analysis

This tests whether the judge can validate edges that are from published CLDs.

Usage:
    nohup python test_cld_citation_judging.py > logs/cld_citation_test_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    tail -f logs/cld_citation_test_*.log
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from modules import CausalDiscovery, run_discovery_experiment
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ground truth CLD configurations
GT_CLDS = {
    "Social_norms_and_obesity_prevalence": {
        "excel_path": "ground_truth_CLDs/Social_norms_and_obesity_prevalence.xlsx",
        "target_variable": "Group-level BMI",
        "temporal_scale": "Long-term social dynamics",
        "spatial_scale": "Sociocultural groups",
        "source_paper": {
            "doi": "10.1111/obr.13044",
            "citation": "Crielaard et al. (2020) Obesity Reviews"
        }
    },
    "Depressive_symptoms_in_response_to_a_stressor": {
        "excel_path": "ground_truth_CLDs/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
        "target_variable": "Depressive symptoms (IDS-SR)",
        "temporal_scale": "Response to stressors over weeks to months",
        "spatial_scale": "Healthy adults",
        "source_paper": {
            "doi": "10.1038/s44260-024-00017-9",
            "citation": "Uleman et al. (2024) npj Complexity"
        }
    },
    "Emergency_department": {
        "excel_path": "ground_truth_CLDs/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx",
        "target_variable": "Acute care demand leading to ED visit",
        "temporal_scale": "Healthcare decision-making timeline",
        "spatial_scale": "Older persons (65+) in Amsterdam",
        "source_paper": {
            "doi": "10.1007/s41999-023-00816-8",
            "citation": "Smeekes et al. (2023) European Geriatric Medicine"
        }
    }
}

# Output directory
OUTPUT_DIR = Path(__file__).parent / "cld_citation_test_results"


def test_single_cld(cld_name: str, config: dict, run_idx: int = 1) -> dict:
    """
    Test a single CLD with citation-based judging.
    
    Args:
        cld_name: Name of the CLD
        config: Configuration dictionary for the CLD
        run_idx: Run index for multi-run experiments
        
    Returns:
        Dictionary with test results
    """
    logger.info("=" * 80)
    logger.info(f"TESTING CLD: {cld_name} (Run {run_idx})")
    logger.info("=" * 80)
    logger.info(f"Source paper: {config['source_paper']['citation']}")
    
    start_time = time.time()
    
    try:
        # Run discovery experiment with citation judging
        experiment_info = run_discovery_experiment(
            # CLD Configuration
            excel_path=config["excel_path"],
            yaml_path="parameter_tuning_experiments/prompts_Nitai_C.yaml",
            
            # Target context
            overide_target_variable=config["target_variable"],
            
            # Generator config
            generator_config={"provider": "openai", "model": "gpt-4.1"},
            generator_temperature=0.7,
            
            # Corruptor config (required even if not corrupting)
            corruptor_config={"provider": "openai", "model": "gpt-4.1"},
            
            # Citation search
            citation_search_provider="brave",  # Use Brave Search like Ulemans
            
            # Judge config - citation mode
            judge_config={"provider": "openai", "model": "gpt-4.1"},
            judge_models=["gpt-4.1"],  # Required for judging
            num_judges=1,
            judge_edges=True,
            judge_approach="per_citation_aggregate",  # Like Ulemans
            judge_temperature=0.3,
            judge_parallel=True,
            judge_max_workers=5,
            
            # Skip corruption (we're testing ground truth)
            corruption_rate=0.0,
            
            # Skip node generation (use Excel nodes)
            node_generation_only=False,
            
            # CI metrics
            embedding_enable=True,
            ci_compute_embeddings=True,
            
            # Output
            output_dir=str(OUTPUT_DIR / cld_name / f"run_{run_idx}"),
            result_excel_path=str(OUTPUT_DIR / cld_name / f"run_{run_idx}" / f"{cld_name}_results.xlsx"),
        )
        
        duration = time.time() - start_time
        
        # Extract key metrics
        result = {
            "cld_name": cld_name,
            "run_idx": run_idx,
            "source_paper": config["source_paper"],
            "session_id": experiment_info.get("session_id"),
            "duration_seconds": duration,
            "success": True,
            "experiment_info": experiment_info
        }
        
        logger.info(f"✓ {cld_name} completed in {duration:.1f}s")
        logger.info(f"  Session ID: {result['session_id']}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Error testing {cld_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "cld_name": cld_name,
            "run_idx": run_idx,
            "source_paper": config["source_paper"],
            "duration_seconds": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def run_all_cld_tests(num_runs: int = 1) -> list:
    """
    Run citation judging tests on all 3 CLDs.
    
    Args:
        num_runs: Number of runs per CLD
        
    Returns:
        List of result dictionaries
    """
    logger.info("=" * 80)
    logger.info("CLD CITATION JUDGING TEST - ULEMANS METHODOLOGY")
    logger.info("=" * 80)
    logger.info(f"Testing {len(GT_CLDS)} CLDs with {num_runs} runs each")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for cld_name, config in GT_CLDS.items():
        for run_idx in range(1, num_runs + 1):
            result = test_single_cld(cld_name, config, run_idx)
            all_results.append(result)
            
            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = OUTPUT_DIR / f"intermediate_results_{timestamp}.json"
            with open(intermediate_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def save_final_results(results: list):
    """Save final results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"cld_citation_test_results_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Final results saved to: {output_file}")
    return output_file


def print_summary(results: list):
    """Print summary of test results."""
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} {result['cld_name']} (Run {result['run_idx']})")
        print(f"   Source: {result['source_paper']['citation']}")
        print(f"   Duration: {result['duration_seconds']:.1f}s")
        if result["success"]:
            print(f"   Session: {result.get('session_id', 'N/A')}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("CLD CITATION JUDGING TEST")
    print("Following Ulemans Methodology for Ground Truth CLDs")
    print("=" * 80)
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run tests
    results = run_all_cld_tests(num_runs=1)
    
    # Save results
    output_file = save_final_results(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 80)
    print(f"COMPLETED: {datetime.now().isoformat()}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()






