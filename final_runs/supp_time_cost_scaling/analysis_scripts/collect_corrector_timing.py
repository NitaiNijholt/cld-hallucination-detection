#!/usr/bin/env python3
"""
Collect corrector wall-clock timing for scaling analysis.

Runs correction (without rejudging) on 3 CLDs to measure timing.
Saves results to corrector_timing_data.json for use in figure generation.

Usage:
    python collect_corrector_timing.py
"""

import time
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data_science"))

# Load environment
load_dotenv(PROJECT_ROOT / ".env.dev")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# CLD configuration with pre-extracted session IDs from RQ1a experiments
CLD_CONFIG = {
    "depressive": {
        "n_edges": 34,
        "judged_session_id": "7cc70e7f-546b-45d3-9132-be8d8aae69fa",
    },
    "social_norms": {
        "n_edges": 12,
        "judged_session_id": "81297f79-62a0-433d-9cc7-6942e1ba1009",
    },
    "emergency_department": {
        "n_edges": 66,
        "judged_session_id": "39c84e01-0bd6-48eb-b5d5-c3c4305d83c0",
    }
}


def run_correction_timing(judged_session_id: str, cld_name: str, n_edges: int) -> dict:
    """Run correction and measure wall-clock time."""
    from neo4j_session_cloner import SessionCloner
    from modules import CausalDiscovery
    
    logger.info(f"Starting correction timing for {cld_name} ({n_edges} edges)")
    
    # Clone session
    logger.info(f"  Cloning session {judged_session_id[:8]}...")
    cloner = SessionCloner()
    corrected_session_id = cloner.clone_session(
        original_session_id=judged_session_id,
        include_judge_data=True
    )
    logger.info(f"  Cloned to {corrected_session_id[:8]}...")
    
    # Setup discovery with minimal config
    prompts_path = PROJECT_ROOT / "data_science" / "parameter_tuning_experiments" / "alternative_prompts" / "prompts_citation_mechanistic_lit.yaml"
    
    discovery = CausalDiscovery(
        target_variable="Unknown",
        temporal_scale="Unknown",
        spatial_scale="Unknown",
        yaml_path=str(prompts_path),
        dev_mode=False,
        generator_config={"provider": "openai", "model": "gpt-4.1"},
        corruptor_config={"provider": "openai", "model": "gpt-4.1"},
        judge_config={"provider": "openai", "model": "gpt-4.1"},
        judge_enable_web_search=False,
        generator_enable_web_search=False,
        corruptor_enable_web_search=False
    )
    
    # Set to corrected session
    discovery.session_id = corrected_session_id
    
    # Time the correction (no rejudging)
    logger.info(f"  Running correction (rejudge_after=False)...")
    start_time = time.time()
    
    outcomes = discovery.correct_edges_serial(
        corrector_models=["gpt-4.1"],
        max_rounds=1,
        action_order=("revise", "recite", "remove"),
        rejudge_after=False,  # Skip rejudging to isolate corrector time
        use_simple_corrector=False,
        corrector_prompt_variant="baseline"
    )
    
    wall_time_s = time.time() - start_time
    
    # Count actions
    action_counts = {}
    for o in outcomes:
        act = o.get("action", "none")
        action_counts[act] = action_counts.get(act, 0) + 1
    
    logger.info(f"  Completed in {wall_time_s:.2f}s - {len(outcomes)} edges processed")
    
    cloner.close()
    
    return {
        "cld": cld_name,
        "n_edges": n_edges,
        "corrector_time_s": wall_time_s,
        "edges_processed": len(outcomes),
        "action_counts": action_counts,
        "session_id": corrected_session_id,
        "timestamp": datetime.now().isoformat()
    }


def main():
    logger.info("=" * 60)
    logger.info("CORRECTOR TIMING COLLECTION")
    logger.info("=" * 60)
    logger.info(f"CLDs to process: {list(CLD_CONFIG.keys())}")
    logger.info("")
    
    results = []
    
    for cld_name, config in CLD_CONFIG.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {cld_name}")
        logger.info(f"{'='*40}")
        
        try:
            result = run_correction_timing(
                judged_session_id=config["judged_session_id"],
                cld_name=cld_name,
                n_edges=config["n_edges"]
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for {cld_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = SCRIPT_DIR / "corrector_timing_data.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "corrector_timing_collection",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for r in results:
        time_per_edge = r["corrector_time_s"] / r["n_edges"] if r["n_edges"] > 0 else 0
        logger.info(f"  {r['cld']}: {r['corrector_time_s']:.2f}s ({r['n_edges']} edges, {time_per_edge:.2f}s/edge)")
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
