#!/usr/bin/env python3
"""
Generate Base Sessions for RQ1a Ground Truth Experiments

Generates 3 different base CLD sessions per CLD using seeds 1, 2, 3.
This allows testing whether judge performance conclusions hold across
different CLD generations.

Outputs: rq1a_ground_truth_base_sessions.json

Usage:
    python generate_rq1a_ground_truth_base_sessions.py
    python generate_rq1a_ground_truth_base_sessions.py --clds depressive social_norms
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# CLD configurations
CLDS = {
    "test": {
        "name": "CLD_test",
        "excel": "test_CLD/CLD_test.xlsx",
        "target_variable": "BMI",
        "temporal_scale": "Weeks to Months",
        "spatial_scale": "Individual Level"
    },
    "depressive": {
        "name": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
        "excel": "ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
        "target_variable": "Depressive Symptoms",
        "temporal_scale": "Days to Weeks",
        "spatial_scale": "Individual Level"
    },
    "social_norms": {
        "name": "Social_norms_and_obesity_prevalence",
        "excel": "ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx",
        "target_variable": "Obesity Prevalence",
        "temporal_scale": "Years to Decades",
        "spatial_scale": "Community/Population Level"
    },
    "emergency_department": {
        "name": "older_persons_emergency_department_visits_and_interactions",
        "excel": "ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx",
        "target_variable": "Emergency Department Visits",
        "temporal_scale": "Hours to Days",
        "spatial_scale": "Individual to Facility Level"
    }
}

SEEDS = [1, 2, 3]

def generate_base_session(cld_key: str, cld_config: dict, seed: int) -> dict:
    """
    Generate a single base CLD session with a specific seed.
    
    Returns:
        dict with session_id, seed, timestamp, and generation metadata
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Generating: {cld_key} with seed={seed}")
    logging.info(f"{'='*80}")
    
    script_dir = Path(__file__).parent
    excel_path = script_dir / cld_config["excel"]
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Generator config with seed
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "seed": seed,  # Fixed seed for reproducibility
        "logprobs": True,
        "top_logprobs": 5
    }
    
    # Dummy configs (not used for base generation)
    dummy_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    # Use a temporary directory (we only care about the session ID in Neo4j)
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix=f"cld_gen_{cld_key}_seed{seed}_"))
    result_path = temp_dir / f"{cld_config['name']}_seed_{seed}.xlsx"
    
    logging.info(f"Generator seed: {seed}")
    logging.info(f"Temp output: {temp_dir} (will be cleaned up)")
    
    start_time = datetime.now()
    yaml_path = str(script_dir / "alternative_prompts" / "prompts_correctness_baseline.yaml")
    
    # Run discovery experiment (generates nodes and edges in Neo4j)
    logging.info(f"🚀 Generating CLD with parallel edge discovery (10 workers)...")
    result = run_discovery_experiment(
        excel_path=str(excel_path),
        yaml_path=yaml_path,
        generator_config=generator_config,
        corruptor_config=dummy_config,
        judge_config=dummy_config,
        corruption_rate=0.0,  # No corruption
        judge_edges=False,  # No judging
        judge_parallel=True,  # Parallel generation
        judge_max_workers=10,
        citation_search_provider="brave",
        output_json_prefix=Path(excel_path).stem
    )
    
    # Extract session ID
    session_id = result.get("session_id") or result.get("retrieved_session_id")
    
    if not session_id:
        raise ValueError(f"Failed to get session_id for {cld_key} seed={seed}")
    
    generation_time = (datetime.now() - start_time).total_seconds()
    num_nodes = result.get("num_nodes", 0)
    num_edges = result.get("num_edges", 0)
    
    logging.info(f"✓ Generated session: {session_id}")
    logging.info(f"  Nodes: {num_nodes}, Edges: {num_edges}, Time: {generation_time:.1f}s")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    logging.info(f"✓ Cleaned up temp directory")
    
    return {
        "session_id": session_id,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "cld_key": cld_key,
        "cld_name": cld_config["name"],
        "target_variable": cld_config["target_variable"],
        "temporal_scale": cld_config["temporal_scale"],
        "spatial_scale": cld_config["spatial_scale"],
        "generator_config": generator_config,
        "generation_info": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "generation_time_seconds": generation_time
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate base sessions for RQ1a ground truth experiments")
    parser.add_argument(
        "--clds",
        nargs="+",
        default=list(CLDS.keys()),
        choices=list(CLDS.keys()),
        help="CLDs to generate (default: all)"
    )
    args = parser.parse_args()
    
    logging.info(f"\n{'='*80}")
    logging.info("RQ1a GROUND TRUTH: BASE SESSION GENERATION")
    logging.info(f"{'='*80}")
    logging.info(f"CLDs: {', '.join(args.clds)}")
    logging.info(f"Seeds: {SEEDS}")
    logging.info(f"Total sessions: {len(args.clds)} CLDs × {len(SEEDS)} seeds = {len(args.clds) * len(SEEDS)}")
    logging.info(f"Note: Sessions stored in Neo4j, only IDs saved to JSON registry")
    logging.info("")
    
    # Generate all sessions
    all_sessions = {}
    
    for cld_key in args.clds:
        if cld_key not in CLDS:
            logging.error(f"Unknown CLD: {cld_key}")
            continue
        
        cld_config = CLDS[cld_key]
        all_sessions[cld_key] = {}
        
        for seed in SEEDS:
            try:
                session_info = generate_base_session(cld_key, cld_config, seed)
                all_sessions[cld_key][f"seed_{seed}"] = session_info
                logging.info(f"✓ SUCCESS: {cld_key} seed={seed}")
            except Exception as e:
                logging.error(f"✗ FAILED: {cld_key} seed={seed}")
                logging.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
                all_sessions[cld_key][f"seed_{seed}"] = {
                    "error": str(e),
                    "seed": seed,
                    "timestamp": datetime.now().isoformat()
                }
    
    # Save consolidated session registry
    registry_path = Path(__file__).parent / "rq1a_ground_truth_base_sessions.json"
    with open(registry_path, 'w') as f:
        json.dump(all_sessions, f, indent=2)
    
    logging.info(f"\n{'='*80}")
    logging.info("SESSION REGISTRY SAVED")
    logging.info(f"{'='*80}")
    logging.info(f"Location: {registry_path}")
    logging.info("")
    logging.info("Summary:")
    
    for cld_key, sessions in all_sessions.items():
        successful = sum(1 for s in sessions.values() if "session_id" in s)
        failed = len(sessions) - successful
        logging.info(f"  {cld_key}: {successful}/{len(sessions)} successful")
        if failed > 0:
            logging.info(f"    ({failed} failed)")
    
    logging.info("")
    logging.info("Next steps:")
    logging.info("  1. Verify session registry: cat rq1a_ground_truth_base_sessions.json")
    logging.info("  2. Update multirun script to use these sessions")
    logging.info("  3. Run judging experiments: run_rq1a_ground_truth_multirun.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
