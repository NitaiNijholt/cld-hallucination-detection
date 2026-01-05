#!/usr/bin/env python3
"""
RQ1 Prompt Sensitivity Test: Re-run Chain-of-Thought ONLY with increased max_tokens

This script:
1. Loads base CLD session ID
2. Applies ONLY the Chain-of-Thought judge prompt
3. Uses increased max_tokens (3000) to allow full CoT reasoning
4. Saves result for comparing with baseline/mechanistic from previous run

Usage:
    python run_rq1_cot_only.py <base_session_id>
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env.dev")

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment
from neo4j_session_cloner import SessionCloner

# CoT Prompt configuration
COT_PROMPT = {
    "yaml_file": "prompts_correctness_cot.yaml",
    "description": "Chain-of-Thought",
    "prompt_style": "Step-by-step analysis before judgment"
}

# Judge config with INCREASED max_tokens for CoT reasoning
JUDGE_CONFIG = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.3,
    "max_tokens": 3000  # Increased from 250 to allow full CoT output
}

def apply_cot_prompt(
    base_session_id: str,
    cld_name: str,
    original_excel: str,
    output_dir: Path,
    cloner: SessionCloner
) -> dict:
    """Apply CoT prompt variant to a cloned copy of the base session."""
    
    # Step 1: Clone the base session for this prompt variant
    logging.info(f"📋 Cloning session for {COT_PROMPT['description']}...")
    judged_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        include_judge_data=False  # Exclude any existing judge data
    )
    logging.info(f"Cloned: {base_session_id[:8]}... → {judged_session_id[:8]}...")
    
    yaml_path = f"parameter_tuning_experiments/alternative_prompts/{COT_PROMPT['yaml_file']}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"judged_{cld_name}_chain_of_thought_{timestamp}.xlsx"
    
    logging.info(f"🔍 Judging with {COT_PROMPT['description']}...")
    logging.info(f"Prompt style: {COT_PROMPT['prompt_style']}")
    logging.info(f"Max tokens: {JUDGE_CONFIG['max_tokens']}")
    
    # Step 2: Apply judging to the cloned session
    exp_info = run_discovery_experiment(
        retrieved_session_id=judged_session_id,  # Use the fresh clone
        excel_path=original_excel,
        yaml_path=yaml_path,
        
        # Generator config (required but not used - session already exists)
        generator_config={
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7
        },
        
        # Corruptor config (not used - no corruption)
        corruption_rate=0.0,
        corruptor_config={
            "provider": "perplexity",
            "model": "mistral-7b-instruct",
            "temperature": 0.7
        },
        
        # Judging settings
        judge_edges=True,
        judge_approach="correctness",  # Correctness judging
        judge_enable_web_search=False,
        judge_config=JUDGE_CONFIG,  # INCREASED max_tokens
        num_judges=1,
        judge_models=["gpt-4.1"],
        judge_parallel=True,
        judge_max_workers=10,
        
        # Output settings
        result_excel_path=str(result_path),
        output_dir=str(output_dir),
        plot_session_graph=False,
        plot_validation_graph=False,
        export_json=True,
        
        # Citation search settings
        embedding_enable=False,
        citation_search_provider="brave",  # Always use Brave for citation filling
        node_comparison_enable=False
    )
    
    return {
        'result_file': str(result_path),
        'base_session_id': base_session_id,
        'judged_session_id': judged_session_id,
        'prompt_variant': 'chain_of_thought',
        'cld_name': cld_name,
        'judge_approach': 'correctness',
        'timestamp': timestamp
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ1 CoT-Only Re-run')
    parser.add_argument('base_session_id', help='Base session ID to judge')
    parser.add_argument('--cld-name', default='Depressive_symptoms_in_response_to_a_stressor',
                       help='CLD name for output file naming')
    parser.add_argument('--excel', 
                       default='parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx',
                       help='Path to original Excel file')
    parser.add_argument('--output-dir',
                       default='final_runs/RQ1a_judging_correctness_GT_depresive/run_1',
                       help='Output directory for results')
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("RQ1 CHAIN-OF-THOUGHT RE-RUN (Increased max_tokens)")
    logging.info("=" * 80)
    logging.info(f"Base session ID: {args.base_session_id}")
    logging.info(f"CLD name: {args.cld_name}")
    logging.info(f"Judge model: {JUDGE_CONFIG['model']}")
    logging.info(f"Max tokens: {JUDGE_CONFIG['max_tokens']} (increased for CoT)")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("")
    
    # Initialize Neo4j session cloner
    logging.info("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run CoT judging
    try:
        result = apply_cot_prompt(
            base_session_id=args.base_session_id,
            cld_name=args.cld_name,
            original_excel=args.excel,
            output_dir=output_dir,
            cloner=cloner
        )
        
        logging.info("\n" + "=" * 80)
        logging.info("CHAIN-OF-THOUGHT RE-RUN COMPLETE")
        logging.info("=" * 80)
        logging.info(f"✅ Result saved to: {result['result_file']}")
        logging.info(f"📊 Next step: Analyze results and compare with baseline/mechanistic")
        
        # Save metadata
        metadata_file = output_dir / f"cot_rerun_metadata_{result['timestamp']}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'timestamp': result['timestamp'],
                'judge_config': JUDGE_CONFIG,
                'base_session_id': args.base_session_id,
                'judged_session_id': result['judged_session_id'],
                'result': result
            }, f, indent=2)
        logging.info(f"📝 Metadata saved to: {metadata_file}")
        
    except Exception as e:
        logging.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cloner.close()

if __name__ == "__main__":
    main()
