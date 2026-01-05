#!/usr/bin/env python3
"""
RQ1 Prompt Sensitivity Test: Structured Criteria Evaluation (SCE)

This script tests the SCE prompt - a scientifically defensible alternative to CoT
based on:
- G-Eval framework (structured criteria)
- Logic-of-Thought (verification-based approach for causal reasoning)
- Best practices (concise, multi-dimensional, explicit criteria)

Usage:
    python run_rq1_sce_test.py <base_session_id>
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

# SCE Prompt configuration
SCE_PROMPT = {
    "yaml_file": "prompts_correctness_structured_criteria.yaml",
    "description": "Structured-Criteria-Evaluation",
    "prompt_style": "Multi-dimensional evaluation with explicit criteria (G-Eval + LoT)"
}

# Judge config - moderate tokens for structured output
JUDGE_CONFIG = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.3,
    "max_tokens": 500  # Sufficient for structured criteria output
}

def apply_sce_prompt(
    base_session_id: str,
    cld_name: str,
    original_excel: str,
    output_dir: Path,
    cloner: SessionCloner
) -> dict:
    """Apply SCE prompt variant to a cloned copy of the base session."""
    
    # Step 1: Clone the base session for this prompt variant
    logging.info(f"📋 Cloning session for {SCE_PROMPT['description']}...")
    judged_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        include_judge_data=False  # Exclude any existing judge data
    )
    logging.info(f"Cloned: {base_session_id[:8]}... → {judged_session_id[:8]}...")
    
    yaml_path = f"parameter_tuning_experiments/alternative_prompts/{SCE_PROMPT['yaml_file']}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"judged_{cld_name}_structured_criteria_{timestamp}.xlsx"
    
    logging.info(f"🔍 Judging with {SCE_PROMPT['description']}...")
    logging.info(f"Prompt style: {SCE_PROMPT['prompt_style']}")
    logging.info(f"Max tokens: {JUDGE_CONFIG['max_tokens']}")
    logging.info(f"Evaluation criteria: 4-dimensional (Logic, Direction, Specificity, Plausibility)")
    
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
        judge_config=JUDGE_CONFIG,
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
        citation_search_provider="brave",
        node_comparison_enable=False
    )
    
    return {
        'result_file': str(result_path),
        'base_session_id': base_session_id,
        'judged_session_id': judged_session_id,
        'prompt_variant': 'structured_criteria',
        'cld_name': cld_name,
        'judge_approach': 'correctness',
        'timestamp': timestamp
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ1 SCE Test')
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
    logging.info("RQ1 STRUCTURED CRITERIA EVALUATION (SCE) TEST")
    logging.info("=" * 80)
    logging.info(f"Base session ID: {args.base_session_id}")
    logging.info(f"CLD name: {args.cld_name}")
    logging.info(f"Judge model: {JUDGE_CONFIG['model']}")
    logging.info(f"Max tokens: {JUDGE_CONFIG['max_tokens']}")
    logging.info(f"Scientific basis: G-Eval + Logic-of-Thought + Best Practices")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("")
    
    # Initialize Neo4j session cloner
    logging.info("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run SCE judging
    try:
        result = apply_sce_prompt(
            base_session_id=args.base_session_id,
            cld_name=args.cld_name,
            original_excel=args.excel,
            output_dir=output_dir,
            cloner=cloner
        )
        
        logging.info("\n" + "=" * 80)
        logging.info("STRUCTURED CRITERIA EVALUATION TEST COMPLETE")
        logging.info("=" * 80)
        logging.info(f"✅ Result saved to: {result['result_file']}")
        logging.info(f"📊 Next step: Analyze results and compare with Baseline/Mechanistic/CoT")
        
        # Save metadata
        metadata_file = output_dir / f"sce_test_metadata_{result['timestamp']}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'timestamp': result['timestamp'],
                'judge_config': JUDGE_CONFIG,
                'base_session_id': args.base_session_id,
                'judged_session_id': result['judged_session_id'],
                'result': result,
                'scientific_basis': {
                    'G-Eval': 'Structured criteria-based evaluation framework',
                    'Logic-of-Thought': 'Verification-based approach for causal reasoning',
                    'Best_Practices': 'Explicit criteria, low-precision scores, structured output'
                }
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
