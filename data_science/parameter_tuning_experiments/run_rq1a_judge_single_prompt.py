#!/usr/bin/env python3
"""
RQ1a: Judge with a Single Specified Prompt

Runs a single prompt variant on an existing session (e.g., corrupted session with citations).
Use this to add missing prompt variants to existing experiment runs without re-running others.

Usage:
    python run_rq1a_judge_single_prompt.py \
        --session-id <session_id> \
        --prompt <prompt_variant> \
        --excel <excel_file> \
        --output-dir <output_dir> \
        [--model <model_name>] \
        [--judge-type <citation|correctness>]

Example:
    python run_rq1a_judge_single_prompt.py \
        --session-id abc12345 \
        --prompt mechanistic_lit \
        --excel ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx \
        --output-dir final_runs/RQ1a_gt_synth_citation/depressive/run_1 \
        --model gpt-5-mini \
        --judge-type citation
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging
import json
import platform
import yaml

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


def get_prompt_info(prompt_name: str, judge_type: str = "citation") -> dict:
    """
    Get prompt info for a specific prompt variant.
    
    Args:
        prompt_name: Name of the prompt variant (e.g., "mechanistic_lit")
        judge_type: Either "citation" or "correctness"
    
    Returns:
        Dictionary with yaml_file, description, max_tokens
    """
    prompts_dir = Path(__file__).parent / "alternative_prompts"
    yaml_file = prompts_dir / f"prompts_{judge_type}_{prompt_name}.yaml"
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {yaml_file}")
    
    # Try to read description from YAML metadata
    try:
        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)
            description = yaml_content.get('metadata', {}).get('description', 
                f"{judge_type.title()} {prompt_name.title()}")
    except:
        description = f"{judge_type.title()} {prompt_name.title()}"
    
    # Determine max_tokens based on variant type
    if "cot" in prompt_name.lower():
        max_tokens = 10000
    elif prompt_name == "sce" or "structured" in prompt_name:
        max_tokens = 2000
    else:
        max_tokens = 1500
    
    return {
        "yaml_file": yaml_file.name,
        "yaml_path": str(yaml_file),
        "description": description,
        "max_tokens": max_tokens
    }


def judge_with_prompt(
    session_id: str,
    prompt_name: str,
    prompt_info: dict,
    excel_file: str,
    output_dir: Path,
    cloner: SessionCloner,
    judge_config: dict,
    judge_type: str = "citation"
) -> dict:
    """Judge a session with a specific prompt variant."""
    
    # Clone the session for this prompt
    logging.info(f"📋 Cloning session for {prompt_info['description']}...")
    judged_session_id = cloner.clone_session(
        original_session_id=session_id,
        include_judge_data=False
    )
    logging.info(f"Cloned: {session_id[:8]}... → {judged_session_id[:8]}...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cld_name = Path(excel_file).stem.replace('_in_healthy_adults', '')
    
    if judge_type == "citation":
        result_path = output_dir / f"judged_citation_{cld_name}_{prompt_name}_{timestamp}.xlsx"
        judge_approach = "per_citation_aggregate"
    else:
        result_path = output_dir / f"judged_{cld_name}_{prompt_name}_{timestamp}.xlsx"
        judge_approach = "correctness"
    
    logging.info(f"🔍 Judging with {prompt_info['description']}...")
    logging.info(f"Judge approach: {judge_approach}")
    logging.info(f"Max tokens: {prompt_info['max_tokens']}")
    
    # Update judge config with max_tokens
    judge_config_copy = judge_config.copy()
    judge_config_copy['max_tokens'] = prompt_info['max_tokens']
    
    # Judge the session
    exp_info = run_discovery_experiment(
        retrieved_session_id=judged_session_id,
        excel_path=excel_file,
        yaml_path=prompt_info['yaml_path'],
        
        # Generator config (required but not used)
        generator_config={
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7
        },
        
        # No corruption (already corrupted)
        corruption_rate=0.0,
        corruptor_config={
            "provider": "perplexity",
            "model": "mistral-7b-instruct",
            "temperature": 0.7
        },
        
        # Judging settings
        judge_edges=True,
        judge_approach=judge_approach,
        judge_enable_web_search=False,
        judge_config=judge_config_copy,
        judge_temperature=judge_config_copy.get('temperature', 0.0),
        judge_seed=judge_config_copy.get('seed', 42),
        num_judges=1,
        judge_models=[judge_config_copy['model']],
        judge_parallel=True,
        judge_max_workers=10,
        
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
    
    logging.info(f"✅ {prompt_name} complete: {result_path}")
    
    return {
        'prompt_name': prompt_name,
        'prompt_description': prompt_info['description'],
        'result_file': str(result_path),
        'judged_session_id': judged_session_id,
        'timestamp': timestamp
    }


def main():
    parser = argparse.ArgumentParser(
        description='RQ1a: Judge with a Single Specified Prompt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run mechanistic_lit on a corrupted session
  python run_rq1a_judge_single_prompt.py \\
    --session-id abc12345 \\
    --prompt mechanistic_lit \\
    --excel ground_truth_clds_for_experiments/Depressive.xlsx \\
    --output-dir final_runs/RQ1a_gt_synth_citation/depressive/run_1 \\
    --model gpt-5-mini \\
    --judge-type citation
        """
    )
    parser.add_argument('--session-id', required=True, help='Session ID to judge')
    parser.add_argument('--prompt', required=True, help='Prompt variant name (e.g., mechanistic_lit)')
    parser.add_argument('--excel', required=True, help='Path to ground truth Excel file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--model', default='gpt-5-mini', help='Judge model (default: gpt-5-mini)')
    parser.add_argument('--judge-type', default='citation', choices=['citation', 'correctness'],
                        help='Judge type (default: citation)')
    args = parser.parse_args()
    
    # Build judge config
    judge_config = {
        "provider": "openai",
        "model": args.model,
        "temperature": 0.0,
        "seed": 42,
        "use_structured_outputs": False
    }
    
    logging.info("=" * 80)
    logging.info("RQ1a: JUDGE WITH SINGLE PROMPT")
    logging.info("=" * 80)
    logging.info(f"Session ID: {args.session_id}")
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Excel file: {args.excel}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Judge model: {args.model}")
    logging.info(f"Judge type: {args.judge_type}")
    logging.info("")
    
    # Get prompt info
    try:
        prompt_info = get_prompt_info(args.prompt, args.judge_type)
        logging.info(f"📋 Prompt file: {prompt_info['yaml_file']}")
        logging.info(f"   Max tokens: {prompt_info['max_tokens']}")
    except FileNotFoundError as e:
        logging.error(f"❌ {e}")
        return 1
    
    # Initialize Neo4j session cloner
    logging.info("\nInitializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Judge with the specified prompt
    try:
        result = judge_with_prompt(
            session_id=args.session_id,
            prompt_name=args.prompt,
            prompt_info=prompt_info,
            excel_file=args.excel,
            output_dir=output_dir,
            cloner=cloner,
            judge_config=judge_config,
            judge_type=args.judge_type
        )
        
        logging.info("\n" + "=" * 80)
        logging.info("JUDGING COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Prompt: {result['prompt_description']}")
        logging.info(f"Output: {Path(result['result_file']).name}")
        
        # Save metadata
        metadata_file = output_dir / f"single_prompt_metadata_{args.prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metadata = {
            'experiment_info': {
                'experiment_type': f'single_prompt_{args.judge_type}',
                'timestamp': datetime.now().isoformat(),
                'script_name': Path(__file__).name,
            },
            'input_parameters': {
                'base_session_id': args.session_id,
                'prompt_variant': args.prompt,
                'excel_file': args.excel,
                'output_dir': str(output_dir),
            },
            'judge_configuration': judge_config,
            'prompt_info': {
                'yaml_file': prompt_info['yaml_file'],
                'description': prompt_info['description'],
                'max_tokens': prompt_info['max_tokens']
            },
            'execution_result': result,
            'system_info': {
                'python_version': platform.python_version(),
                'platform': platform.platform(),
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"\n📝 Metadata saved to: {metadata_file}")
        
    except Exception as e:
        logging.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cloner.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())






