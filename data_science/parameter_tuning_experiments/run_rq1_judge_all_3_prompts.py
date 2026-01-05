#!/usr/bin/env python3
"""
RQ1: Judge with All 3 Prompts (Baseline, Mechanistic, SCE)

Judges a session with all 3 validated correctness prompts in parallel.
Use this for Run 2 (corruption detection experiments).

Usage:
    python run_rq1_judge_all_3_prompts.py <session_id> <excel_file> <output_dir>
    
Example:
    python run_rq1_judge_all_3_prompts.py \\
      abc12345 \\
      parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx \\
      final_runs/RQ1a_corruption_detection/run_2
"""

import sys
import os
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
import glob
import yaml

def discover_prompts(prompt_type: str = "correctness") -> dict:
    """
    Auto-discover prompt YAML files from alternative_prompts/ directory.
    
    Args:
        prompt_type: Either "citation" or "correctness"
    
    Returns:
        Dictionary of prompts with structure:
        {
            "baseline": {"yaml_file": "...", "description": "...", "max_tokens": ...},
            ...
        }
    """
    prompts_dir = Path(__file__).parent / "alternative_prompts"
    pattern = f"prompts_{prompt_type}_*.yaml"
    yaml_files = sorted(prompts_dir.glob(pattern))
    
    if not yaml_files:
        raise FileNotFoundError(f"No prompts found matching pattern: {pattern} in {prompts_dir}")
    
    prompts = {}
    for yaml_file in yaml_files:
        # Extract variant name from filename: prompts_correctness_baseline.yaml -> baseline
        variant_name = yaml_file.stem.replace(f"prompts_{prompt_type}_", "")
        
        # Try to read description from YAML metadata if available
        try:
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
                description = yaml_content.get('metadata', {}).get('description', f"{prompt_type.title()} {variant_name.title()}")
        except:
            description = f"{prompt_type.title()} {variant_name.title()}"
        
        # Determine max_tokens based on variant type
        if "cot" in variant_name.lower():
            max_tokens = 10000  # CoT needs LOTS of tokens for step-by-step reasoning
        elif variant_name == "sce" or "structured" in variant_name:
            max_tokens = 2000   # Structured criteria need generous tokens
        else:
            max_tokens = 1500   # Baseline/simple prompts (generous for detailed reasoning)
        
        prompts[variant_name] = {
            "yaml_file": yaml_file.name,
            "description": description,
            "max_tokens": max_tokens
        }
    
    logging.info(f"📋 Auto-discovered {len(prompts)} {prompt_type} prompts:")
    for name, info in prompts.items():
        logging.info(f"   - {name}: {info['yaml_file']} (max_tokens={info['max_tokens']})")
    
    return prompts

# Auto-discover correctness prompts
PROMPTS = discover_prompts(prompt_type="correctness")

# Judge config
JUDGE_CONFIG = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.0,
    "seed": 42
}

def judge_with_prompt(
    session_id: str,
    prompt_name: str,
    prompt_info: dict,
    excel_file: str,
    output_dir: Path,
    cloner: SessionCloner
) -> dict:
    """Judge a session with a specific prompt variant."""
    
    # Clone the session for this prompt
    logging.info(f"📋 Cloning session for {prompt_info['description']}...")
    judged_session_id = cloner.clone_session(
        original_session_id=session_id,
        include_judge_data=False
    )
    logging.info(f"Cloned: {session_id[:8]}... → {judged_session_id[:8]}...")
    
    # Use absolute path for YAML file to avoid path issues with uv run
    yaml_path = Path(__file__).parent / "alternative_prompts" / prompt_info['yaml_file']
    yaml_path = str(yaml_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cld_name = Path(excel_file).stem.replace('_in_healthy_adults', '')
    result_path = output_dir / f"judged_{cld_name}_{prompt_name}_{timestamp}.xlsx"
    
    logging.info(f"🔍 Judging with {prompt_info['description']}...")
    logging.info(f"Max tokens: {prompt_info['max_tokens']}")
    
    # Update judge config with max_tokens
    judge_config = JUDGE_CONFIG.copy()
    judge_config['max_tokens'] = prompt_info['max_tokens']
    
    # Judge the session
    exp_info = run_discovery_experiment(
        retrieved_session_id=judged_session_id,
        excel_path=excel_file,
        yaml_path=yaml_path,
        
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
        judge_approach="correctness",
        judge_enable_web_search=False,
        judge_config=judge_config,
        judge_temperature=judge_config.get('temperature', 0.3),
        judge_seed=judge_config.get('seed', None),
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
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ1 Judge with All 3 Prompts')
    parser.add_argument('session_id', help='Session ID to judge')
    parser.add_argument('excel_file', help='Path to Excel file')
    parser.add_argument('output_dir', help='Output directory (e.g., final_runs/RQ1a_corruption_detection/run_2)')
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("RQ1: JUDGE WITH ALL 3 PROMPTS")
    logging.info("=" * 80)
    logging.info(f"Session ID: {args.session_id}")
    logging.info(f"Excel file: {args.excel_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Judge model: {JUDGE_CONFIG['model']}")
    logging.info("")
    
    # Initialize Neo4j session cloner
    logging.info("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Judge with all 3 prompts
    results = []
    try:
        for prompt_name, prompt_info in PROMPTS.items():
            logging.info(f"\n{'='*80}")
            logging.info(f"PROMPT {len(results)+1}/3: {prompt_info['description'].upper()}")
            logging.info(f"{'='*80}\n")
            
            result = judge_with_prompt(
                session_id=args.session_id,
                prompt_name=prompt_name,
                prompt_info=prompt_info,
                excel_file=args.excel_file,
                output_dir=output_dir,
                cloner=cloner
            )
            results.append(result)
        
        logging.info("\n" + "=" * 80)
        logging.info("ALL 3 PROMPTS COMPLETE")
        logging.info("=" * 80)
        
        for i, result in enumerate(results, 1):
            logging.info(f"{i}. {result['prompt_description']}")
            logging.info(f"   File: {Path(result['result_file']).name}")
        
        # Save comprehensive metadata for reproducibility
        import json
        import platform
        metadata_file = output_dir / f"judging_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metadata = {
            'experiment_info': {
                'experiment_type': 'corruption_detection_correctness',
                'timestamp': datetime.now().isoformat(),
                'script_name': Path(__file__).name,
                'script_version': '2.0_auto_discovery',
            },
            'input_parameters': {
                'base_session_id': args.session_id,
                'excel_file': args.excel_file,
                'output_dir': str(output_dir),
            },
            'judge_configuration': {
                'provider': JUDGE_CONFIG['provider'],
                'model': JUDGE_CONFIG['model'],
                'temperature': JUDGE_CONFIG['temperature'],
                'seed': JUDGE_CONFIG['seed'],
                'judge_approach': 'correctness',
                'num_judges': 1,
                'judge_parallel': True,
                'judge_max_workers': 10,
                'judge_enable_web_search': False,
            },
            'prompt_variants': {
                name: {
                    'yaml_file': info['yaml_file'],
                    'description': info['description'],
                    'max_tokens': info['max_tokens']
                } for name, info in PROMPTS.items()
            },
            'execution_results': results,
            'system_info': {
                'python_version': platform.python_version(),
                'platform': platform.platform(),
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"\n📝 Metadata saved to: {metadata_file}")
        logging.info("\n🎯 Next step: Analyze results with analyze_corruption_detection.py")
        
    except Exception as e:
        logging.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cloner.close()

if __name__ == "__main__":
    main()
