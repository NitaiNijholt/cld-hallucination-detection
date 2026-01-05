#!/usr/bin/env python3
"""
RQ1a_ground_truth: Judge Clean CLDs with All 3 Prompts (TP/FP Analysis)

Judges a CLEAN (non-corrupted) session with all 3 validated correctness prompts.
Enables CI metrics for RQ2 analysis.

Key differences from corruption judging:
- NO corruption step (judges base session directly)
- CI metrics ENABLED (for RQ2 bonus analysis)
- Output includes TP/FP ground truth classifications

Usage:
    python run_rq1a_ground_truth_judge.py <session_id> <excel_file> <output_dir>
    
Example:
    python run_rq1a_ground_truth_judge.py \\
      49d62c01-e85c-4392-bb6d-eb89f53136a8 \\
      parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx \\
      final_runs/RQ1a_gt_lit_correctness/depressive/run_1
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
from logit_metrics import clear_embedding_cache
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

# Default judge config (can be overridden via CLI flags or env vars)
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_LOCAL_MODEL = "qwen2.5:32b-instruct"

def get_judge_config(local_llm: bool = False, model_override: str = None) -> dict:
    """
    Get judge configuration based on local LLM mode or environment.
    
    Args:
        local_llm: If True, use local LLM settings
        model_override: Override model name (takes precedence)
    
    Returns:
        Judge config dictionary with provider, model, temperature, seed
    """
    # Detect local LLM mode from environment if not explicitly set
    openai_api_url = os.getenv("OPENAI_API_URL", "")
    is_local = local_llm or openai_api_url.startswith("http://localhost")
    
    if is_local:
        model = model_override or DEFAULT_LOCAL_MODEL
        logging.info(f"🏠 Using LOCAL LLM mode: {model}")
        logging.info(f"   API URL: {openai_api_url or 'http://localhost:11434/v1/chat/completions'}")
    else:
        model = model_override or DEFAULT_OPENAI_MODEL
        logging.info(f"☁️ Using OpenAI API: {model}")
    
    return {
        "provider": "openai",
        "model": model,
        "temperature": 0.0,
        "seed": 42
    }

# Placeholder for runtime config (set in main())
JUDGE_CONFIG = None

def judge_with_prompt(
    session_id: str,
    prompt_name: str,
    prompt_info: dict,
    excel_file: str,
    output_dir: Path,
    cloner: SessionCloner,
    judge_approach: str = "correctness",
    judge_enable_web_search: bool = False,
    embedding_device: str = "cuda",
    is_local_llm: bool = False
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
    logging.info(f"⚡ CI metrics: ENABLED (for RQ2 analysis)")
    
    # Update judge config with max_tokens
    judge_config = JUDGE_CONFIG.copy()
    judge_config['max_tokens'] = prompt_info['max_tokens']
    
    # Judge the session with CI METRICS ENABLED
    exp_info = run_discovery_experiment(
        retrieved_session_id=judged_session_id,
        excel_path=excel_file,
        yaml_path=yaml_path,
        
        # Generator config (required but not used for judging-only)
        generator_config={
            "provider": "openai",
            "model": judge_config['model'],  # Use same model as judge
            "temperature": 0.7
        },
        
        # No corruption (clean CLD)
        corruption_rate=0.0,
        corruptor_config={
            "provider": "perplexity",
            "model": "mistral-7b-instruct",
            "temperature": 0.7
        },
        
        # Judging settings
        judge_edges=True,
        judge_approach=judge_approach,
        judge_enable_web_search=judge_enable_web_search,
        judge_config=judge_config,
        judge_temperature=judge_config.get('temperature', 0.3),
        judge_seed=judge_config.get('seed', None),
        num_judges=1,
        judge_models=[judge_config['model']],  # Use model from config (local or remote)
        # Use sequential processing for local LLM (Ollama can't handle many parallel requests)
        judge_parallel=not is_local_llm,
        judge_max_workers=1 if is_local_llm else 10,
        
        # ⭐ CI METRICS ENABLED FOR RQ2 ⭐
        embedding_enable=True,           # Enable embeddings
        ci_compute_embeddings=True,      # Compute expensive embeddings
        ci_parallel=True,                # Parallel computation for speed
        ci_max_workers=10,              # Max workers for CI metrics
        embedding_provider="local",      # Use local embeddings
        embedding_device=embedding_device,  # Device for embeddings
        
        # Citation settings
        citation_search_provider="brave",
        
        # Output settings
        result_excel_path=str(result_path),
        output_dir=str(output_dir),
        plot_session_graph=False,
        plot_validation_graph=False,
        export_json=True,
        
        # Node comparison
        node_comparison_enable=False
    )
    
    logging.info(f"✅ {prompt_name} complete: {result_path}")
    
    # Clear embedding cache to prevent CUDA memory fragmentation in multi-run scenarios
    # This ensures each session gets fresh CUDA state, mimicking the sequential script behavior
    clear_embedding_cache()
    logging.info("🧹 Cleared embedding cache (prevents CUDA errors in multi-run)")
    
    return {
        'prompt_name': prompt_name,
        'prompt_description': prompt_info['description'],
        'result_file': str(result_path),
        'judged_session_id': judged_session_id,
        'timestamp': timestamp
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ1a_ground_truth: Judge Clean CLD with All 3 Prompts')
    parser.add_argument('session_id', help='Session ID to judge (base session, no corruption)')
    parser.add_argument('excel_file', help='Path to Excel file (ground truth)')
    parser.add_argument('output_dir', help='Output directory (e.g., final_runs/RQ1a_gt_lit_correctness/depressive/run_1)')
    parser.add_argument('--seed', type=int, default=None, help='Base session generation seed (for metadata)')
    parser.add_argument('--judge-type', type=str, default='correctness', choices=['correctness', 'citation'],
                        help='Type of judging: correctness or citation (default: correctness)')
    parser.add_argument('--embedding-device', type=str, default='cuda',
                        help='Device for embeddings: cpu, cuda (default), cuda:0, etc.')
    parser.add_argument('--local-llm', action='store_true',
                        help='Use local LLM via Ollama instead of OpenAI API')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model name (e.g., qwen2.5:32b-instruct for local, gpt-4.1 for OpenAI)')
    args = parser.parse_args()
    
    # Initialize judge config based on local LLM mode
    global JUDGE_CONFIG
    JUDGE_CONFIG = get_judge_config(local_llm=args.local_llm, model_override=args.model)
    
    # Auto-discover prompts based on judge type
    PROMPTS = discover_prompts(prompt_type=args.judge_type)
    
    # Map user-friendly judge_type to actual judge_approach values
    judge_approach_map = {
        'correctness': 'correctness',
        'citation': 'per_citation_aggregate'
    }
    judge_approach = judge_approach_map[args.judge_type]
    
    logging.info("=" * 80)
    logging.info("RQ1a_ground_truth: JUDGE CLEAN CLD WITH ALL 3 PROMPTS")
    logging.info("=" * 80)
    logging.info(f"Session ID: {args.session_id}")
    logging.info(f"Excel file: {args.excel_file}")
    logging.info(f"Output directory: {args.output_dir}")
    if args.seed is not None:
        logging.info(f"Base session seed: {args.seed}")
    logging.info(f"Judge type: {args.judge_type}")
    logging.info(f"Judge model: {JUDGE_CONFIG['model']}")
    logging.info(f"Web search: {'ENABLED' if args.judge_type == 'citation' else 'DISABLED'}")
    logging.info(f"Embedding device: {args.embedding_device}")
    logging.info(f"CI metrics: ENABLED (for RQ2 analysis)")
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
            logging.info(f"PROMPT {len(results)+1}/{len(PROMPTS)}: {prompt_info['description'].upper()}")
            logging.info(f"{'='*80}\n")
            
            result = judge_with_prompt(
                session_id=args.session_id,
                prompt_name=prompt_name,
                prompt_info=prompt_info,
                excel_file=args.excel_file,
                output_dir=output_dir,
                cloner=cloner,
                judge_approach=judge_approach,
                judge_enable_web_search=(args.judge_type == 'citation'),
                embedding_device=args.embedding_device,
                is_local_llm=args.local_llm or os.getenv("OPENAI_API_URL", "").startswith("http://localhost")
            )
            results.append(result)
        
        logging.info("\n" + "=" * 80)
        logging.info(f"ALL {len(PROMPTS)} PROMPTS COMPLETE")
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
                'experiment_type': 'RQ1a_gt_lit_correctness',
                'research_question': 'Can judges distinguish TP from FP edges in clean CLDs?',
                'timestamp': datetime.now().isoformat(),
                'script_name': Path(__file__).name,
                'script_version': '1.0_ci_metrics_enabled',
            },
            'input_parameters': {
                'base_session_id': args.session_id,
                'base_session_seed': args.seed,
                'excel_file': args.excel_file,
                'output_dir': str(output_dir),
                'corruption_applied': False,  # Key difference from corruption experiments
            },
            'judge_configuration': {
                'provider': JUDGE_CONFIG['provider'],
                'model': JUDGE_CONFIG['model'],
                'temperature': JUDGE_CONFIG['temperature'],
                'seed': JUDGE_CONFIG['seed'],
                'judge_approach': args.judge_type,
                'num_judges': 1,
                'judge_parallel': not (args.local_llm or os.getenv("OPENAI_API_URL", "").startswith("http://localhost")),
                'judge_max_workers': 1 if (args.local_llm or os.getenv("OPENAI_API_URL", "").startswith("http://localhost")) else 10,
                'judge_enable_web_search': (args.judge_type == 'citation'),
                'local_llm_mode': args.local_llm or os.getenv("OPENAI_API_URL", "").startswith("http://localhost"),
                'openai_api_url': os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
            },
            'ci_metrics_configuration': {
                'enabled': True,
                'embedding_enable': True,
                'ci_compute_embeddings': True,
                'ci_parallel': True,
                'ci_max_workers': 10,
                'purpose': 'Enable RQ2 analysis (CI metrics vs TP/FP)',
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
        logging.info(f"\n🎯 Next step: Analyze results with analyze_rq1a_ground_truth_tp_fp.py")
        logging.info(f"📊 BONUS: CI metrics included in output for RQ2 analysis!")
        
    except Exception as e:
        logging.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cloner.close()

if __name__ == "__main__":
    main()
