#!/usr/bin/env python3
"""
RQ1 Prompt Sensitivity Test: Apply Multiple Judge Prompts to Base CLDs

This script:
1. Loads base CLD session IDs (uncorrupted)
2. Applies 4 different judge prompts to each session:
   - Baseline (direct)
   - Chain-of-Thought (explicit reasoning)
   - Skeptical (conservative standards)
   - Mechanistic (mechanism-focused)
3. Uses the SAME judge model (GPT-4.1) across all prompts
4. Saves results for comparing prompt effectiveness

Usage:
    python run_rq1_prompt_sensitivity.py <base_cld_sessions.json>
    
Example:
    python run_rq1_prompt_sensitivity.py results/rq1_base_generation_20251011_043119/base_cld_sessions.json
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

# 3 Prompt variants to test (same judge model across all)
# Available for BOTH citation and correctness judging approaches
CITATION_PROMPTS = {
    "baseline": {
        "yaml_file": "prompts_judge_baseline.yaml",
        "description": "Baseline (direct)",
        "prompt_style": "Direct judgment without scaffolding"
    },
    "chain_of_thought": {
        "yaml_file": "prompts_judge_cot.yaml",
        "description": "Chain-of-Thought",
        "prompt_style": "Step-by-step analysis before judgment"
    },
    "mechanistic": {
        "yaml_file": "prompts_judge_mechanistic.yaml",
        "description": "Mechanistic",
        "prompt_style": "Focus on evaluating causal mechanisms"
    }
}

CORRECTNESS_PROMPTS = {
    "baseline": {
        "yaml_file": "prompts_correctness_baseline.yaml",
        "description": "Baseline (direct)",
        "prompt_style": "Direct judgment without scaffolding"
    },
    "chain_of_thought": {
        "yaml_file": "prompts_correctness_cot.yaml",
        "description": "Chain-of-Thought",
        "prompt_style": "Step-by-step analysis before judgment"
    },
    "mechanistic": {
        "yaml_file": "prompts_correctness_mechanistic.yaml",
        "description": "Mechanistic",
        "prompt_style": "Focus on evaluating causal mechanisms"
    }
}

# Fixed judge config (same across all prompts)
JUDGE_CONFIG = {
    "provider": "openai",
    "model": "gpt-4.1",
    "temperature": 0.3,
    "max_tokens": 250
}

def load_base_sessions(sessions_file: str) -> list:
    """Load base CLD session IDs."""
    sessions_path = Path(sessions_file)
    
    if not sessions_path.exists():
        raise FileNotFoundError(f"Sessions file not found: {sessions_path}")
    
    with open(sessions_path, 'r') as f:
        data = json.load(f)
    
    sessions = data.get('sessions', [])
    print(f"✅ Loaded {len(sessions)} base CLD sessions")
    return sessions

def apply_prompt_variant(
    base_session_id: str,
    cld_name: str,
    original_excel: str,
    prompt_name: str,
    prompt_config: dict,
    output_dir: Path,
    cloner: SessionCloner,
    judge_approach: str,
    enable_web_search: bool
) -> dict:
    """Apply a prompt variant to a cloned copy of the base session."""
    
    # Step 1: Clone the base session for this prompt variant
    print(f"  📋 Cloning session for {prompt_config['description']}...")
    judged_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        include_judge_data=False  # Exclude any existing judge data
    )
    print(f"     Cloned: {base_session_id[:8]}... → {judged_session_id[:8]}...")
    
    yaml_path = f"parameter_tuning_experiments/alternative_prompts/{prompt_config['yaml_file']}"
    
    result_path = output_dir / f"judged_{cld_name}_{prompt_name}.xlsx"
    
    print(f"  🔍 Judging with {prompt_config['description']}...")
    print(f"     Prompt style: {prompt_config['prompt_style']}")
    
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
        judge_approach=judge_approach,  # citation or correctness
        judge_enable_web_search=enable_web_search,  # Only for citation mode
        judge_config=JUDGE_CONFIG,  # Same judge model for all prompts
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
        embedding_enable=False,  # Skip expensive embeddings
        citation_search_provider="brave",  # Always use Brave for citation filling
        node_comparison_enable=False
    )
    
    # Store which approach was used in the return dict
    return {
        'result_file': str(result_path),
        'base_session_id': base_session_id,
        'judged_session_id': judged_session_id,
        'prompt_variant': prompt_name,
        'cld_name': cld_name,
        'judge_approach': judge_approach
    }
    
    print(f"  ✅ Judging complete: {result_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ1 Prompt Sensitivity Test')
    parser.add_argument('sessions_file', help='Path to base_cld_sessions.json')
    parser.add_argument('--approach', choices=['citation', 'correctness'], default='citation',
                       help='Judging approach: citation (default) or correctness')
    args = parser.parse_args()
    
    sessions_file = args.sessions_file
    approach = args.approach
    
    # Select prompt set based on approach
    if approach == 'citation':
        PROMPT_VARIANTS = CITATION_PROMPTS
        judge_approach = "per_citation_aggregate"
        mode_label = "CITATION-BASED"
        enable_web_search = True
    else:  # correctness
        PROMPT_VARIANTS = CORRECTNESS_PROMPTS
        judge_approach = "correctness"
        mode_label = "CORRECTNESS-BASED"
        enable_web_search = False
    
    logging.info("=" * 80)
    logging.info(f"RQ1 PROMPT SENSITIVITY TEST - {mode_label} JUDGING")
    logging.info("=" * 80)
    logging.info(f"Sessions file: {sessions_file}")
    logging.info(f"Judge model: {JUDGE_CONFIG['model']} (FIXED across all prompts)")
    logging.info(f"Judge approach: {judge_approach}")
    logging.info(f"Prompt variants: {len(PROMPT_VARIANTS)}")
    logging.info("")
    
    # Load base sessions
    sessions = load_base_sessions(sessions_file)
    
    # Initialize Neo4j session cloner
    logging.info("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Create output directory with approach suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"parameter_tuning_experiments/results/rq1_prompt_sensitivity_{approach}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all judging results
    judging_results = []
    
    logging.info("\n" + "=" * 80)
    logging.info("APPLYING PROMPT VARIANTS")
    logging.info("=" * 80)
    logging.info(f"Prompt variants to test: {len(PROMPT_VARIANTS)}")
    logging.info(f"Base CLDs to judge: {len(sessions)}")
    logging.info(f"Total experiments: {len(PROMPT_VARIANTS) * len(sessions)}")
    logging.info("")
    
    # Print prompt variants
    logging.info("Prompt variants:")
    for name, config in PROMPT_VARIANTS.items():
        logging.info(f"  {name}: {config['description']}")
        logging.info(f"    → {config['prompt_style']}")
    logging.info("")
    
    total_experiments = len(sessions) * len(PROMPT_VARIANTS)
    experiment_num = 0
    
    for session_idx, session_info in enumerate(sessions, 1):
        base_session_id = session_info['session_id']
        cld_name = session_info['cld_name']
        original_excel = session_info['original_excel']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"CLD {session_idx}/{len(sessions)}: {cld_name}")
        logging.info(f"Base session: {base_session_id}")
        logging.info(f"{'='*80}")
        
        for prompt_name, prompt_config in PROMPT_VARIANTS.items():
            experiment_num += 1
            logging.info(f"\n[{experiment_num}/{total_experiments}] Prompt: {prompt_name}")
            
            try:
                result = apply_prompt_variant(
                    base_session_id=base_session_id,
                    cld_name=cld_name,
                    original_excel=original_excel,
                    prompt_name=prompt_name,
                    prompt_config=prompt_config,
                    output_dir=output_dir,
                    cloner=cloner,
                    judge_approach=judge_approach,
                    enable_web_search=enable_web_search
                )
                
                # Record result
                judging_results.append({
                    'base_session_id': base_session_id,
                    'judged_session_id': result['judged_session_id'],
                    'cld_name': cld_name,
                    'original_excel': original_excel,
                    'prompt_variant': prompt_name,
                    'prompt_description': prompt_config['description'],
                    'prompt_style': prompt_config['prompt_style'],
                    'yaml_file': prompt_config['yaml_file'],
                    'judge_model': JUDGE_CONFIG['model'],
                    'judge_approach': result['judge_approach'],
                    'result_file': result['result_file'],
                    'timestamp': timestamp
                })
            except Exception as e:
                logging.error(f"\n❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Save mapping for analysis
    results_df = pd.DataFrame(judging_results)
    mapping_file = output_dir / f"prompt_sensitivity_results_{timestamp}.csv"
    results_df.to_csv(mapping_file, index=False)
    
    # Also save as JSON for easier reading
    json_file = output_dir / f"prompt_sensitivity_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'judge_config': JUDGE_CONFIG,
            'judge_approach': approach,
            'total_experiments': len(judging_results),
            'prompt_variants': list(PROMPT_VARIANTS.keys()),
            'clds': [s['cld_name'] for s in sessions],
            'results': judging_results
        }, f, indent=2)
    
    logging.info("\n" + "=" * 80)
    logging.info("PROMPT SENSITIVITY TEST COMPLETE")
    logging.info("=" * 80)
    logging.info(f"✅ Completed {len(judging_results)} judging experiments")
    logging.info(f"   - {len(sessions)} CLDs")
    logging.info(f"   - {len(PROMPT_VARIANTS)} prompt variants")
    logging.info(f"✅ Results saved to:")
    logging.info(f"   - CSV: {mapping_file}")
    logging.info(f"   - JSON: {json_file}")
    
    # Print prompt breakdown
    logging.info(f"\nResults by prompt variant:")
    for prompt_name in PROMPT_VARIANTS.keys():
        count = len([r for r in judging_results if r['prompt_variant'] == prompt_name])
        logging.info(f"  - {prompt_name}: {count} experiments")
    
    logging.info(f"\nResults by CLD:")
    for session in sessions:
        count = len([r for r in judging_results if r['cld_name'] == session['cld_name']])
        logging.info(f"  - {session['cld_name']}: {count} experiments")
    
    # Close Neo4j connection
    cloner.close()
    
    logging.info(f"\n📊 Next step: Analyze prompt sensitivity")
    logging.info(f"Command:")
    logging.info(f"  cd parameter_tuning_experiments/analysis")
    logging.info(f"  python analyze_prompt_sensitivity.py \\")
    logging.info(f"    {output_dir.name}")

if __name__ == "__main__":
    main()
