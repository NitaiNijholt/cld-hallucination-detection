#!/usr/bin/env python3
"""
RQ1b Correction - Single Run Worker Script (Uses Existing RQ1a Judged Sessions)

This script tests multiple corrector prompt variants on a single judged CLD session from RQ1a.
It reads the judged session ID from the RQ1a directory.

Usage:
    python run_rq1b_correction_single.py <rq1a_run_dir> <output_dir>
    
Example:
    python run_rq1b_correction_single.py \
      final_runs/RQ1a_gt_synth_correctness/depressive/run_1 \
      final_runs/RQ1b_correction_experiment_final/depressive/run_1
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import CausalDiscovery

# Corrector prompt variants to test
CORRECTOR_VARIANTS = {
    "original": {"description": "Original (Weak Instructions)"},
    "improved": {"description": "Improved (Explicit Guidance)"},
    "cot": {"description": "Chain of Thought"},
    "mechanistic": {"description": "Bradford Hill Framework"}
}

def load_judged_session_id(rq1a_dir: Path) -> tuple:
    """Load the judged session ID from an RQ1a judged Excel file."""
    # Find a judged Excel file (any prompt variant will work, they all use the same session)
    judged_files = list(rq1a_dir.glob("judged_*.xlsx"))
    if not judged_files:
        raise FileNotFoundError(f"No judged Excel files found in {rq1a_dir}")
    
    # Read session ID from Params sheet
    excel_file = judged_files[0]
    params_df = pd.read_excel(excel_file, sheet_name='Params')
    session_id_row = params_df[params_df['Parameter'] == 'session_id']
    
    if session_id_row.empty:
        raise ValueError(f"No session_id found in {excel_file}")
    
    judged_session_id = session_id_row.iloc[0]['Value']
    
    # Also load base and corrupted session IDs from session_info.txt
    session_info_file = rq1a_dir / "session_info.txt"
    info = {}
    if session_info_file.exists():
        with open(session_info_file, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    info[k] = v
    
    base_session_id = info.get('base_session_id')
    corrupted_session_id = info.get('corrupted_session_id')
    
    return judged_session_id, base_session_id, corrupted_session_id

def run_correction_with_variant(
    judged_session_id: str,
    variant_name: str,
    variant_info: dict,
    cloner: SessionCloner,
    corrector_model: str = "gpt-4.1"
) -> dict:
    """Run correction with a specific prompt variant."""
    
    logging.info(f"📋 Cloning judged session for {variant_info['description']}...")
    corrected_session_id = cloner.clone_session(
        original_session_id=judged_session_id,
        include_judge_data=True  # Keep judge verdicts
    )
    logging.info(f"Cloned: {judged_session_id[:8]}... → {corrected_session_id[:8]}...")
    
    logging.info(f"🔧 Running correction with {variant_info['description']}...")
    
    # Initialize CausalDiscovery
    prompts_path = Path(__file__).parent / "alternative_prompts" / "prompts_citation_mechanistic_lit.yaml"
    
    discovery = CausalDiscovery(
        target_variable="Unknown",
        temporal_scale="Unknown",
        spatial_scale="Unknown",
        yaml_path=str(prompts_path),
        dev_mode=False,
        generator_config={"provider": "openai", "model": corrector_model},
        corruptor_config={"provider": "openai", "model": corrector_model},
        judge_config={"provider": "openai", "model": corrector_model},
        judge_enable_web_search=False,  # Disable web search for correctness judging
        generator_enable_web_search=False,
        corruptor_enable_web_search=False
    )
    
    # Set to corrected session
    discovery.session_id = corrected_session_id
    
    # Run correction with re-judging
    outcomes = discovery.correct_edges_serial(
        corrector_models=[corrector_model],
        max_rounds=1,
        action_order=("revise", "recite", "remove"),
        rejudge_after=True,  # Re-judge after correction to measure score improvement
        rejudge_approach="correctness",
        rejudge_parallel=False,
        judge_models=[corrector_model],
        num_judges=1,
        use_simple_corrector=False,
        corrector_prompt_variant=variant_name
    )
    
    logging.info(f"✅ {variant_name} complete: {len(outcomes)} edges processed")
    
    # Count actions
    action_counts = {}
    for o in outcomes:
        act = o.get("action", "none")
        action_counts[act] = action_counts.get(act, 0) + 1
    
    # Get judge scores before and after from Neo4j
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # Get scores from corrected session (after re-judging)
        query = '''
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $sid AND r.aggregate_score IS NOT NULL
        RETURN avg(r.aggregate_score) as avg_score,
               sum(CASE WHEN r.judge_verdict = 'CORRECT' THEN 1 ELSE 0 END) as correct,
               sum(CASE WHEN r.judge_verdict = 'PARTIALLY_CORRECT' THEN 1 ELSE 0 END) as partial,
               sum(CASE WHEN r.judge_verdict = 'INCORRECT' THEN 1 ELSE 0 END) as incorrect
        '''
        result = session.run(query, {'sid': corrected_session_id}).single()
        
        avg_score_after = result['avg_score'] if result and result['avg_score'] else 0.0
        correct_after = result['correct'] if result else 0
        partial_after = result['partial'] if result else 0
        incorrect_after = result['incorrect'] if result else 0
    
    driver.close()
    
    return {
        'variant_name': variant_name,
        'variant_description': variant_info['description'],
        'corrected_session_id': corrected_session_id,
        'edges_processed': len(outcomes),
        'action_counts': action_counts,
        'avg_score_after_rejudge': avg_score_after,
        'correct_after': correct_after,
        'partial_after': partial_after,
        'incorrect_after': incorrect_after
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_rq1b_correction_single.py <rq1a_run_dir> <output_dir>")
        print("\nExample:")
        print("  python run_rq1b_correction_single.py \\")
        print("    final_runs/RQ1a_gt_synth_correctness/depressive/run_1 \\")
        print("    final_runs/RQ1b_correction_experiment_final/depressive/run_1")
        sys.exit(1)
    
    rq1a_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load session IDs from RQ1a
    judged_session_id, base_session_id, corrupted_session_id = load_judged_session_id(rq1a_dir)
    
    logging.info("=" * 80)
    logging.info("RQ1b: CORRECTION EXPERIMENT - ALL CORRECTOR PROMPTS")
    logging.info("=" * 80)
    logging.info(f"RQ1a Source: {rq1a_dir}")
    logging.info(f"Output Dir: {output_dir}")
    logging.info(f"Judged Session ID: {judged_session_id}")
    logging.info(f"Base Session ID (GT): {base_session_id}")
    logging.info(f"Corrupted Session ID: {corrupted_session_id}")
    logging.info(f"Corrector variants: {len(CORRECTOR_VARIANTS)}")
    logging.info("")
    
    # Initialize Neo4j session cloner
    logging.info("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    logging.info("✅ Session cloner ready\n")
    
    # Run correction with all variants
    results = []
    try:
        for variant_name, variant_info in CORRECTOR_VARIANTS.items():
            logging.info(f"\n{'='*80}")
            logging.info(f"VARIANT {len(results)+1}/{len(CORRECTOR_VARIANTS)}: {variant_info['description'].upper()}")
            logging.info(f"{'='*80}\n")
            
            result = run_correction_with_variant(
                judged_session_id=judged_session_id,
                variant_name=variant_name,
                variant_info=variant_info,
                cloner=cloner
            )
            results.append(result)
        
        logging.info("\n" + "=" * 80)
        logging.info(f"ALL {len(CORRECTOR_VARIANTS)} VARIANTS COMPLETE")
        logging.info("=" * 80)
        
        for i, result in enumerate(results, 1):
            logging.info(f"{i}. {result['variant_description']}")
            logging.info(f"   Session: {result['corrected_session_id'][:8]}...")
            logging.info(f"   Actions: {result['action_counts']}")
        
        # Save comprehensive metadata
        metadata_file = output_dir / f"correction_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metadata = {
            'experiment_info': {
                'experiment_type': 'RQ1b_correction_multi_prompt',
                'timestamp': datetime.now().isoformat(),
                'script_name': Path(__file__).name,
            },
            'input_parameters': {
                'rq1a_source_dir': str(rq1a_dir),
                'judged_session_id': judged_session_id,
                'base_session_id': base_session_id,
                'corrupted_session_id': corrupted_session_id,
                'output_dir': str(output_dir),
            },
            'corrector_variants': CORRECTOR_VARIANTS,
            'execution_results': results
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"\n📝 Metadata saved to: {metadata_file}")
        logging.info(f"\n🎯 Next step: python analyze_rq1b_correction_variants.py {output_dir}")
        
    except Exception as e:
        logging.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cloner.close()

if __name__ == "__main__":
    main()
