#!/usr/bin/env python3
"""
Batch Judge Base CLDs

Applies correctness-based judging to base CLD sessions from phase 0.
Uses the session IDs from base_cld_sessions.json.

Usage:
    python run_batch_judge_base_clds.py <base_cld_sessions.json>
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment
from neo4j_session_cloner import SessionCloner

def judge_base_session(session_info: dict, output_dir: Path, cloner: SessionCloner):
    """Apply correctness-based judging to a base CLD session."""
    
    session_id = session_info['session_id']
    cld_name = session_info['cld_name']
    original_excel = session_info['original_excel']
    
    print(f"\n{'='*80}")
    print(f"JUDGING: {cld_name}")
    print(f"{'='*80}")
    print(f"Base session: {session_id}")
    
    # Clone the session for judging
    print(f"  📋 Cloning session...")
    judged_session_id = cloner.clone_session(
        original_session_id=session_id,
        include_judge_data=False
    )
    print(f"     Cloned: {session_id[:8]}... → {judged_session_id[:8]}...")
    
    # Generator config (not used - session already exists)
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    # Dummy corruptor config (not used)
    dummy_corruptor_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    # Judge config: GPT-4.1 for correctness evaluation
    judge_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.3,
        "seed": 42,
        "max_tokens": 250
    }
    
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    result_path = output_dir / f"judged_{cld_name}_citations.xlsx"
    
    print(f"  🔍 Judging with GPT-4.1 (citation-based with Brave search)...")
    
    # Run citation-based judging
    result = run_discovery_experiment(
        retrieved_session_id=judged_session_id,
        excel_path=original_excel,
        yaml_path=str(yaml_path),
        
        # Generator config (required but not used)
        generator_config=generator_config,
        
        # Corruptor config (required but not used)
        corruption_rate=0.0,
        corruptor_config=dummy_corruptor_config,
        
        # Judging settings - CITATION-BASED MODE
        judge_edges=True,
        judge_approach="per_citation_aggregate",  # Judge each citation separately
        judge_enable_web_search=True,  # Enable web search for citations
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
        
        # Citation search settings
        embedding_enable=True,  # Required for CI metrics (RQ2 analysis)
        citation_search_provider="brave",  # Use Brave for citation search
        node_comparison_enable=False
    )
    
    print(f"  ✅ Judging complete: {result_path}")
    
    return {
        'cld_name': cld_name,
        'base_session_id': session_id,
        'judged_session_id': judged_session_id,
        'result_file': str(result_path),
        'original_excel': original_excel
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_batch_judge_base_clds.py <base_cld_sessions.json>")
        print("\nExample:")
        print("  python run_batch_judge_base_clds.py results/rq1_base_generation_original_20251012_024712/base_cld_sessions.json")
        sys.exit(1)
    
    sessions_file = Path(sys.argv[1])
    
    if not sessions_file.exists():
        print(f"❌ Sessions file not found: {sessions_file}")
        sys.exit(1)
    
    print("="*80)
    print("BATCH JUDGE BASE CLDs - CORRECTNESS-BASED")
    print("="*80)
    print(f"Sessions file: {sessions_file}")
    print()
    
    # Load sessions
    with open(sessions_file, 'r') as f:
        sessions_data = json.load(f)
    
    sessions = sessions_data['sessions']
    print(f"✅ Loaded {len(sessions)} base CLD sessions")
    
    # Initialize Neo4j session cloner
    print("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    print("✅ Session cloner ready\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq1_base_judging_correctness_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Judge each session
    results = []
    for i, session_info in enumerate(sessions, 1):
        print(f"\n{'#'*80}")
        print(f"Session {i}/{len(sessions)}")
        print(f"{'#'*80}")
        
        try:
            result = judge_base_session(session_info, output_dir, cloner)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR judging {session_info['cld_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary
    summary_file = output_dir / f"judging_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_clds': len(results),
            'results': results
        }, f, indent=2)
    
    # Close Neo4j connection
    cloner.close()
    
    print(f"\n{'='*80}")
    print(f"✅ BATCH JUDGING COMPLETE")
    print(f"{'='*80}")
    print(f"Judged {len(results)}/{len(sessions)} CLDs")
    print(f"\nResults directory: {output_dir}")
    print(f"Summary: {summary_file}")
    print()
    print("📊 Next: Analyze judging results in the Excel files")
    print(f"   Check: {output_dir}/*.xlsx")


if __name__ == "__main__":
    main()
