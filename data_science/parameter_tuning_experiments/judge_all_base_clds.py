#!/usr/bin/env python3
"""
Judge All Base CLDs in Batch Mode (Phase 2)

Reads base_cld_sessions.json from a generation directory and judges all sessions.

Usage:
    python judge_all_base_clds.py <path_to_base_cld_sessions.json>
    
Example:
    python judge_all_base_clds.py parameter_tuning_experiments/results/rq1_base_generation_malaysia_20251011_045123/base_cld_sessions.json
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment
from neo4j_session_cloner import SessionCloner

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable not set!")

print("OpenAI API key is set:", os.getenv('OPENAI_API_KEY')[:10] + "..." + os.getenv('OPENAI_API_KEY')[-4:])


def judge_single_cld(
    cld_name: str,
    base_session_id: str,
    excel_file: str,
    output_dir: Path,
    judge_approach: str = "correctness"
):
    """
    Judge a single base CLD.
    
    Args:
        cld_name: Name of the CLD
        base_session_id: Base session ID to judge
        excel_file: Original Excel file path
        output_dir: Output directory for results
        judge_approach: "correctness" or "per_citation_aggregate"
    
    Returns:
        dict with judged_session_id and excel_path
    """
    print(f"\n{'='*80}")
    print(f"JUDGING CLD: {cld_name}")
    print(f"{'='*80}")
    print(f"Base session: {base_session_id}")
    print(f"Excel file: {excel_file}")
    print(f"Judge approach: {judge_approach}")
    print()
    
    # Clone session to avoid contamination
    print(f"📋 Cloning base session to preserve original...")
    cloner = SessionCloner()
    judged_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        new_session_id=None,
        include_judge_data=False
    )
    print(f"✅ Cloned: {base_session_id} → {judged_session_id}\n")
    
    # Dummy configs
    dummy_generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 1.0
    }
    
    dummy_corruptor_config = None
    
    judge_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0,
        "seed": 42,
        "max_tokens": 250
    }
    
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    # Run judging with PARALLEL execution
    print(f"🚀 Starting PARALLEL edge judging (10 workers)...\n")
    result = run_discovery_experiment(
        excel_path=str(excel_file),
        yaml_path=str(yaml_path),
        dev_mode=False,
        generator_config=dummy_generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=judge_config,
        judge_temperature=judge_config.get('temperature', 1.0),
        judge_seed=judge_config.get('seed', None),
        judge_edges=True,
        judge_models=[judge_config["model"]],
        num_judges=1,
        judge_approach=judge_approach,
        judge_enable_web_search=(judge_approach != "correctness"),
        judge_parallel=True,  # 🚀 PARALLEL judging
        judge_max_workers=10,
        retrieved_session_id=judged_session_id,
        output_json_prefix=Path(excel_file).stem,
        experiment_description=f"Judged_{cld_name}",
        citation_search_provider="brave" if judge_approach != "correctness" else None,
        ci_compute_embeddings=False,  # Skip expensive embeddings
    )
    
    excel_path = result.get('result_excel_path') or result.get('excel_filename')
    
    # Move to output directory with consistent naming
    if excel_path and Path(excel_path).exists():
        new_path = output_dir / f"judged_{cld_name}_base_{judge_approach}.xlsx"
        Path(excel_path).rename(new_path)
        excel_path = str(new_path)
    
    print(f"\n✅ JUDGING COMPLETE!")
    print(f"   Judged session: {judged_session_id}")
    print(f"   Excel: {excel_path}")
    
    return {
        'cld_name': cld_name,
        'base_session_id': base_session_id,
        'judged_session_id': judged_session_id,
        'excel_path': excel_path,
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Judge all base CLDs from a sessions JSON file."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Judge all base CLDs in batch mode')
    parser.add_argument('sessions_file', help='Path to base_cld_sessions.json from Phase 0')
    parser.add_argument('--approach', choices=['correctness', 'per_citation_aggregate', 'one_causal_source_found'], 
                       default='correctness',
                       help='Judging approach (default: correctness)')
    args = parser.parse_args()
    
    sessions_file = Path(args.sessions_file)
    judge_approach = args.approach
    
    if not sessions_file.exists():
        print(f"Error: Sessions file not found: {sessions_file}")
        sys.exit(1)
    
    # Read sessions
    with open(sessions_file, 'r') as f:
        sessions_data = json.load(f)
    
    sessions = sessions_data.get('sessions', [])
    
    print("="*80)
    print("RQ1: Judge All Base CLDs (Phase 2)")
    print("="*80)
    print(f"Sessions file: {sessions_file}")
    print(f"Judging approach: {judge_approach}")
    print(f"Total CLDs: {len(sessions)}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq1_judged_base_clds_{judge_approach}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Judge each CLD
    judged_results = []
    for i, session in enumerate(sessions, 1):
        print(f"\n{'#'*80}")
        print(f"CLD {i}/{len(sessions)}: {session['cld_name']}")
        print(f"{'#'*80}")
        
        try:
            result = judge_single_cld(
                cld_name=session['cld_name'],
                base_session_id=session['session_id'],
                excel_file=session['original_excel'],
                output_dir=output_dir,
                judge_approach=judge_approach
            )
            judged_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR judging {session['cld_name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_file = output_dir / "judged_cld_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'source_sessions_file': str(sessions_file),
            'total_judged': len(judged_results),
            'results': judged_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ ALL BASE CLDs JUDGED!")
    print(f"{'='*80}")
    print(f"\nJudged {len(judged_results)}/{len(sessions)} CLDs")
    print(f"\n📋 Results saved to: {results_file}")
    print(f"\nJudged CLDs:")
    for r in judged_results:
        print(f"  - {r['cld_name']}: {r['judged_session_id']}")
        print(f"    Excel: {r['excel_path']}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP: Run RQ1 analysis on judged CLDs")
    print(f"{'='*80}")
    print(f"\npython parameter_tuning_experiments/analysis/rq1_analysis.py {output_dir}")
    print()


if __name__ == "__main__":
    main()
