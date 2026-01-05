#!/usr/bin/env python3
"""
RQ1: Judge Base Session (No Corruption)

Judges original generated motivations to correlate judge scores with FP/FN/TP/TN classifications.
This is different from judging corrupted sessions - here we evaluate the quality of the 
generator's original causal explanations.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment, CausalDiscovery, export_edges_comparison_to_excel
from neo4j_session_cloner import SessionCloner

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable not set!")

print("OpenAI API key is set:", os.getenv('OPENAI_API_KEY')[:10] + "..." + os.getenv('OPENAI_API_KEY')[-4:])


def judge_base_session(
    base_session_id: str,
    excel_file: str,
    judge_config: dict,
    judge_name: str,
    judge_approach: str = "correctness",
    ci_compute_embeddings: bool = False,
    citation_search_provider: str = "brave",
    output_dir: Path = None
):
    """
    Judge a base session without corruption.
    
    Args:
        base_session_id: The base session ID to judge
        excel_file: Original Excel file path
        judge_config: Judge configuration
        judge_name: Name for this judging strategy
        judge_approach: "correctness" or "per_citation_aggregate"
        ci_compute_embeddings: Enable embedding computation
        citation_search_provider: Search provider ('brave', 'perplexity', 'llm')
        output_dir: Output directory for results
    
    Returns:
        Result dictionary from run_discovery_experiment
    """
    
    print(f"\n{'='*80}")
    print(f"Judging Base Session: {judge_name}")
    print(f"{'='*80}")
    print(f"Base session: {base_session_id}")
    print(f"Excel file: {excel_file}")
    print(f"Judge approach: {judge_approach}")
    print(f"Citation search provider: {citation_search_provider}")
    print()
    
    # Dummy configs (not used for judging-only)
    dummy_generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 1.0
    }
    
    dummy_corruptor_config = None  # No corruption
    
    # Load the experiment YAML to get paths
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    # 🔄 CLONE the base session to avoid contamination
    print(f"📋 Cloning base session to preserve original...")
    cloner = SessionCloner()
    judged_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        new_session_id=None,  # Auto-generate new session ID
        include_judge_data=False  # Don't copy any existing judge data
    )
    print(f"✅ Cloned session: {judged_session_id}\n")
    
    # Run judging on the CLONED session with PARALLEL execution
    result = run_discovery_experiment(
        excel_path=str(excel_file),
        yaml_path=str(yaml_path),
        dev_mode=False,
        generator_config=dummy_generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=judge_config,
        judge_edges=True,  # CRITICAL: Enable judging!
        judge_models=[judge_config["model"]],  # CRITICAL: Pass model explicitly!
        num_judges=1,  # Single judge
        judge_approach=judge_approach,
        judge_enable_web_search=(judge_approach != "correctness"),
        judge_parallel=True,  # 🚀 ENABLE PARALLEL JUDGING!
        judge_max_workers=10,  # Process up to 10 edges concurrently
        retrieved_session_id=judged_session_id,  # Use CLONED session
        output_json_prefix=Path(excel_file).stem,
        experiment_description=f"Base_{judge_name}_parallel",
        citation_search_provider=citation_search_provider,
        ci_compute_embeddings=ci_compute_embeddings
    )
    
    # Move result file to output directory
    excel_path = result.get('result_excel_path') or result.get('excel_filename')
    if output_dir and excel_path:
        result_file = Path(excel_path)
        if result_file.exists():
            new_path = output_dir / f"judged_{base_session_id}_{judge_name}.xlsx"
            result_file.rename(new_path)
            result['excel_filename'] = str(new_path)
            print(f"✅ Moved result to: {new_path}")
        else:
            print(f"⚠️  Warning: Excel file not found at {result_file}")
    else:
        print(f"⚠️  Warning: No Excel path found in result (output_dir={output_dir}, excel_path={excel_path})")
    
    # Store both session IDs
    result['base_session_id'] = base_session_id
    result['judged_session_id'] = judged_session_id
    
    return result


def main():
    """Main execution."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge base session for FP/FN/TP/TN correlation")
    parser.add_argument("base_session_id", help="Base session ID to judge")
    parser.add_argument("excel_file", help="Path to original Excel file")
    parser.add_argument("--approach", type=str, default="correctness", 
                        help="Judging approach: 'correctness' or 'per_citation_aggregate' (default: correctness)")
    parser.add_argument("--search-provider", type=str, default="brave",
                        help="Citation search provider: 'brave', 'perplexity', or 'llm' (default: brave)")
    parser.add_argument("--embeddings", action="store_true", help="Enable CI embeddings computation (slower but more metrics)")
    
    args = parser.parse_args()
    
    # Use the specified approach
    judge_approach = args.approach
    search_provider = args.search_provider
    
    print("="*80)
    print("RQ1: Judge Base Session (FP/FN/TP/TN Correlation)")
    print("="*80)
    print(f"Base session: {args.base_session_id}")
    print(f"Excel file: {args.excel_file}")
    print(f"Judge approach: {judge_approach}")
    print(f"Search provider: {search_provider}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq1_base_judging_{judge_approach}_{search_provider}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Judge configuration: Single Claude Sonnet 4.5
    judge_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0,
        "max_tokens": 250
    }
    
    print("="*80)
    print("JUDGING BASE SESSION")
    print("="*80)
    print()
    
    result = judge_base_session(
        base_session_id=args.base_session_id,
        excel_file=args.excel_file,
        judge_config=judge_config,
        judge_name=f"base_{judge_approach}",
        judge_approach=judge_approach,
        ci_compute_embeddings=args.embeddings,
        citation_search_provider=search_provider,
        output_dir=output_dir
    )
    
    print()
    print("="*80)
    print("✅ BASE SESSION JUDGING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nExcel file: {result.get('excel_filename')}")
    print()
    print("Next steps:")
    print(f"  1. Run analysis: python parameter_tuning_experiments/analysis/rq1_analysis.py {output_dir}")
    print(f"  2. Check FP/FN/TP/TN correlation in the analysis report")
    print()


if __name__ == "__main__":
    main()
