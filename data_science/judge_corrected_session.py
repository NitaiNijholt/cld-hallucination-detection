#!/usr/bin/env python3
"""
Judge a Corrected Session

Takes a corrected session ID and judges it to see if the revised motivations
lead to better judge scores.

Usage:
    python judge_corrected_session.py <corrected_session_id> <excel_file> <output_file>
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import CausalDiscovery

def main():
    if len(sys.argv) < 4:
        print("Usage: python judge_corrected_session.py <corrected_session_id> <excel_file> <output_file>")
        return 1
    
    session_id = sys.argv[1]
    excel_file = sys.argv[2]
    output_file = sys.argv[3]
    
    print("="*80)
    print("JUDGING CORRECTED SESSION")
    print("="*80)
    print(f"Session ID: {session_id}")
    print(f"Excel file: {excel_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Initialize
    prompts_path = Path("data_science/parameter_tuning_experiments/alternative_prompts/prompts_citation_mechanistic_lit.yaml")
    
    discovery = CausalDiscovery(
        target_variable="Unknown",
        temporal_scale="Unknown",
        spatial_scale="Unknown",
        yaml_path=str(prompts_path),
        dev_mode=False,
        generator_config={"provider": "openai", "model": "gpt-4.1"},
        corruptor_config={"provider": "openai", "model": "gpt-4.1"},
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.0, "seed": 42},
        judge_enable_web_search=False  # Disable web search for correctness judging
    )
    
    discovery.session_id = session_id
    
    # Judge all edges
    print("Judging all edges...")
    judged_edges = discovery.judge_all_edges_serial(
        judge_models=["gpt-4.1"],
        num_judges=1
    )
    
    print(f"✅ Judged {len(judged_edges)} edges")
    
    # Export to Excel (simple version without full pipeline)
    from modules import run_discovery_experiment
    
    result = run_discovery_experiment(
        retrieved_session_id=session_id,
        excel_path=excel_file,
        yaml_path=str(prompts_path),
        generator_config={"provider": "openai", "model": "gpt-4.1"},
        corruptor_config={"provider": "openai", "model": "gpt-4.1"},
        corruption_rate=0.0,
        judge_edges=False,  # Already judged above
        run_correction=False,
        result_excel_path=output_file,
        output_dir=str(Path(output_file).parent),
        embedding_enable=False,
        node_comparison_enable=False
    )
    
    print(f"✅ Results saved to {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

