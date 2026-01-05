#!/usr/bin/env python3
"""
Batch Runner for Malaysia CLDs - Phase 0 and Phase 2

Runs all 4 Malaysia CLDs:
1. malaysia_healthcare_financing.xlsx
2. malaysia_hospital_capacity.xlsx
3. malaysia_patient_pathways.xlsx
4. malaysia_physician_flows.xlsx

For each CLD:
- Phase 0: Generate base CLD
- Phase 2: Judge the CLD
- Save results

Usage:
    python run_malaysia_batch.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment
from neo4j_session_cloner import SessionCloner

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable not set!")

print("OpenAI API key is set:", os.getenv('OPENAI_API_KEY')[:10] + "..." + os.getenv('OPENAI_API_KEY')[-4:])

# Define all 4 Malaysia CLDs
MALAYSIA_CLDS = [
    {
        'name': 'malaysia_healthcare_financing',
        'excel': 'parameter_tuning_experiments/ground_truth_clds_for_experiments/malaysia_healthcare_financing.xlsx'
    },
    {
        'name': 'malaysia_hospital_capacity',
        'excel': 'parameter_tuning_experiments/ground_truth_clds_for_experiments/malaysia_hospital_capacity.xlsx'
    },
    {
        'name': 'malaysia_patient_pathways',
        'excel': 'parameter_tuning_experiments/ground_truth_clds_for_experiments/malaysia_patient_pathways.xlsx'
    },
    {
        'name': 'malaysia_physician_flows',
        'excel': 'parameter_tuning_experiments/ground_truth_clds_for_experiments/malaysia_physician_flows.xlsx'
    }
]


def generate_base_cld(cld_name: str, excel_file: str):
    """Phase 0: Generate base CLD."""
    print(f"\n{'='*80}")
    print(f"PHASE 0: {cld_name}")
    print(f"{'='*80}")
    
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    dummy_corruptor_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    dummy_judge_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0
    }
    
    yaml_path = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"
    
    result = run_discovery_experiment(
        excel_path=excel_file,
        yaml_path=yaml_path,
        dev_mode=False,
        generator_config=generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=dummy_judge_config,
        corruption_rate=0.0,
        judge_edges=False,
        parallel=True,
        max_workers=10,
        citation_search_provider="brave",
        output_json_prefix=cld_name,
        experiment_description=f"Base_{cld_name}",
        embedding_enable=False,
        ci_compute_embeddings=False,
        node_comparison_enable=False,
    )
    
    session_id = result.get('session_id')
    print(f"✅ Phase 0 complete: {session_id}")
    return session_id


def judge_cld(cld_name: str, excel_file: str, base_session_id: str, output_dir: Path):
    """Phase 2: Judge the CLD."""
    print(f"\n{'='*80}")
    print(f"PHASE 2: {cld_name}")
    print(f"{'='*80}")
    
    # Clone session
    cloner = SessionCloner()
    judged_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        new_session_id=None,
        include_judge_data=False
    )
    print(f"Cloned: {base_session_id} → {judged_session_id}")
    
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 1.0
    }
    
    dummy_corruptor_config = None
    
    judge_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0,
        "max_tokens": 250
    }
    
    yaml_path = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"
    
    result = run_discovery_experiment(
        excel_path=excel_file,
        yaml_path=yaml_path,
        dev_mode=False,
        generator_config=generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=judge_config,
        judge_edges=True,
        judge_models=[judge_config["model"]],
        num_judges=1,
        judge_approach="correctness",
        judge_enable_web_search=False,
        judge_parallel=True,
        judge_max_workers=10,
        retrieved_session_id=judged_session_id,
        output_json_prefix=Path(excel_file).stem,
        experiment_description=f"Judged_{cld_name}",
        citation_search_provider="brave",
        ci_compute_embeddings=False
    )
    
    excel_path = result.get('result_excel_path') or result.get('excel_filename')
    
    # Move to output directory
    if excel_path and Path(excel_path).exists():
        new_path = output_dir / f"judged_{cld_name}_base_correctness.xlsx"
        Path(excel_path).rename(new_path)
        excel_path = new_path
        print(f"✅ Phase 2 complete: {excel_path}")
    
    return {
        'base_session_id': base_session_id,
        'judged_session_id': judged_session_id,
        'excel_path': str(excel_path)
    }


def main():
    """Main execution."""
    print("="*80)
    print("MALAYSIA CLDs BATCH RUNNER")
    print("="*80)
    print(f"Processing {len(MALAYSIA_CLDS)} CLDs")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq1_malaysia_batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    results = []
    
    for i, cld in enumerate(MALAYSIA_CLDS, 1):
        print(f"\n{'#'*80}")
        print(f"CLD {i}/4: {cld['name']}")
        print(f"{'#'*80}")
        
        try:
            # Phase 0: Generate
            session_id = generate_base_cld(cld['name'], cld['excel'])
            
            # Phase 2: Judge
            judge_result = judge_cld(cld['name'], cld['excel'], session_id, output_dir)
            
            results.append({
                'cld_name': cld['name'],
                'excel_file': cld['excel'],
                **judge_result
            })
            
        except Exception as e:
            print(f"\n❌ ERROR processing {cld['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_clds': len(results),
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ BATCH COMPLETE - {len(results)}/{len(MALAYSIA_CLDS)} CLDs processed")
    print(f"{'='*80}")
    print(f"\nResults directory: {output_dir}")
    print(f"Summary: {summary_file}")
    print()
    print("NEXT: Run analysis")
    print(f"  python parameter_tuning_experiments/analysis/rq1_analysis.py {output_dir}")


if __name__ == "__main__":
    main()
