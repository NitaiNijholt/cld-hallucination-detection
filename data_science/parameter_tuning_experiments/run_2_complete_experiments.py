#!/usr/bin/env python3
"""
Run 2 Complete Experiments on 3 Original CLDs

Generates 2 independent runs with BOTH generation and judging:
- Run 1: Generate + Judge all 3 CLDs
- Run 2: Generate + Judge all 3 CLDs

This will produce ~400-500 edges total with both Gen and Judge CI metrics.

Usage:
    python run_2_complete_experiments.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable not set!")

print("OpenAI API key is set:", os.getenv('OPENAI_API_KEY')[:10] + "..." + os.getenv('OPENAI_API_KEY')[-4:])


def run_single_cld_with_judging(cld_name: str, excel_file: str, output_dir: Path, run_number: int):
    """
    Run a complete experiment: Generate + Judge a single CLD.
    
    Args:
        cld_name: Name of the CLD
        excel_file: Path to Excel file
        output_dir: Output directory for results
        run_number: Run number (1 or 2)
    
    Returns:
        dict with session_id, excel_path, and statistics
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_number} - CLD: {cld_name}")
    print(f"{'='*80}")
    print(f"Excel file: {excel_file}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Generator config: GPT-4.1 with logprobs
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "logprobs": True,
        "top_logprobs": 5
    }
    
    # Judge config: GPT-4.1 for correctness evaluation with logprobs
    judge_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.3,
        "max_tokens": 250,
        "logprobs": True,
        "top_logprobs": 5
    }
    
    # Dummy corruptor (not used)
    dummy_corruptor_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    # Load the experiment YAML
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    print(f"🚀 Run {run_number}: Generating edges with PARALLEL discovery (10 workers)...")
    print(f"🔍 Then judging with citation verification using Brave search...")
    print()
    
    # Run COMPLETE experiment: Generation + Judging
    result = run_discovery_experiment(
        excel_path=str(excel_file),
        yaml_path=str(yaml_path),
        dev_mode=False,
        generator_config=generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=judge_config,  # ✅ Judge settings (logprobs, temperature, etc.)
        judge_models=["gpt-4.1"],  # ✅ Judge model names
        corruption_rate=0.0,  # No corruption
        judge_edges=True,  # ✅ ENABLE JUDGING
        judge_approach="per_citation_aggregate",  # Citation-based judging
        judge_enable_web_search=True,  # Enable web search
        parallel=True,  # Parallel edge discovery
        max_workers=10,
        citation_search_provider="brave",
        output_json_prefix=f"{Path(excel_file).stem}_run{run_number}",
        embedding_enable=True  # Enable cosine similarity
    )
    
    session_id = result['session_id']
    excel_path = result['excel_path']
    
    print(f"\n✅ Run {run_number} complete for {cld_name}")
    print(f"   Session: {session_id}")
    print(f"   Excel: {excel_path}")
    
    return {
        'run_number': run_number,
        'cld_name': cld_name,
        'session_id': session_id,
        'excel_path': excel_path,
        'original_excel': excel_file
    }


def main():
    print("\n" + "="*80)
    print("RUNNING 2 COMPLETE EXPERIMENTS (Generate + Judge)")
    print("="*80)
    print("CLDs: 3 original (Social Norms, Older Persons, Depressive Symptoms)")
    print("Runs: 2 independent experiments")
    print("Expected output: ~400-500 edges with Gen + Judge CI metrics")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq2_complete_runs_2x3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Define 3 original CLDs with absolute paths
    cld_folder = PROJECT_ROOT / "data_science" / "parameter_tuning_experiments" / "ground_truth_clds_for_experiments"
    
    clds = [
        {
            'name': 'Social_norms_and_obesity_prevalence',
            'excel': str(cld_folder / 'Social_norms_and_obesity_prevalence.xlsx')
        },
        {
            'name': 'older_persons_emergency_department_visits',
            'excel': str(cld_folder / 'older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx')
        },
        {
            'name': 'Depressive_symptoms_in_response_to_a_stressor',
            'excel': str(cld_folder / 'Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx')
        }
    ]
    
    # Run 2 complete experiments
    all_results = []
    
    for run_num in [1, 2]:
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT RUN {run_num}/2")
        print(f"{'#'*80}\n")
        
        run_results = []
        
        for i, cld in enumerate(clds, 1):
            print(f"\n{'*'*80}")
            print(f"Run {run_num} - CLD {i}/{len(clds)}: {cld['name']}")
            print(f"{'*'*80}")
            
            try:
                result = run_single_cld_with_judging(
                    cld_name=cld['name'],
                    excel_file=cld['excel'],
                    output_dir=output_dir,
                    run_number=run_num
                )
                run_results.append(result)
            except Exception as e:
                print(f"\n❌ ERROR in Run {run_num}, CLD {cld['name']}: {e}")
                import traceback
                traceback.print_exc()
        
        all_results.extend(run_results)
        
        print(f"\n✅ Run {run_num} complete: {len(run_results)}/{len(clds)} CLDs processed")
    
    # Save all session info
    results_file = output_dir / "experiment_runs_summary.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_runs': 2,
            'clds_per_run': len(clds),
            'total_experiments': len(all_results),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {results_file.name}")
    
    print("\n📊 Results summary:")
    for result in all_results:
        print(f"  Run {result['run_number']} - {result['cld_name']}")
        print(f"    Session: {result['session_id']}")
        print(f"    Excel: {Path(result['excel_path']).name}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP: Run feature selection analysis on combined dataset")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
