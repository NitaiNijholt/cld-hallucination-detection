#!/usr/bin/env python3
"""
Generate Base CLDs in Batch Mode

Supports 3 batches:
1. original: 3 CLDs (Social_norms, older_persons, Depressive_symptoms)
2. malaysia: 4 CLDs (healthcare_financing, hospital_capacity, patient_pathways, physician_flows)
3. all: All 7 CLDs

Uses parallel edge discovery (max_workers=10) for speed.
Saves session IDs to: base_cld_sessions.json

Usage:
    python generate_all_base_clds.py --batch original
    python generate_all_base_clds.py --batch malaysia
    python generate_all_base_clds.py --batch all
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Load environment variables from .env.dev
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


def generate_base_cld(cld_name: str, excel_file: str, output_dir: Path, seed: int = None):
    """
    Generate a base CLD with parallel edge discovery.
    
    Args:
        cld_name: Name of the CLD
        excel_file: Path to Excel file
        output_dir: Output directory for results
        seed: Random seed for reproducibility (optional)
    
    Returns:
        dict with session_id and excel_path
    """
    print(f"\n{'='*80}")
    print(f"GENERATING BASE CLD: {cld_name}")
    print(f"{'='*80}")
    print(f"Excel file: {excel_file}")
    print(f"Output dir: {output_dir}")
    if seed is not None:
        print(f"Seed: {seed}")
    print()
    
    # Generator config: GPT-4.1
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "seed": seed,  # Random seed for reproducibility
        "logprobs": True,  # ✅ Request log probabilities for CI metrics
        "top_logprobs": 5   # ✅ Request top 5 log probabilities
    }
    
    # No corruptor/judge needed for base generation
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
    
    # Load the experiment YAML
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    # Run base CLD generation with PARALLEL edge discovery
    print(f"🚀 Generating CLD with PARALLEL edge discovery (10 workers)...\n")
    result = run_discovery_experiment(
        excel_path=str(excel_file),
        yaml_path=str(yaml_path),
        dev_mode=False,
        generator_config=generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=dummy_judge_config,
        corruption_rate=0.0,  # No corruption
        judge_edges=False,  # No judging yet
        judge_parallel=True,  # 🚀 PARALLEL edge discovery
        judge_max_workers=10,  # 10 concurrent workers
        citation_search_provider="brave",  # Use Brave for citations
        output_json_prefix=Path(excel_file).stem,
        output_dir=str(output_dir),
        experiment_description=f"Base_{cld_name}",
        embedding_enable=True,  # ✅ ENABLE CI metrics
        ci_compute_embeddings=True,  # ✅ Compute expensive embeddings (for RQ2 analysis)
        ci_parallel=True,  # ✅ Parallel CI computation for speed
        ci_max_workers=10,  # ✅ Max workers for CI metrics
        node_comparison_enable=False,  # Skip comparison for base generation
    )
    
    session_id = result.get('session_id')
    excel_path = result.get('result_excel_path')
    
    print(f"\n✅ BASE CLD GENERATED!")
    print(f"   Session ID: {session_id}")
    print(f"   Excel: {excel_path}")
    
    return {
        'cld_name': cld_name,
        'session_id': session_id,
        'excel_path': excel_path,
        'original_excel': str(excel_file),
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Generate all base CLDs."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate base CLDs')
    parser.add_argument('--batch', choices=['original', 'malaysia', 'all'], default='original',
                       help='Which batch to process: original (3 CLDs), malaysia (4 CLDs), or all (7 CLDs)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (optional)')
    args = parser.parse_args()
    
    print("="*80)
    print("RQ1: Generate All Base CLDs (Parallel Edge Discovery)")
    print("="*80)
    print(f"Batch: {args.batch}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq1_base_generation_{args.batch}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Define CLD batches with absolute paths
    cld_folder = PROJECT_ROOT / "data_science" / "parameter_tuning_experiments" / "ground_truth_clds_for_experiments"
    
    original_clds = [
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
    
    malaysia_clds = [
        {
            'name': 'malaysia_healthcare_financing',
            'excel': str(cld_folder / 'malaysia_healthcare_financing.xlsx')
        },
        {
            'name': 'malaysia_hospital_capacity',
            'excel': str(cld_folder / 'malaysia_hospital_capacity.xlsx')
        },
        {
            'name': 'malaysia_patient_pathways',
            'excel': str(cld_folder / 'malaysia_patient_pathways.xlsx')
        },
        {
            'name': 'malaysia_physician_flows',
            'excel': str(cld_folder / 'malaysia_physician_flows.xlsx')
        }
    ]
    
    # Select which CLDs to process
    if args.batch == 'original':
        clds = original_clds
    elif args.batch == 'malaysia':
        clds = malaysia_clds
    else:  # all
        clds = original_clds + malaysia_clds
    
    # Generate each base CLD
    sessions = []
    for i, cld in enumerate(clds, 1):
        print(f"\n{'#'*80}")
        print(f"CLD {i}/{len(clds)}: {cld['name']}")
        print(f"{'#'*80}")
        
        try:
            session_info = generate_base_cld(
                cld_name=cld['name'],
                excel_file=cld['excel'],
                output_dir=output_dir,
                seed=args.seed
            )
            sessions.append(session_info)
        except Exception as e:
            print(f"\n❌ ERROR generating {cld['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save session IDs to a JSON file
    session_file = output_dir / "base_cld_sessions.json"
    with open(session_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_clds': len(sessions),
            'sessions': sessions
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ ALL BASE CLDs GENERATED!")
    print(f"{'='*80}")
    print(f"\nGenerated {len(sessions)} base CLDs")
    print(f"\n📋 Session IDs saved to: {session_file}")
    print(f"\nSession details:")
    for s in sessions:
        print(f"  - {s['cld_name']}: {s['session_id']}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP: Run parallel judging on all base sessions")
    print(f"{'='*80}")
    print(f"\nUse this file for batch judging: {session_file}")
    print()


if __name__ == "__main__":
    main()
