#!/usr/bin/env python3
"""
RQ1 Phase 1: Clone and Corrupt Single Session

Clones a base session and applies 50% corruption.

Usage:
    python run_rq1_phase1_single_corrupt.py <base_session_id> <excel_file>
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST
env_path = Path(__file__).parent.parent.parent / ".env.dev"
load_dotenv(env_path, override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import run_discovery_experiment

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_rq1_phase1_single_corrupt.py <base_session_id> <excel_file> [random_seed] [output_dir]")
        sys.exit(1)
    
    base_session_id = sys.argv[1]
    excel_file = sys.argv[2]
    random_seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    output_dir_arg = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("="*80)
    print("RQ1 PHASE 1: CLONE & CORRUPT")
    print("="*80)
    print(f"Base session: {base_session_id}")
    print(f"Excel file: {excel_file}")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
    print()
    
    # Clone session
    print("Cloning base session...")
    cloner = SessionCloner()
    corrupted_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        include_judge_data=False
    )
    print(f"✅ Cloned: {base_session_id[:8]}... → {corrupted_session_id[:8]}...")
    print()
    
    # Apply 30% corruption
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided output directory or create default one
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(__file__).parent / "results" / f"rq1_corruption_experiment_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print("Applying 30% corruption (mixed: motivation/spurious/flip)...")
    print()
    
    # Use absolute path for YAML file to avoid path issues with uv run
    yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_Nitai_C.yaml"
    
    result = run_discovery_experiment(
        retrieved_session_id=corrupted_session_id,
        excel_path=excel_file,
        yaml_path=str(yaml_path),
        generator_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},
        corruption_rate=0.3,  # 30% corruption rate
        corruption_seed=random_seed,  # Use seed for reproducible corruption
        corruptor_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},  # LLM-powered corruption
        judge_edges=False,
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.3},
        result_excel_path=str(output_dir / f"corrupted_{corrupted_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        embedding_enable=False,  # Disabled to avoid OpenAI quota issues
        citation_search_provider="brave"
    )
    
    print()
    print("="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"Corrupted session ID: {corrupted_session_id}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Save comprehensive corruption metadata for reproducibility
    import json
    import platform
    
    session_info_file = output_dir / "session_info.txt"
    with open(session_info_file, "w") as f:
        f.write(f"base_session_id={base_session_id}\n")
        f.write(f"corrupted_session_id={corrupted_session_id}\n")
        f.write(f"excel_file={excel_file}\n")
        f.write(f"timestamp={timestamp}\n")
    
    # Save detailed JSON metadata
    metadata_file = output_dir / "corruption_metadata.json"
    metadata = {
        'experiment_info': {
            'experiment_type': 'corruption',
            'timestamp': datetime.now().isoformat(),
            'script_name': Path(__file__).name,
            'script_version': '1.0',
        },
        'input_parameters': {
            'base_session_id': base_session_id,
            'excel_file': excel_file,
            'random_seed': random_seed,
            'output_dir': str(output_dir),
        },
        'corruption_configuration': {
            'corruption_rate': 0.3,
            'corruption_seed': random_seed,
            'corruptor_provider': 'openai',
            'corruptor_model': 'gpt-4.1',
            'corruptor_temperature': 0.7,
        },
        'generator_configuration': {
            'provider': 'openai',
            'model': 'gpt-4.1',
            'temperature': 0.7,
        },
        'output': {
            'corrupted_session_id': corrupted_session_id,
            'result_excel': str(output_dir / f"corrupted_{corrupted_session_id[:8]}.xlsx"),
        },
        'system_info': {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
        }
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Session info saved to: {session_info_file}")
    print(f"Metadata saved to: {metadata_file}")
    print()
    print("Next steps:")
    print(f"  1. Judge without correction:")
    print(f"     python parameter_tuning_experiments/run_rq1_phase2_judge_single.py \\")
    print(f"       {corrupted_session_id} \\")
    print(f"       {excel_file} \\")
    print(f"       {output_dir}")
    print()
    print(f"  2. Judge with correction:")
    print(f"     python parameter_tuning_experiments/run_rq1_phase2_correct_single.py \\")
    print(f"       {corrupted_session_id} \\")
    print(f"       {excel_file} \\")
    print(f"       {output_dir}")

if __name__ == "__main__":
    main()
