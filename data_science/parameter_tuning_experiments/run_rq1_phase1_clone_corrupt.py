#!/usr/bin/env python3
"""
RQ1 Phase 1: Clone Base Sessions and Apply Corruption

This script:
1. Loads clean base CLD sessions from Phase 0
2. Clones each session once (will apply 50% corruption)
3. Applies corruption to the clones using the corruptor agent
4. Saves a mapping CSV for Phase 2 judging

Usage:
    python run_rq1_phase1_clone_corrupt.py <phase0_experiment_id>
"""

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env.dev")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import run_discovery_experiment
from backend.db_clients.neo4j_client import Neo4jClient

# Corruptor configuration (Mistral 7B - different from all judges)
CORRUPTOR_CONFIG = {
    "provider": "perplexity",
    "model": "mistral-7b-instruct",
    "temperature": 0.7
}

def load_phase0_sessions(experiment_id: str) -> pd.DataFrame:
    """Load base session IDs from Phase 0."""
    results_dir = Path("parameter_tuning_experiments/results")
    mapping_file = results_dir / f"param_combo_mapping_{experiment_id}.csv"
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Phase 0 mapping file not found: {mapping_file}")
    
    df = pd.read_csv(mapping_file)
    
    # Extract session_id from each Excel file's Params sheet
    session_ids = []
    for idx, row in df.iterrows():
        # The excel_filename in CSV might not have the timestamp, so find the actual file
        excel_path_base = results_dir / row['excel_filename']
        excel_dir = excel_path_base.parent
        
        # Find any .xlsx file in that directory
        if excel_dir.exists():
            excel_files = list(excel_dir.glob("*.xlsx"))
            if excel_files:
                excel_path = excel_files[0]  # Take the first (should be only one)
                params = pd.read_excel(excel_path, sheet_name='Params')
                session_id_row = params[params['Parameter'] == 'session_id']
                if not session_id_row.empty:
                    session_id = session_id_row.iloc[0]['Value']
                    session_ids.append(session_id)
                else:
                    raise ValueError(f"No session_id found in {excel_path}")
            else:
                raise FileNotFoundError(f"No Excel files found in {excel_dir}")
        else:
            raise FileNotFoundError(f"Directory not found: {excel_dir}")
    
    df['session_id'] = session_ids
    df['excel_file'] = df['cld_prefix'].apply(lambda x: f"{x}.xlsx")
    df['run'] = df['run_idx']
    
    print(f"✅ Loaded {len(df)} base sessions from Phase 0")
    return df

def clone_session(base_session_id: str) -> str:
    """Clone a Neo4j session and return new session ID."""
    cloner = SessionCloner()
    new_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        include_judge_data=False  # Don't copy any judge data
    )
    print(f"  ✅ Cloned {base_session_id} → {new_session_id}")
    return new_session_id

def apply_corruption(
    session_id: str,
    excel_file: str,
    corruption_rate: float,
    output_dir: Path
) -> int:
    """Apply corruption to an existing session."""
    
    excel_path = f"parameter_tuning_experiments/ground_truth_clds_for_experiments/{excel_file}"
    yaml_path = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"
    
    result_path = output_dir / f"corrupted_{session_id}_cr{int(corruption_rate*100)}.xlsx"
    
    print(f"  🔧 Applying {corruption_rate*100}% corruption...")
    
    # Run corruption on existing session (no generation, no judging)
    exp_info = run_discovery_experiment(
        retrieved_session_id=session_id,  # Reuse cloned session
        excel_path=excel_path,
        yaml_path=yaml_path,
        
        # Generator config (required even though we're reusing session)
        generator_config={
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7
        },
        
        # Corruption settings
        corruption_rate=corruption_rate,
        corruptor_config=CORRUPTOR_CONFIG,
        
        # Disable judging (Phase 2 will handle this)
        judge_edges=False,
        judge_config={"provider": "anthropic", "model": "claude-3-7-sonnet-20250219", "temperature": 0.3},  # Dummy config (not used)
        
        # Output settings
        result_excel_path=str(result_path),
        output_dir=str(output_dir),
        plot_session_graph=False,
        plot_validation_graph=False,
        export_json=True,
        
        # CI metrics (needed for RQ2)
        embedding_enable=True,
        citation_search_provider="brave"
    )
    
    corrupted_count = exp_info.get("corrupted_edges_count", 0)
    print(f"  ✅ Corrupted {corrupted_count} edges")
    
    return corrupted_count

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_rq1_phase1_clone_corrupt.py <phase0_experiment_id>")
        sys.exit(1)
    
    phase0_exp_id = sys.argv[1]
    
    print("=" * 80)
    print("RQ1 PHASE 1: Clone & Corrupt")
    print("=" * 80)
    print(f"Phase 0 Experiment ID: {phase0_exp_id}")
    print()
    
    # Load Phase 0 base sessions
    base_df = load_phase0_sessions(phase0_exp_id)
    
    # Create output directory for Phase 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase1_output_dir = Path(f"parameter_tuning_experiments/results/rq1_phase1_{timestamp}")
    phase1_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all cloned sessions
    cloned_sessions = []
    
    print("\n" + "=" * 80)
    print("CLONING & CORRUPTING SESSIONS")
    print("=" * 80)
    
    for idx, row in base_df.iterrows():
        base_session_id = row['session_id']
        excel_file = row['excel_file']
        run_number = row['run']
        
        print(f"\n[{idx+1}/{len(base_df)}] Processing: {excel_file} (Run {run_number})")
        print(f"  Base session: {base_session_id}")
        
        # Clone session
        cloned_id = clone_session(base_session_id)
        
        # Apply 50% corruption to clone
        corrupted_count = apply_corruption(
            session_id=cloned_id,
            excel_file=excel_file,
            corruption_rate=0.5,
            output_dir=phase1_output_dir
        )
        
        # Record mapping
        cloned_sessions.append({
            'base_session_id': base_session_id,
            'corrupted_session_id': cloned_id,
            'excel_file': excel_file,
            'run': run_number,
            'corruption_rate': 0.5,
            'corrupted_edges_count': corrupted_count,
            'phase0_exp_id': phase0_exp_id,
            'phase1_timestamp': timestamp
        })
    
    # Save mapping for Phase 2
    mapping_df = pd.DataFrame(cloned_sessions)
    mapping_file = phase1_output_dir / f"phase1_session_mapping_{timestamp}.csv"
    mapping_df.to_csv(mapping_file, index=False)
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print(f"✅ Cloned and corrupted {len(cloned_sessions)} sessions")
    print(f"✅ Session mapping saved to: {mapping_file}")
    print(f"\nNext step: Run Phase 2 judging")
    print(f"Command:")
    print(f"  python parameter_tuning_experiments/run_rq1_phase2_judge.py {phase1_output_dir.name}")

if __name__ == "__main__":
    main()
