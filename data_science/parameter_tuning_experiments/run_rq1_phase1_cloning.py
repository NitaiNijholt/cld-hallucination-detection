"""
RQ1 Phase 1: Clone Base CLDs and Apply Corruption

This script takes the 6 clean base CLDs from Phase 0 and creates
experimental variants by cloning + applying different corruption rates.

Usage:
    python run_rq1_phase1_cloning.py <phase0_experiment_id>

Example:
    python run_rq1_phase1_cloning.py exp_20251008_143022_a1b2c3d4

This will:
1. Load the 6 base session IDs from Phase 0
2. Clone each base 2 times (12 clones total)
3. Apply 50% corruption to 6 clones
4. Apply 100% corruption to 6 clones
5. Save mapping for Phase 2 judging
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from neo4j_session_cloner import clone_session
from modules import CausalDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_sessions(phase0_exp_id: str, results_folder: str = "parameter_tuning_experiments/results"):
    """
    Load base session IDs from Phase 0.
    
    Args:
        phase0_exp_id: Experiment ID from Phase 0
        results_folder: Path to results folder
    
    Returns:
        DataFrame with base session info
    """
    mapping_file = Path(results_folder) / f"param_combo_mapping_{phase0_exp_id}.csv"
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Phase 0 mapping file not found: {mapping_file}")
    
    df = pd.read_csv(mapping_file)
    
    # Verify these are clean CLDs (corruption_rate = 0.0)
    if 'corruption_rate' in df.columns:
        non_zero = df[df['corruption_rate'] != 0.0]
        if len(non_zero) > 0:
            logger.warning(f"Warning: Found {len(non_zero)} sessions with non-zero corruption!")
    
    logger.info(f"Loaded {len(df)} base sessions from Phase 0")
    return df


def apply_corruption_to_session(
    session_id: str,
    corruption_rate: float,
    excel_path: str,
    yaml_path: str = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"
):
    """
    Apply corruption to a cloned session.
    
    Args:
        session_id: Neo4j session ID
        corruption_rate: Corruption rate (0.0 to 1.0)
        excel_path: Path to ground truth Excel file
        yaml_path: Path to prompts YAML
    
    Returns:
        Number of edges corrupted
    """
    from modules import run_discovery_experiment
    
    logger.info(f"  Applying {corruption_rate*100}% corruption to session {session_id[:8]}...")
    
    # Load environment for API keys
    load_dotenv("../.env.dev")
    
    # Get corruptor config
    corruptor_config = {
        "provider": "perplexity",
        "model": "mistral-7b-instruct",
        "temperature": 0.7
    }
    
    try:
        # Run corruption on existing session
        # We use run_discovery_experiment but with retrieved_session_id set
        # This loads the existing session and applies corruption
        exp_info = run_discovery_experiment(
            retrieved_session_id=session_id,  # Reuse existing session
            excel_path=excel_path,
            yaml_path=yaml_path,
            corruption_rate=corruption_rate,  # Apply corruption
            corruptor_config=corruptor_config,
            judge_edges=False,  # Don't judge yet
            export_json=False,
            plot_session_graph=False,
            plot_validation_graph=False,
            embedding_enable=False
        )
        
        corrupted_count = exp_info.get("corrupted_count", 0)
        logger.info(f"  ✅ Corrupted {corrupted_count} edges")
        return corrupted_count
        
    except Exception as e:
        logger.error(f"  ❌ Failed to apply corruption: {e}")
        raise


def main():
    """Main execution function."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_rq1_phase1_cloning.py <phase0_experiment_id>")
        print("Example: python run_rq1_phase1_cloning.py exp_20251008_143022_a1b2c3d4")
        sys.exit(1)
    
    phase0_exp_id = sys.argv[1]
    
    print("\n" + "="*80)
    print(" "*20 + "RQ1 PHASE 1: CLONING & CORRUPTION")
    print("="*80)
    print(f"\nPhase 0 Experiment ID: {phase0_exp_id}")
    print(f"Corruption rates: 0.5, 1.0")
    print("\n")
    
    # Load Phase 0 base sessions
    print("Loading Phase 0 base sessions...")
    base_sessions = load_base_sessions(phase0_exp_id)
    print(f"✓ Found {len(base_sessions)} base sessions")
    
    # Load environment
    load_dotenv("../.env.dev")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Track all cloned sessions
    cloned_sessions = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nCreating {len(base_sessions) * 2} clones...")
    print("="*80 + "\n")
    
    # For each base session, create 2 clones with different corruption
    for idx, row in base_sessions.iterrows():
        base_session_id = row["session_id"]
        cld_name = row.get("cld_prefix", "unknown")
        run_number = row.get("run_number", 1)
        excel_file = row.get("excel_file", "")
        
        print(f"\n[{idx+1}/{len(base_sessions)}] Processing: {cld_name} (run {run_number})")
        print(f"Base session: {base_session_id[:8]}")
        print("-" * 80)
        
        # Clone 1: 50% corruption
        print(f"\n1️⃣  Creating 50% corruption variant...")
        clone_50_id = clone_session(
            original_session_id=base_session_id,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            include_judge_data=False
        )
        
        # Apply 50% corruption to clone
        corrupted_count_50 = apply_corruption_to_session(
            session_id=clone_50_id,
            corruption_rate=0.5,
            excel_path=excel_file
        )
        
        cloned_sessions.append({
            "base_session_id": base_session_id,
            "cloned_session_id": clone_50_id,
            "cld_name": cld_name,
            "corruption_rate": 0.5,
            "run_number": run_number,
            "excel_file": excel_file,
            "corrupted_edges": corrupted_count_50,
            "clone_timestamp": timestamp
        })
        
        # Clone 2: 100% corruption
        print(f"\n2️⃣  Creating 100% corruption variant...")
        clone_100_id = clone_session(
            original_session_id=base_session_id,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            include_judge_data=False
        )
        
        # Apply 100% corruption to clone
        corrupted_count_100 = apply_corruption_to_session(
            session_id=clone_100_id,
            corruption_rate=1.0,
            excel_path=excel_file
        )
        
        cloned_sessions.append({
            "base_session_id": base_session_id,
            "cloned_session_id": clone_100_id,
            "cld_name": cld_name,
            "corruption_rate": 1.0,
            "run_number": run_number,
            "excel_file": excel_file,
            "corrupted_edges": corrupted_count_100,
            "clone_timestamp": timestamp
        })
        
        print(f"✅ Created 2 clones for {cld_name} (run {run_number})")
    
    # Save mapping
    output_dir = Path("parameter_tuning_experiments/results")
    output_file = output_dir / f"cloned_sessions_{phase0_exp_id}_{timestamp}.csv"
    
    df_clones = pd.DataFrame(cloned_sessions)
    df_clones.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("✅ PHASE 1 CLONING COMPLETE!")
    print("="*80)
    print(f"\nBase sessions: {len(base_sessions)}")
    print(f"Cloned sessions: {len(cloned_sessions)}")
    print(f"Total sessions for judging: {len(cloned_sessions)}")
    print(f"\nMapping saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY CORRUPTION RATE")
    print("="*80)
    summary = df_clones.groupby('corruption_rate').agg({
        'cloned_session_id': 'count',
        'corrupted_edges': 'sum'
    })
    print(summary)
    
    print("\n" + "="*80)
    print("NEXT STEP: Run Phase 2 Judging")
    print("="*80)
    print(f"\npython run_rq1_phase2_judging.py --clones-csv {output_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
