#!/usr/bin/env python3
"""
RQ1 Phase 2: Apply Judging Strategies to Corrupted Sessions

This script:
1. Loads corrupted session IDs from Phase 1
2. Applies 3 judging strategies to each session:
   - Serial Claude-3.7 (baseline)
   - Homogeneous 3× Claude (redundancy test)
   - Heterogeneous GPT+Claude+Sonar (diversity test)
3. Saves results for analysis

Usage:
    python run_rq1_phase2_judge.py <phase1_directory_name>
    python run_rq1_phase2_judge.py <phase1_directory_name> --correctness
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env.dev")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment
from neo4j_session_cloner import SessionCloner

# 3 Judging strategies to test
JUDGING_STRATEGIES = {
    # Strategy 1: Serial Baseline (Claude Sonnet 4.5)
    "serial_baseline": {
        "judge_config": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "temperature": 0.3
        },
        "num_judges": 1,
        "judge_models": ["claude-sonnet-4-5-20250929"],
        "description": "Single Claude Sonnet 4.5 judge (baseline)"
    },
    
    # Strategy 2: Homogeneous Ensemble (3× Claude Sonnet 4.5)
    "homogeneous_ensemble": {
        "judge_config": [
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "temperature": 0.3},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "temperature": 0.3},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "temperature": 0.3}
        ],
        "num_judges": 3,
        "judge_models": ["claude-sonnet-4-5-20250929"] * 3,
        "description": "3× Claude Sonnet 4.5 ensemble (redundancy test)"
    },
    
    # Strategy 3: Heterogeneous Ensemble (GPT + Claude + Sonar)
    "heterogeneous_ensemble": {
        "judge_config": [
            {"provider": "openai", "model": "gpt-4.1", "temperature": 0.3},
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "temperature": 0.3},
            {"provider": "perplexity", "model": "sonar-pro", "temperature": 0.3}
        ],
        "num_judges": 3,
        "judge_models": ["gpt-4.1", "claude-sonnet-4-5-20250929", "sonar-pro"],
        "description": "GPT+Claude Sonnet 4.5+Sonar ensemble (diversity test)"
    }
}

def load_phase1_sessions(phase1_dir: str) -> pd.DataFrame:
    """Load corrupted session IDs from Phase 1."""
    # Handle both directory name and full CSV path
    input_path = Path(phase1_dir)
    
    if input_path.suffix == '.csv':
        # User provided full CSV path
        mapping_file = input_path
        results_dir = mapping_file.parent
    else:
        # User provided directory name (e.g., "rq1_phase1_20251010_010857")
        results_dir = Path("parameter_tuning_experiments/results") / phase1_dir
        
        if not results_dir.exists():
            raise FileNotFoundError(f"Phase 1 directory not found: {results_dir}")
        
        # Find mapping file
        mapping_files = list(results_dir.glob("phase1_session_mapping_*.csv"))
        if not mapping_files:
            raise FileNotFoundError(f"No mapping file found in {results_dir}")
        
        mapping_file = mapping_files[0]
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    df = pd.read_csv(mapping_file)
    print(f"✅ Loaded {len(df)} corrupted sessions from Phase 1")
    return df, results_dir

def apply_judging(
    corrupted_session_id: str,
    excel_file: str,
    strategy_name: str,
    strategy_config: dict,
    output_dir: Path,
    cloner: SessionCloner,
    judge_approach: str = "per_citation_aggregate"
) -> dict:
    """Apply a judging strategy to a cloned copy of the corrupted session."""
    
    # Step 1: Clone the corrupted session for this judging strategy
    print(f"  📋 Cloning session for {strategy_config['description']}...")
    judged_session_id = cloner.clone_session(
        original_session_id=corrupted_session_id,
        include_judge_data=False  # Exclude any existing judge data
    )
    print(f"     Cloned: {corrupted_session_id[:8]}... → {judged_session_id[:8]}...")
    
    excel_path = f"parameter_tuning_experiments/ground_truth_clds_for_experiments/{excel_file}"
    yaml_path = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"
    
    result_path = output_dir / f"judged_{judged_session_id}_{strategy_name}.xlsx"
    
    print(f"  🔍 Judging with {strategy_config['description']} (approach={judge_approach})...")
    
    # Step 2: Apply judging to the cloned session
    exp_info = run_discovery_experiment(
        retrieved_session_id=judged_session_id,  # Use the fresh clone
        excel_path=excel_path,
        yaml_path=yaml_path,
        
        # Generator config (required but not used - session already exists)
        generator_config={
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7
        },
        
        # Corruptor config (required but not used - corruption already applied in Phase 1)
        corruption_rate=0.0,
        corruptor_config={
            "provider": "perplexity",
            "model": "mistral-7b-instruct",
            "temperature": 0.7
        },
        
        # Judging settings
        judge_edges=True,
        judge_approach=judge_approach,  # Pass the judging approach
        judge_enable_web_search=(judge_approach != "correctness"),  # Disable web search for correctness mode
        judge_config=strategy_config["judge_config"],
        num_judges=strategy_config["num_judges"],
        judge_models=strategy_config["judge_models"],
        parallel=True,
        
        # Output settings
        result_excel_path=str(result_path),
        output_dir=str(output_dir),
        plot_session_graph=False,
        plot_validation_graph=False,
        export_json=True,
        
        # CI metrics and citation search - disable for correctness mode
        embedding_enable=(judge_approach != "correctness"),  # Don't compute CI metrics in correctness mode
        ci_compute_embeddings=False,  # ⚡ Disable expensive embeddings for faster processing
        citation_search_provider="brave" if judge_approach != "correctness" else None  # No citations in correctness mode
    )
    
    print(f"  ✅ Judging complete")
    
    return {
        'result_file': str(result_path),
        'corrupted_session_id': corrupted_session_id,
        'judged_session_id': judged_session_id,
        'strategy': strategy_name
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='RQ1 Phase 2: Apply Judging Strategies to Corrupted Sessions'
    )
    parser.add_argument('phase1_dir', help='Phase 1 directory name or path')
    parser.add_argument(
        '--correctness', 
        action='store_true',
        help='Use correctness-based judging (no citations) instead of citation-based judging'
    )
    args = parser.parse_args()
    
    phase1_dir = args.phase1_dir
    correctness_mode = args.correctness
    
    # Determine judge approach
    judge_approach = "correctness" if correctness_mode else "per_citation_aggregate"
    mode_label = "CORRECTNESS-BASED" if correctness_mode else "CITATION-BASED"
    
    print("=" * 80)
    print(f"RQ1 PHASE 2: {mode_label} JUDGING")
    print("=" * 80)
    print(f"Phase 1 Directory: {phase1_dir}")
    print(f"Judging Approach: {judge_approach}")
    print()
    
    # Load Phase 1 corrupted sessions
    corrupted_df, phase1_path = load_phase1_sessions(phase1_dir)
    
    # Initialize Neo4j session cloner
    print("Initializing Neo4j session cloner...")
    cloner = SessionCloner()
    print("✅ Session cloner ready\n")
    
    # Create output directory for Phase 2 with mode suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_correctness" if correctness_mode else ""
    phase2_output_dir = Path(f"parameter_tuning_experiments/results/rq1_phase2{mode_suffix}_{timestamp}")
    phase2_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all judging results
    judging_results = []
    
    print("\n" + "=" * 80)
    print("APPLYING JUDGING STRATEGIES")
    print("=" * 80)
    print(f"Strategies to test: {len(JUDGING_STRATEGIES)}")
    print(f"Sessions to judge: {len(corrupted_df)}")
    print(f"Total experiments: {len(JUDGING_STRATEGIES) * len(corrupted_df)}")
    print()
    
    total_experiments = len(corrupted_df) * len(JUDGING_STRATEGIES)
    experiment_num = 0
    
    for session_idx, row in corrupted_df.iterrows():
        corrupted_session_id = row['corrupted_session_id']
        excel_file = row['excel_file']
        run_number = row['run']
        
        print(f"\n{'='*80}")
        print(f"Session {session_idx+1}/{len(corrupted_df)}: {excel_file} (Run {run_number})")
        print(f"Corrupted session: {corrupted_session_id}")
        print(f"{'='*80}")
        
        for strategy_name, strategy_config in JUDGING_STRATEGIES.items():
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Strategy: {strategy_name}")
            
            result = apply_judging(
                corrupted_session_id=corrupted_session_id,
                excel_file=excel_file,
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                output_dir=phase2_output_dir,
                cloner=cloner,
                judge_approach=judge_approach
            )
            
            # Record result
            judging_results.append({
                'base_session_id': row['base_session_id'],
                'corrupted_session_id': corrupted_session_id,
                'judged_session_id': result['judged_session_id'],
                'excel_file': excel_file,
                'run': run_number,
                'corruption_rate': row['corruption_rate'],
                'judging_strategy': strategy_name,
                'judge_description': strategy_config['description'],
                'num_judges': strategy_config['num_judges'],
                'result_file': result['result_file'],
                'phase1_dir': phase1_dir,
                'phase2_timestamp': timestamp
            })
    
    # Save mapping for analysis
    results_df = pd.DataFrame(judging_results)
    mapping_file = phase2_output_dir / f"phase2_judging_results_{timestamp}.csv"
    results_df.to_csv(mapping_file, index=False)
    
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"✅ Completed {len(judging_results)} judging experiments")
    print(f"   - {len(corrupted_df)} sessions")
    print(f"   - {len(JUDGING_STRATEGIES)} strategies")
    print(f"✅ Results saved to: {mapping_file}")
    
    # Print strategy breakdown
    print(f"\nResults by strategy:")
    for strategy in JUDGING_STRATEGIES.keys():
        count = len([r for r in judging_results if r['judging_strategy'] == strategy])
        print(f"  - {strategy}: {count} experiments")
    
    # Close Neo4j connection
    cloner.close()
    
    print(f"\n📊 Next step: Run RQ1 analysis")
    print(f"Command:")
    print(f"  cd parameter_tuning_experiments/analysis")
    print(f"  python rq1_master_pipeline.py \\")
    print(f"    --phase2_dir {phase2_output_dir.name} \\")
    print(f"    --bootstrap 10000 \\")
    print(f"    --permutations 10000")

if __name__ == "__main__":
    main()
