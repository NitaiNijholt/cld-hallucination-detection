"""
RQ1 Phase 2: Judging on Pre-Generated Sessions

This script runs 7 different judging strategies on each of the 18 sessions
generated in Phase 1, without regenerating any edges.

Usage:
    python run_rq1_phase2_judging.py <phase1_experiment_id>

Example:
    python run_rq1_phase2_judging.py exp_20251008_143022_a1b2c3d4

This will:
1. Load the 18 session IDs from Phase 1
2. Run each of 7 judging strategies on each session
3. Total: 18 × 7 = 126 judging runs
4. Save results to phase2_judging_results/
"""

import sys
import os
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from modules import run_discovery_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 7 judging strategies to test
JUDGING_STRATEGIES = {
    # Serial strategies
    "serial_gpt4.1": {
        "judge_config": {"provider": "openai", "model": "gpt-4.1"},
        "num_judges": 1,
        "judge_models": ["gpt-4.1"]
    },
    "serial_claude3.7": {
        "judge_config": {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
        "num_judges": 1,
        "judge_models": ["claude-3-7-sonnet-20250219"]
    },
    "serial_sonar": {
        "judge_config": {"provider": "perplexity", "model": "sonar-pro"},
        "num_judges": 1,
        "judge_models": ["sonar-pro"]
    },
    
    # Homogeneous parallel strategies
    "parallel_3x_gpt4.1": {
        "judge_config": [
            {"provider": "openai", "model": "gpt-4.1"},
            {"provider": "openai", "model": "gpt-4.1"},
            {"provider": "openai", "model": "gpt-4.1"}
        ],
        "num_judges": 3,
        "judge_models": ["gpt-4.1", "gpt-4.1", "gpt-4.1"]
    },
    "parallel_3x_claude3.7": {
        "judge_config": [
            {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
            {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
            {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"}
        ],
        "num_judges": 3,
        "judge_models": ["claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219"]
    },
    "parallel_3x_sonar": {
        "judge_config": [
            {"provider": "perplexity", "model": "sonar-pro"},
            {"provider": "perplexity", "model": "sonar-pro"},
            {"provider": "perplexity", "model": "sonar-pro"}
        ],
        "num_judges": 3,
        "judge_models": ["sonar-pro", "sonar-pro", "sonar-pro"]
    },
    
    # Heterogeneous parallel strategy
    "parallel_heterogeneous": {
        "judge_config": [
            {"provider": "openai", "model": "gpt-4.1"},
            {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
            {"provider": "perplexity", "model": "sonar-pro"}
        ],
        "num_judges": 3,
        "judge_models": ["gpt-4.1", "claude-3-7-sonnet-20250219", "sonar-pro"]
    }
}


def load_phase1_sessions(phase1_exp_id: str, results_folder: str = "parameter_tuning_experiments/results"):
    """
    Load session IDs from Phase 1 experiment.
    
    Args:
        phase1_exp_id: Experiment ID from Phase 1 (e.g., "exp_20251008_143022_a1b2c3d4")
        results_folder: Path to results folder
    
    Returns:
        List of dicts with session info
    """
    # Load param combo mapping
    mapping_file = Path(results_folder) / f"param_combo_mapping_{phase1_exp_id}.csv"
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Phase 1 mapping file not found: {mapping_file}")
    
    df = pd.read_csv(mapping_file)
    logger.info(f"Loaded {len(df)} sessions from Phase 1")
    
    # Extract session info
    sessions = []
    for _, row in df.iterrows():
        sessions.append({
            "session_id": row["session_id"],
            "cld_name": row["cld_prefix"],
            "corruption_rate": row.get("corruption_rate", "unknown"),
            "run_number": row["run_number"],
            "excel_path": row.get("excel_file", "")
        })
    
    return sessions


def run_judging_on_session(
    session_id: str,
    excel_path: str,
    strategy_name: str,
    strategy_config: dict,
    output_dir: str,
    yaml_path: str = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"
):
    """
    Run a specific judging strategy on a pre-generated session.
    
    Args:
        session_id: Neo4j session ID from Phase 1
        excel_path: Path to ground truth Excel file
        strategy_name: Name of judging strategy
        strategy_config: Strategy configuration dict
        output_dir: Output directory for results
        yaml_path: Path to prompts YAML
    
    Returns:
        Experiment info dict
    """
    logger.info(f"Running {strategy_name} on session {session_id[:8]}...")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"rq1_phase2_{strategy_name}_{session_id[:8]}_{timestamp}.xlsx"
    result_path = Path(output_dir) / result_filename
    
    try:
        # Run judging on existing session
        exp_info = run_discovery_experiment(
            retrieved_session_id=session_id,  # ⭐ KEY: Reuse existing session
            excel_path=excel_path,
            yaml_path=yaml_path,
            judge_edges=True,  # Enable judging
            judge_config=strategy_config["judge_config"],
            num_judges=strategy_config["num_judges"],
            judge_models=strategy_config["judge_models"],
            result_excel_path=str(result_path),
            output_dir=output_dir,
            plot_session_graph=False,
            plot_validation_graph=False,
            export_json=True,
            embedding_enable=False,  # Don't need CI metrics for RQ1
            ci_compute_embeddings=False,  # ⚡ Disable expensive embeddings for faster processing
            citation_search_provider="brave"
        )
        
        logger.info(f"✅ Completed {strategy_name} on session {session_id[:8]}")
        return exp_info
        
    except Exception as e:
        logger.error(f"❌ Failed {strategy_name} on session {session_id[:8]}: {e}")
        return None


def main():
    """Main execution function."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_rq1_phase2_judging.py <phase1_experiment_id>")
        print("Example: python run_rq1_phase2_judging.py exp_20251008_143022_a1b2c3d4")
        sys.exit(1)
    
    phase1_exp_id = sys.argv[1]
    
    print("\n" + "="*80)
    print(" "*25 + "RQ1 PHASE 2: JUDGING")
    print("="*80)
    print(f"\nPhase 1 Experiment ID: {phase1_exp_id}")
    print(f"Judging strategies: {len(JUDGING_STRATEGIES)}")
    print("\n")
    
    # Load Phase 1 sessions
    print("Loading Phase 1 sessions...")
    sessions = load_phase1_sessions(phase1_exp_id)
    print(f"✓ Found {len(sessions)} sessions")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"parameter_tuning_experiments/rq1_phase2_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Track results
    results = []
    total_runs = len(sessions) * len(JUDGING_STRATEGIES)
    current_run = 0
    
    print(f"\nStarting {total_runs} judging runs ({len(sessions)} sessions × {len(JUDGING_STRATEGIES)} strategies)")
    print("="*80 + "\n")
    
    # Run each strategy on each session
    for session_info in sessions:
        session_id = session_info["session_id"]
        cld_name = session_info["cld_name"]
        corruption_rate = session_info["corruption_rate"]
        
        print(f"\nSession: {session_id[:8]} | CLD: {cld_name} | Corruption: {corruption_rate}")
        print("-" * 80)
        
        for strategy_name, strategy_config in JUDGING_STRATEGIES.items():
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] {strategy_name}...")
            
            exp_info = run_judging_on_session(
                session_id=session_id,
                excel_path=session_info["excel_path"],
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                output_dir=str(output_dir)
            )
            
            if exp_info:
                results.append({
                    "phase1_exp_id": phase1_exp_id,
                    "session_id": session_id,
                    "cld_name": cld_name,
                    "corruption_rate": corruption_rate,
                    "run_number": session_info["run_number"],
                    "judging_strategy": strategy_name,
                    "num_judges": strategy_config["num_judges"],
                    "status": "success"
                })
            else:
                results.append({
                    "phase1_exp_id": phase1_exp_id,
                    "session_id": session_id,
                    "cld_name": cld_name,
                    "corruption_rate": corruption_rate,
                    "run_number": session_info["run_number"],
                    "judging_strategy": strategy_name,
                    "num_judges": strategy_config["num_judges"],
                    "status": "failed"
                })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    summary_path = output_dir / "phase2_judging_summary.csv"
    results_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("✅ PHASE 2 JUDGING COMPLETE!")
    print("="*80)
    print(f"\nTotal runs: {total_runs}")
    print(f"Successful: {(results_df['status'] == 'success').sum()}")
    print(f"Failed: {(results_df['status'] == 'failed').sum()}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {summary_path}")
    print("\n" + "="*80)
    print("\nNext step: Run RQ1 analysis pipeline on Phase 2 results")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
