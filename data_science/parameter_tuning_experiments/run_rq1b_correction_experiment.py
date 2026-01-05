#!/usr/bin/env python3
"""
RQ1b Correction Experiment Runner

This script:
1. Takes an existing CORRUPTED session ID (from RQ1a output).
2. Runs the Corrector on it (producing a Corrected session).
3. Calculates metrics comparing Corrupted vs. Base and Corrected vs. Base.

Usage:
    python run_rq1b_correction_experiment.py <corrupted_session_id> <base_session_id> <excel_file> <output_dir>
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import run_discovery_experiment

def main():
    if len(sys.argv) < 5:
        print("Usage: python run_rq1b_correction_experiment.py <corrupted_session_id> <base_session_id> <excel_file> <output_dir>")
        sys.exit(1)
    
    corrupted_session_id = sys.argv[1]
    base_session_id = sys.argv[2]
    excel_file = sys.argv[3]
    output_dir = Path(sys.argv[4])
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RQ1b: CORRECTION EXPERIMENT")
    print("="*80)
    print(f"Corrupted Session: {corrupted_session_id}")
    print(f"Base Session (GT): {base_session_id}")
    print(f"Excel File: {excel_file}")
    print(f"Output Dir: {output_dir}")
    print()
    
    # 1. Clone Corrupted Session for Correction
    print("1. Cloning corrupted session for correction...")
    cloner = SessionCloner()
    corrected_session_id = cloner.clone_session(
        original_session_id=corrupted_session_id,
        include_judge_data=False # Don't copy judge verdicts, we will re-judge or let corrector judge
    )
    print(f"✅ Cloned: {corrupted_session_id[:8]}... → {corrected_session_id[:8]}...")
    print()
    
    # 2. Run Correction (Judge -> Correct -> Re-judge)
    print("2. Running Correction Pipeline (Judge -> Correct -> Re-judge)...")
    # Use a known valid prompt file
    prompts_path = "parameter_tuning_experiments/alternative_prompts/prompts_citation_mechanistic_lit.yaml"
    
    result = run_discovery_experiment(
        retrieved_session_id=corrected_session_id,
        excel_path=excel_file,
        yaml_path=prompts_path,
        generator_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},
        corruptor_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.7}, # Not used, but required config
        corruption_rate=0.0, # No new corruption
        
        # Judge settings
        judge_edges=True,
        judge_approach="correctness",
        judge_enable_web_search=False,
        judge_parallel=True,
        judge_max_workers=10,
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0},
        num_judges=1,
        judge_models=["gpt-4.1"],
        
        # Correction settings
        run_correction=True,
        corrector_models=["gpt-4.1"],
        correction_rounds=1,
        use_simple_corrector=False,
        corrector_prompt_variant="cot", # Use CoT to fix 'over-deletion' issue
        
        # Re-judge settings
        rejudge_approach="correctness",
        rejudge_parallel=True,
        rejudge_max_workers=10,
        
        # Output
        result_excel_path=str(output_dir / f"corrected_{corrected_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        
        # Metrics
        embedding_enable=True,
        ci_compute_embeddings=False,
        citation_search_provider="brave"
    )
    
    print(f"✅ Correction complete. Result saved to {result.get('result_excel_path')}")
    print()
    
    # 3. Compare Metrics (Calculate F1 Improvement)
    print("3. Calculating F1 Improvement...")
    
    # Helper to calculate metrics
    def calculate_metrics(pred_file, gt_session_id):
        # This is a simplified placeholder. Ideally we'd load the Base session edges 
        # from Neo4j or an Excel file if we had one for the base session readily available.
        # However, we can use the 'analyze_corruption_recovery.py' logic if we have the base Excel.
        # Since we don't strictly have the base Excel here (only session ID), we might skip precise calculation 
        # in this script and rely on the analysis script later.
        pass

    print("   (Metrics calculation deferred to analysis script)")
    
    # Save session info for the analysis script
    session_info_file = output_dir / "session_info.txt"
    with open(session_info_file, "w") as f:
        f.write(f"base_session_id={base_session_id}\n")
        f.write(f"corrupted_session_id={corrupted_session_id}\n")
        f.write(f"corrected_session_id={corrected_session_id}\n")
        f.write(f"excel_file={excel_file}\n")
        f.write(f"timestamp={datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
    
    print(f"✅ Session info saved to {session_info_file}")
    print("="*80)

if __name__ == "__main__":
    main()

