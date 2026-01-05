#!/usr/bin/env python3
"""
RQ1 Phase 2b: Judge WITH Correction

Judges corrupted edges and applies correction to fix them, then re-judges.

Usage:
    python run_rq1_phase2_correct_single.py <corrupted_session_id> <excel_file> <output_dir>
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import run_discovery_experiment

def main():
    if len(sys.argv) < 4:
        print("Usage: python run_rq1_phase2_correct_single.py <corrupted_session_id> <excel_file> <output_dir>")
        sys.exit(1)
    
    corrupted_session_id = sys.argv[1]
    excel_file = sys.argv[2]
    output_dir = Path(sys.argv[3])
    
    print("="*80)
    print("RQ1 PHASE 2b: JUDGE WITH CORRECTION")
    print("="*80)
    print(f"Corrupted session: {corrupted_session_id}")
    print(f"Excel file: {excel_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Clone corrupted session for correction
    print("Cloning corrupted session for correction...")
    cloner = SessionCloner()
    corrected_session_id = cloner.clone_session(
        original_session_id=corrupted_session_id,
        include_judge_data=False
    )
    print(f"✅ Cloned: {corrupted_session_id[:8]}... → {corrected_session_id[:8]}...")
    print()
    
    # Judge + Correct with agent-based corrector (pydantic_ai)
    print("Judging with gpt-4.1 + Agent-based Correction with gpt-4.1...")
    print("Will process 10 edges in parallel for judging")
    print("Agent corrector tools: revise_motivation, flip_polarity, change_edge_type, remove_edge")
    print("After correction, edges will be re-judged in parallel")
    print()
    
    result = run_discovery_experiment(
        retrieved_session_id=corrected_session_id,
        excel_path=excel_file,
        yaml_path="parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml",
        generator_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},
        corruptor_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.7},
        corruption_rate=0.0,
        judge_edges=True,
        judge_approach="correctness",  # Judge based on logical correctness, not citations
        judge_enable_web_search=False,
        judge_parallel=True,  # Enable parallel judging for speed
        judge_max_workers=10,  # Process 10 edges in parallel
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.3},
        num_judges=1,
        judge_models=["gpt-4.1"],
        run_correction=True,  # KEY: Run correction
        rejudge_approach="correctness",  # Re-judge after correction using correctness mode
        rejudge_parallel=True,  # Enable parallel re-judging for speed
        rejudge_max_workers=10,  # Process 10 edges in parallel during re-judging
        corrector_models=["gpt-4.1"],  # Agent-based corrector with pydantic_ai
        correction_rounds=1,
        result_excel_path=str(output_dir / f"corrected_{corrected_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        embedding_enable=True,
        ci_compute_embeddings=False,
        citation_search_provider="brave"
    )
    
    print()
    print("="*80)
    print("PHASE 2b COMPLETE")
    print("="*80)
    print(f"Corrected session ID: {corrected_session_id}")
    print(f"Result file: {result.get('result_excel_path')}")
    print()
    
    # Append session info
    session_info_file = output_dir / "session_info.txt"
    with open(session_info_file, "a") as f:
        f.write(f"corrected_session_id={corrected_session_id}\n")
    
    print(f"Session info updated: {session_info_file}")

if __name__ == "__main__":
    main()
