"""
Run RQ1b Phase 2b with SIMPLE corrector (only add/remove tools, no revise/flip/change).

This tests whether sophisticated tools add value vs. simple add/remove operations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment
from neo4j_session_cloner import SessionCloner
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run RQ1b Phase 2b with simple corrector")
    parser.add_argument("corrupted_session_id", help="Session ID of corrupted CLD")
    parser.add_argument("excel_file", help="Path to ground truth Excel file")
    parser.add_argument("output_dir", help="Output directory for results")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("RQ1b PHASE 2b: SIMPLE CORRECTOR (ADD/REMOVE ONLY)")
    print("="*80)
    print(f"Corrupted session: {args.corrupted_session_id}")
    print(f"Excel file: {args.excel_file}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Clone corrupted session for correction
    print("Cloning corrupted session for correction...")
    cloner = SessionCloner()
    corrected_session_id = cloner.clone_session(
        original_session_id=args.corrupted_session_id,
        include_judge_data=False
    )
    print(f"✅ Cloned: {args.corrupted_session_id[:8]}... → {corrected_session_id[:8]}...")
    print()
    
    # Run correction with SIMPLE corrector (only add/remove tools)
    print("Judging with gpt-4o + SIMPLE Corrector with gpt-4o...")
    print("Simple corrector tools: remove_edge, create_edge")
    print("NOT available: revise_motivation, flip_polarity, change_edge_type")
    print()
    
    result = run_discovery_experiment(
        retrieved_session_id=corrected_session_id,
        excel_path=args.excel_file,
        generator_config={"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
        corruptor_config={"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
        corruption_rate=0.0,
        judge_edges=True,
        judge_approach="correctness",
        judge_enable_web_search=False,
        judge_parallel=True,
        judge_max_workers=20,
        judge_config={"provider": "openai", "model": "gpt-4o", "temperature": 0.3},
        num_judges=1,
        judge_models=["gpt-4o"],
        run_correction=True,
        rejudge_approach="correctness",
        rejudge_parallel=True,
        rejudge_max_workers=20,
        corrector_models=["gpt-4o"],
        correction_rounds=1,
        use_simple_corrector=True,  # <-- KEY: Use simple add/remove only
        result_excel_path=str(output_dir / f"simple_corrected_{corrected_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        embedding_enable=True,
        ci_compute_embeddings=False,
        citation_search_provider="brave"
    )
    
    print()
    print("="*80)
    print("PHASE 2b COMPLETE (SIMPLE CORRECTOR)")
    print("="*80)
    print(f"Corrected session ID: {corrected_session_id}")
    print(f"Result file: {result.get('result_excel_path')}")
    print()
    
    # Append session info
    session_info_file = output_dir / "session_info.txt"
    with open(session_info_file, "a") as f:
        f.write(f"simple_corrected_session_id={corrected_session_id}\n")
    
    print(f"Session info updated: {session_info_file}")
    print("Next: Compare simple vs sophisticated corrector results")

if __name__ == "__main__":
    main()
