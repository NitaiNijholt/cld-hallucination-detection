#!/usr/bin/env python3
"""
RQ1b Prompt Ablation Study: Test all corrector prompt variants

Tests 3 prompt variants on the same corrupted CLD:
1. Original (weak) - Current prompt with minimal guidance  
2. Improved - Explicit tool selection guidance
3. Aggressive Fix - Strongly discourages deletion

Usage:
    python run_rq1_prompt_ablation.py <corrupted_session_id> <excel_file> <output_base_dir>
    
Example:
    python run_rq1_prompt_ablation.py 6bd82e7d-db53-43f4-8a16-0f4f4f037dcc \\
        parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx \\
        parameter_tuning_experiments/results/rq1_prompt_ablation_20251113
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import run_discovery_experiment

PROMPT_VARIANTS = [
    ("original", "Original (weak instructions)"),
    ("improved", "Improved (explicit tool guidance)"),
    ("aggressive_fix", "Aggressive Fix (deletion discouraged)")
]

def main():
    if len(sys.argv) < 4:
        print("Usage: python run_rq1_prompt_ablation.py <corrupted_session_id> <excel_file> <output_base_dir>")
        sys.exit(1)
    
    corrupted_session_id = sys.argv[1]
    excel_file = sys.argv[2]
    output_base_dir = Path(sys.argv[3])
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RQ1b: PROMPT ABLATION STUDY")
    print("="*80)
    print(f"Corrupted session: {corrupted_session_id}")
    print(f"Excel file: {excel_file}")
    print(f"Output directory: {output_base_dir}")
    print(f"\nTesting {len(PROMPT_VARIANTS)} prompt variants:")
    for variant, desc in PROMPT_VARIANTS:
        print(f"  - {variant}: {desc}")
    print()
    
    results = {}
    cloner = SessionCloner()
    
    for variant, description in PROMPT_VARIANTS:
        print("\n" + "="*80)
        print(f"TESTING: {variant} - {description}")
        print("="*80)
        
        # Clone corrupted session for this variant
        print(f"Cloning corrupted session for '{variant}' variant...")
        corrected_session_id = cloner.clone_session(
            original_session_id=corrupted_session_id,
            include_judge_data=False
        )
        print(f"✅ Cloned: {corrupted_session_id[:8]}... → {corrected_session_id[:8]}...")
        print()
        
        # Create output directory for this variant
        variant_dir = output_base_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        # Run correction with this prompt variant
        print(f"Running correction with '{variant}' prompt...")
        print(f"Judging with gpt-4o + Corrector with gpt-4o (prompt={variant})")
        print()
        
        try:
            result = run_discovery_experiment(
                retrieved_session_id=corrected_session_id,
                excel_path=excel_file,
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
                use_simple_corrector=False,  # Use sophisticated corrector
                corrector_prompt_variant=variant,  # KEY: Test this prompt variant
                result_excel_path=str(variant_dir / f"corrected_{variant}_{corrected_session_id[:8]}.xlsx"),
                output_dir=str(variant_dir),
                embedding_enable=True,
                ci_compute_embeddings=False,
                citation_search_provider="brave"
            )
            
            results[variant] = {
                "session_id": corrected_session_id,
                "excel_path": result.get('result_excel_path'),
                "status": "success"
            }
            
            print(f"\n✅ {variant} COMPLETE")
            print(f"   Session ID: {corrected_session_id}")
            print(f"   Result: {result.get('result_excel_path')}")
            
        except Exception as e:
            print(f"\n❌ {variant} FAILED: {e}")
            results[variant] = {
                "session_id": corrected_session_id,
                "status": "failed",
                "error": str(e)
            }
    
    # Write summary
    summary_file = output_base_dir / "ablation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("RQ1b PROMPT ABLATION STUDY SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Corrupted session: {corrupted_session_id}\n")
        f.write(f"Excel file: {excel_file}\n\n")
        
        for variant, desc in PROMPT_VARIANTS:
            result = results.get(variant, {})
            f.write(f"\n{variant.upper()} ({desc}):\n")
            f.write(f"  Status: {result.get('status', 'not run')}\n")
            if result.get('status') == 'success':
                f.write(f"  Session ID: {result['session_id']}\n")
                f.write(f"  Excel: {result['excel_path']}\n")
            elif result.get('status') == 'failed':
                f.write(f"  Error: {result['error']}\n")
    
    print("\n" + "="*80)
    print("PROMPT ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Summary written to: {summary_file}")
    print()
    print("Next steps:")
    print("1. Run analysis script to compare tool usage across prompts")
    print("2. python compare_prompt_variants.py <output_base_dir>")
    print("="*80)

if __name__ == "__main__":
    main()
