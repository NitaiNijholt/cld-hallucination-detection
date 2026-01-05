#!/usr/bin/env python3
"""
RQ1 Phase 0: Generate Single Base CLD

Generates a single base session (no corruption, no judging) with parallel edge discovery.

Usage:
    python run_rq1_phase0_generate_single.py <excel_file>
    
Example:
    python run_rq1_phase0_generate_single.py parameter_tuning_experiments/ground_truth_clds_for_experiments/belgian_sugar_transportability_perishability_1.xlsx
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import run_discovery_experiment

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable not set!")

print("OpenAI API key is set:", os.getenv('OPENAI_API_KEY')[:10] + "..." + os.getenv('OPENAI_API_KEY')[-4:])


def generate_base_cld(excel_file: str):
    """
    Generate a base CLD with parallel edge discovery.
    
    Args:
        excel_file: Path to Excel file
    
    Returns:
        dict with session_id and excel_path
    """
    excel_path = Path(excel_file)
    cld_name = excel_path.stem
    
    print(f"\n{'='*80}")
    print(f"PHASE 0: GENERATING BASE CLD - {cld_name}")
    print(f"{'='*80}")
    print(f"Excel file: {excel_file}")
    print()
    
    # Generator config: GPT-4.1
    generator_config = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7
    }
    
    # Dummy configs (not used for base generation)
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
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"rq1_phase0_generation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"🚀 Starting PARALLEL edge discovery (10 workers)...")
    print()
    
    # Determine if we need to override target variable for Social Norms CLD
    override_target = None
    if "Social_norms" in excel_file or "social_norms" in excel_file.lower():
        override_target = "Obesity Prevalence"
        print(f"⚠️  Overriding target variable to: {override_target}")
        print()
    
    # Run base CLD generation with PARALLEL edge discovery
    result = run_discovery_experiment(
        excel_path=str(excel_file),
        yaml_path=str(yaml_path),
        dev_mode=False,
        generator_config=generator_config,
        corruptor_config=dummy_corruptor_config,
        judge_config=dummy_judge_config,
        corruption_rate=0.0,  # No corruption
        judge_edges=False,  # No judging yet
        generation_parallel=True,  # 🚀 PARALLEL edge discovery
        generation_max_workers=10,  # 10 concurrent workers
        overide_target_variable=override_target,  # Override target if needed
        citation_search_provider="brave",  # Use Brave for citations (matches original RQ1b)
        output_json_prefix=cld_name,
        output_dir=str(output_dir),
        experiment_description=f"Base_{cld_name}",
        embedding_enable=False,  # Skip CI metrics for base generation
        ci_compute_embeddings=False,  # Skip embeddings
        node_comparison_enable=False,  # Skip comparison for base generation
    )
    
    session_id = result.get('session_id')
    result_excel = result.get('result_excel_path') or result.get('excel_filename')
    
    print(f"\n{'='*80}")
    print("✅ PHASE 0 COMPLETE - BASE CLD GENERATED!")
    print(f"{'='*80}")
    print(f"Session ID: {session_id}")
    print(f"Excel: {result_excel}")
    print()
    
    # Save session ID to a file for easy access
    session_file = Path(__file__).parent / f"{cld_name}_session_id.txt"
    with open(session_file, 'w') as f:
        f.write(f"{session_id}\n")
        f.write(f"{excel_file}\n")
    
    print(f"✓ Session ID saved to: {session_file}")
    print()
    print(f"{'='*80}")
    print("NEXT: Run Phase 2 (judging) with:")
    print(f"{'='*80}")
    print(f"python parameter_tuning_experiments/run_rq1_judge_base_session.py {session_id} {excel_file}")
    print()
    
    return {
        'cld_name': cld_name,
        'session_id': session_id,
        'excel_path': result_excel,
        'original_excel': str(excel_file),
        'timestamp': timestamp
    }


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python run_rq1_phase0_generate_single.py <excel_file>")
        print("\nExample:")
        print("  python run_rq1_phase0_generate_single.py ground_truth_clds_for_experiments/belgian_sugar_transportability_perishability_1.xlsx")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    
    if not Path(excel_file).exists():
        print(f"Error: Excel file not found: {excel_file}")
        sys.exit(1)
    
    try:
        result = generate_base_cld(excel_file)
        print("✅ Success!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
