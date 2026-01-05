#!/usr/bin/env python3
"""
RQ1 Multi-CLD Experiment Runner

Runs the complete RQ1b experiment pipeline (Phase 0, 1, 2a, 2b) for multiple CLDs.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_session_cloner import SessionCloner
from modules import run_discovery_experiment


# Define CLDs to process
CLDS = [
    {
        "name": "Social_norms",
        "excel_file": "parameter_tuning_experiments/ground_truth_clds_for_experiments/Social_norms_and_obesity_prevalence.xlsx",
        "override_target": "Obesity Prevalence"
    },
    {
        "name": "Depressive_symptoms",
        "excel_file": "parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults.xlsx",
        "override_target": None  # Will be determined from context data
    }
]


# Experiment configuration
GENERATOR_CONFIG = {"provider": "openai", "model": "gpt-4.1", "temperature": 0.7}
CORRUPTOR_CONFIG = {"provider": "openai", "model": "gpt-4.1", "temperature": 0.7}
JUDGE_CONFIG = {"provider": "openai", "model": "gpt-4.1", "temperature": 0.3}
CORRECTOR_MODELS = ["gpt-4.1"]

CORRUPTION_RATE = 0.3
YAML_PATH = "parameter_tuning_experiments/alternative_prompts/prompts_Nitai_C.yaml"


def run_phase_0_generate(cld_config, output_base_dir):
    """Phase 0: Generate base CLD"""
    print("\n" + "="*80)
    print(f"PHASE 0: GENERATING BASE CLD - {cld_config['name']}")
    print("="*80)
    
    excel_file = cld_config["excel_file"]
    override_target = cld_config.get("override_target")
    
    result = run_discovery_experiment(
        excel_path=excel_file,
        yaml_path=YAML_PATH,
        dev_mode=False,
        generator_config=GENERATOR_CONFIG,
        corruptor_config=CORRUPTOR_CONFIG,
        judge_config=JUDGE_CONFIG,
        corruption_rate=0.0,
        judge_edges=False,
        generation_parallel=True,
        generation_max_workers=10,
        overide_target_variable=override_target,
        citation_search_provider="brave",
        output_json_prefix=cld_config['name'],
        output_dir=str(output_base_dir),
        experiment_description=f"Base_{cld_config['name']}",
        embedding_enable=False,
        ci_compute_embeddings=False,
        node_comparison_enable=False,
    )
    
    session_id = result.get('session_id')
    print(f"\n✅ Phase 0 complete: Base session ID = {session_id}")
    return session_id


def run_phase_1_corrupt(base_session_id, cld_config, output_dir):
    """Phase 1: Clone and corrupt"""
    print("\n" + "="*80)
    print(f"PHASE 1: CORRUPTING - {cld_config['name']}")
    print("="*80)
    
    # Clone session
    cloner = SessionCloner()
    corrupted_session_id = cloner.clone_session(
        original_session_id=base_session_id,
        include_judge_data=False
    )
    print(f"✅ Cloned: {base_session_id[:8]}... → {corrupted_session_id[:8]}...")
    
    # Apply corruption
    result = run_discovery_experiment(
        retrieved_session_id=corrupted_session_id,
        excel_path=cld_config["excel_file"],
        yaml_path=YAML_PATH,
        generator_config=GENERATOR_CONFIG,
        corruption_rate=CORRUPTION_RATE,
        corruptor_config=CORRUPTOR_CONFIG,
        judge_edges=False,
        judge_config=JUDGE_CONFIG,
        result_excel_path=str(output_dir / f"corrupted_{corrupted_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        embedding_enable=True,
        citation_search_provider="brave"
    )
    
    print(f"✅ Phase 1 complete: Corrupted session ID = {corrupted_session_id}")
    return corrupted_session_id


def run_phase_2a_judge_no_correction(corrupted_session_id, cld_config, output_dir):
    """Phase 2a: Judge without correction"""
    print("\n" + "="*80)
    print(f"PHASE 2a: JUDGING WITHOUT CORRECTION - {cld_config['name']}")
    print("="*80)
    
    # Clone for judging without correction
    cloner = SessionCloner()
    judged_session_id = cloner.clone_session(
        original_session_id=corrupted_session_id,
        include_judge_data=False
    )
    print(f"✅ Cloned: {corrupted_session_id[:8]}... → {judged_session_id[:8]}...")
    
    # Judge without correction
    result = run_discovery_experiment(
        retrieved_session_id=judged_session_id,
        excel_path=cld_config["excel_file"],
        yaml_path=YAML_PATH,
        generator_config=GENERATOR_CONFIG,
        corruptor_config=CORRUPTOR_CONFIG,
        judge_config=JUDGE_CONFIG,
        judge_models=CORRECTOR_MODELS,
        num_judges=1,
        judge_approach="correctness",
        judge_edges=True,
        judge_parallel=True,
        judge_max_workers=10,
        run_correction=False,
        result_excel_path=str(output_dir / f"judged_no_correction_{judged_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        embedding_enable=True,
        citation_search_provider="brave"
    )
    
    print(f"✅ Phase 2a complete: Judged session ID = {judged_session_id}")
    return judged_session_id


def run_phase_2b_correct(corrupted_session_id, cld_config, output_dir):
    """Phase 2b: Judge with correction"""
    print("\n" + "="*80)
    print(f"PHASE 2b: JUDGING WITH CORRECTION - {cld_config['name']}")
    print("="*80)
    
    # Clone for correction
    cloner = SessionCloner()
    corrected_session_id = cloner.clone_session(
        original_session_id=corrupted_session_id,
        include_judge_data=False
    )
    print(f"✅ Cloned: {corrupted_session_id[:8]}... → {corrected_session_id[:8]}...")
    
    # Judge with correction
    result = run_discovery_experiment(
        retrieved_session_id=corrected_session_id,
        excel_path=cld_config["excel_file"],
        yaml_path=YAML_PATH,
        generator_config=GENERATOR_CONFIG,
        corruptor_config=CORRUPTOR_CONFIG,
        judge_config=JUDGE_CONFIG,
        judge_models=CORRECTOR_MODELS,
        num_judges=1,
        judge_approach="correctness",
        judge_edges=True,
        judge_parallel=True,
        judge_max_workers=10,
        run_correction=True,
        corrector_models=CORRECTOR_MODELS,
        rejudge_approach="correctness",
        rejudge_parallel=True,
        rejudge_max_workers=10,
        result_excel_path=str(output_dir / f"corrected_{corrected_session_id[:8]}.xlsx"),
        output_dir=str(output_dir),
        embedding_enable=True,
        citation_search_provider="brave"
    )
    
    print(f"✅ Phase 2b complete: Corrected session ID = {corrected_session_id}")
    return corrected_session_id


def process_cld(cld_config, base_output_dir):
    """Process a single CLD through all phases"""
    print("\n" + "="*80)
    print(f"PROCESSING CLD: {cld_config['name']}")
    print("="*80)
    
    # Create CLD-specific output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cld_output_dir = base_output_dir / f"{cld_config['name']}_{timestamp}"
    cld_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "cld_name": cld_config['name'],
        "timestamp": timestamp,
        "output_dir": str(cld_output_dir)
    }
    
    try:
        # Phase 0: Generate base
        base_session_id = run_phase_0_generate(cld_config, cld_output_dir)
        results['base_session_id'] = base_session_id
        
        # Phase 1: Corrupt
        corrupted_session_id = run_phase_1_corrupt(base_session_id, cld_config, cld_output_dir)
        results['corrupted_session_id'] = corrupted_session_id
        
        # Phase 2a: Judge without correction
        judged_no_corr_id = run_phase_2a_judge_no_correction(corrupted_session_id, cld_config, cld_output_dir)
        results['judged_no_correction_session_id'] = judged_no_corr_id
        
        # Phase 2b: Judge with correction
        corrected_session_id = run_phase_2b_correct(corrupted_session_id, cld_config, cld_output_dir)
        results['corrected_session_id'] = corrected_session_id
        
        results['status'] = 'SUCCESS'
        
    except Exception as e:
        print(f"\n❌ ERROR processing {cld_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'FAILED'
        results['error'] = str(e)
    
    # Save results
    results_file = cld_output_dir / "session_info.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main execution"""
    print("="*80)
    print("RQ1b MULTI-CLD EXPERIMENT RUNNER")
    print("="*80)
    print(f"\nProcessing {len(CLDS)} CLDs:")
    for cld in CLDS:
        print(f"  - {cld['name']}")
    print()
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"parameter_tuning_experiments/results/rq1_multi_cld_{timestamp}")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {base_output_dir}\n")
    
    # Process each CLD
    all_results = []
    for i, cld_config in enumerate(CLDS, 1):
        print(f"\n{'='*80}")
        print(f"CLD {i}/{len(CLDS)}: {cld_config['name']}")
        print(f"{'='*80}")
        
        results = process_cld(cld_config, base_output_dir)
        all_results.append(results)
    
    # Save summary
    summary_file = base_output_dir / "experiment_summary.json"
    summary = {
        "timestamp": timestamp,
        "clds_processed": len(CLDS),
        "corruption_rate": CORRUPTION_RATE,
        "generator_model": GENERATOR_CONFIG['model'],
        "corrector_model": CORRECTOR_MODELS[0],
        "results": all_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {base_output_dir}")
    print(f"Summary: {summary_file}")
    
    success_count = sum(1 for r in all_results if r['status'] == 'SUCCESS')
    print(f"\nSuccessful: {success_count}/{len(CLDS)}")
    
    for r in all_results:
        status_icon = "✅" if r['status'] == 'SUCCESS' else "❌"
        print(f"  {status_icon} {r['cld_name']}: {r['status']}")


if __name__ == "__main__":
    main()
