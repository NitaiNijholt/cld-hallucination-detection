#!/usr/bin/env python3
"""
Export emergency_department run_3 CoT and Mechanistic sessions from Neo4j
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import CausalDiscovery, export_edges_comparison_to_excel

# Session ID from the log (CoT session that was judged)
RUN_3_COT_SESSION = "c88b47fd-803f-4545-b297-dad76ac39ebf"

# We need to run mechanistic judging still - it never started
# Let's first just export the CoT session

def export_single_session(session_id: str, run_name: str, prompt_type: str):
    """Export a single session from Neo4j to Excel."""
    
    print(f"\n{'='*80}")
    print(f"EXPORTING: {run_name} - {prompt_type}")
    print(f"{'='*80}")
    print(f"Session ID: {session_id}")
    
    # Initialize discovery with the correct prompt file
    prompt_file = f"parameter_tuning_experiments/alternative_prompts/prompts_citation_{prompt_type}.yaml"
    
    discovery = CausalDiscovery(
        target_variable="Acute care demand leading to an ED visit",
        temporal_scale="Various (from hours to years)",
        spatial_scale="Individual to organizational level",
        yaml_path=prompt_file,
        excel_file="parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx",
        judge_approach="per_citation_aggregate",
        judge_enable_web_search=True
    )
    
    # Set the session ID to retrieve from Neo4j
    discovery.set_session_id(session_id)
    
    # Define output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "final_runs" / "RQ1a_gt_lit_citation" / "emergency_department" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"judged_Older_persons_ED_visits_{prompt_type}_{run_name}_{timestamp}.xlsx"
    
    print(f"\nOutput: {output_file}")
    print(f"\nExporting with CI metrics enabled...")
    
    # Export to Excel with CI metrics
    excel_path = export_edges_comparison_to_excel(
        discovery=discovery,
        output_path=str(output_file),
        retrieved_session_id=session_id,
        embedding_enable=True,
        ci_compute_embeddings=True,
        ci_parallel=True,
        ci_max_workers=10,
        embedding_provider="local",
        embedding_device="cuda",
    )
    
    print(f"\n✅ SUCCESS: {excel_path}")
    return excel_path

if __name__ == "__main__":
    print("="*80)
    print("EMERGENCY DEPARTMENT RUN_3 EXPORT")
    print("="*80)
    
    # Export CoT session
    try:
        cot_file = export_single_session(RUN_3_COT_SESSION, "run_3", "cot")
        print(f"\n✅ CoT export complete!")
    except Exception as e:
        print(f"\n❌ CoT export failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("MECHANISTIC PROMPT STILL NEEDS TO BE JUDGED")
    print("="*80)
    print("The mechanistic prompt was never run because the process got stuck.")
    print("You need to run the multirunner again, or manually judge the mechanistic prompt.")
    print("="*80)
