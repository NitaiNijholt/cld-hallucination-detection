#!/usr/bin/env python3
"""
Export the 3 completed emergency_department baseline sessions from Neo4j to Excel.

These sessions are already judged and stored in Neo4j, we just need to export them.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from modules import export_edges_comparison_to_excel, CausalDiscovery
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Session IDs from the cloned sessions
SESSIONS = {
    "run_1": {
        "session_id": "94b7ac86-3664-416a-a05e-55e8dd5cf113",
        "progress": "1184/1190 (99%)"
    },
    "run_2": {
        "session_id": "00f13ae9-9b0e-4d54-94d4-74f8fc8107f3",
        "progress": "1186/1190 (99%)"
    },
    "run_3": {
        "session_id": "056065e3-d749-4576-89ba-32ef9e2dfd45",
        "progress": "1183/1190 (99%)"
    }
}

# Paths
EXCEL_FILE = "parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx"
VARS_JSON = "parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_vars_data.json"
EDGES_JSON = "parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_edges_data.json"
OUTPUT_DIR = Path("../final_runs/RQ1a_gt_lit_citation/emergency_department")

def export_session(run_name, session_id, progress):
    """Export a single session to Excel."""
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPORTING {run_name.upper()} BASELINE")
    logger.info(f"{'='*80}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Progress: {progress}")
    
    # Create discovery instance and set the session ID
    discovery = CausalDiscovery(
        target_variable="ED visits by older persons",  # From Excel
        temporal_scale="Days to Weeks",
        spatial_scale="Individual Level",
        yaml_path="parameter_tuning_experiments/alternative_prompts/prompts_citation_baseline.yaml",
        dev_mode=False
    )
    
    # Set the session ID to the cloned one
    discovery.set_session_id(session_id)
    
    # Create output directory
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    output_file = f"judged_Older_persons_ED_visits_baseline_{run_name}.xlsx"
    output_path = run_dir / output_file
    
    logger.info(f"Output: {output_path}")
    
    try:
        # Export to Excel with CI metrics enabled
        excel_path = export_edges_comparison_to_excel(
            discovery=discovery,
            session_ids=session_id,
            validation_vars_json_path=VARS_JSON,
            validation_edges_json_path=EDGES_JSON,
            output_filename=str(output_path),
            lm_stats=None,
            experiment_params=None,
            node_comparison_data=None,
            node_match_mode="llm_batch",
            embedding_enable=True,
            ci_fetch_reuse=True,
            ci_chunk_chars_override=None,
            ci_overlap_ratio=0.2,
            ci_compute_embeddings=True,
            ci_parallel=True,
            ci_max_workers=10,
            embedding_provider="local",
            embedding_device="cuda",
        )
        
        logger.info(f"✅ SUCCESS: {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Export all 3 sessions."""
    print("\n" + "="*80)
    print("EMERGENCY DEPARTMENT BASELINE SESSIONS - EXCEL EXPORT")
    print("="*80)
    print("\nExporting 3 completed baseline sessions from Neo4j to Excel...")
    print()
    
    results = {}
    for run_name, info in SESSIONS.items():
        result = export_session(run_name, info["session_id"], info["progress"])
        results[run_name] = result
    
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    for run_name, path in results.items():
        status = "✅" if path else "❌"
        print(f"{status} {run_name}: {path if path else 'FAILED'}")
    print("="*80)


if __name__ == "__main__":
    main()
