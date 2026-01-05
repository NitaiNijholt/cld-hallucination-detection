#!/usr/bin/env python3
"""
Export judged sessions to Excel with LOCAL CI metrics computation.

This avoids OpenAI API rate limits by using local GPU embeddings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import CausalDiscovery, export_edges_comparison_to_excel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session IDs from the judging runs
SESSIONS = {
    "run_2_mechanistic": {
        "session_id": "071b547b-5882-4dd9-a99f-d02ef3319eb5",
        "output": "/home/nitai/code/causalix.ai/final_runs/RQ1a_gt_lit_citation/emergency_department/run_2/judged_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_mechanistic_20251125_032605_20251125_040815.xlsx",
        "excel_file": "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx"
    },
    "run_3_cot": {
        "session_id": "77928a30-4375-4a1c-8b17-373968a4b6f4",
        "output": "/home/nitai/code/causalix.ai/final_runs/RQ1a_gt_lit_citation/emergency_department/run_3/judged_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_cot_20251125_032235_20251125_040838.xlsx",
        "excel_file": "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet.xlsx"
    }
}

def export_session(session_name: str, session_info: dict):
    """Export a session to Excel with local CI metrics."""
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPORTING: {session_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Session ID: {session_info['session_id']}")
    logger.info(f"Output: {session_info['output']}")
    
    # Create minimal discovery instance for export
    yaml_path = Path(__file__).parent / "../backend/configs/prompts.yaml"
    discovery = CausalDiscovery(
        target_variable="dummy",
        temporal_scale="dummy",
        spatial_scale="dummy",
        yaml_path=str(yaml_path)
    )
    
    # Export to Excel with LOCAL embeddings and CUDA
    logger.info("Exporting to Excel with LOCAL CI metrics (GPU)...")
    export_edges_comparison_to_excel(
        discovery=discovery,
        session_ids=session_info['session_id'],
        validation_vars_json_path=None,
        validation_edges_json_path=None,
        output_filename=session_info['output'],
        
        # ⭐ LOCAL EMBEDDINGS - NO OPENAI API ⭐
        embedding_enable=True,
        ci_compute_embeddings=True,
        ci_parallel=True,
        ci_max_workers=10,
        embedding_provider="local",  # Use local GPU embeddings
        embedding_device="cuda",     # Use GPU
        
        ci_fetch_reuse=True,
        ci_chunk_chars_override=None,
        ci_overlap_ratio=0.2,
        
        node_comparison_data=None
    )
    
    logger.info(f"✅ {session_name} exported successfully!")
    return session_info['output']

def main():
    logger.info("="*80)
    logger.info("EXPORT SESSIONS WITH LOCAL CI METRICS")
    logger.info("="*80)
    logger.info("Using LOCAL GPU embeddings to avoid OpenAI rate limits")
    logger.info("")
    
    for session_name, session_info in SESSIONS.items():
        try:
            output_file = export_session(session_name, session_info)
            logger.info(f"✓ Created: {output_file}\n")
        except Exception as e:
            logger.error(f"❌ Failed to export {session_name}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL EXPORTS COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()

