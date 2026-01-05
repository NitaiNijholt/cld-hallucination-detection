#!/usr/bin/env python3
"""
Export emergency_department run_3 cot session from Neo4j with local CI metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import CausalDiscovery, export_edges_comparison_to_excel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    session_id = "77928a30-4375-4a1c-8b17-373968a4b6f4"
    output_file = Path("../final_runs/RQ1a_gt_lit_citation/emergency_department/run_3/judged_older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_cot_20251125_032235_20251125_040838.xlsx")
    
    logger.info("=" * 80)
    logger.info("EXPORTING RUN 3 COT WITH LOCAL CI METRICS")
    logger.info("=" * 80)
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Output: {output_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
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
    excel_path = export_edges_comparison_to_excel(
        discovery=discovery,
        session_ids=session_id,
        validation_vars_json_path=None,
        validation_edges_json_path=None,
        output_filename=str(output_file),
        
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
    
    logger.info(f"✅ Export completed successfully!")
    logger.info(f"Output: {excel_path}")

if __name__ == "__main__":
    main()


