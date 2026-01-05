#!/usr/bin/env python3
"""Export run_3 CoT session from Neo4j (without CI metrics for speed)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules import export_edges_comparison_to_excel, CausalDiscovery

SESSION_ID = "c88b47fd-803f-4545-b297-dad76ac39ebf"

discovery = CausalDiscovery(
    target_variable="ED visits by older persons",
    temporal_scale="Days to Weeks",
    spatial_scale="Individual Level",
    yaml_path="parameter_tuning_experiments/alternative_prompts/prompts_citation_cot.yaml",
    dev_mode=False
)
discovery.set_session_id(SESSION_ID)

output_path = Path("../final_runs/RQ1a_gt_lit_citation/emergency_department/run_3/judged_Older_persons_ED_visits_cot_run_3.xlsx")
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Exporting session {SESSION_ID} to {output_path}")

excel_path = export_edges_comparison_to_excel(
    discovery=discovery,
    session_ids=SESSION_ID,
    validation_vars_json_path="parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_vars_data.json",
    validation_edges_json_path="parameter_tuning_experiments/ground_truth_clds_for_experiments/older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_edges_data.json",
    output_filename=str(output_path),
    embedding_enable=False,
    ci_compute_embeddings=False
)

print(f"✅ Excel created: {excel_path}")
