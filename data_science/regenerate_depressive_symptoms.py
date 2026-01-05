#!/usr/bin/env python3
"""
Regenerate Excel from Neo4j using a session ID for Depressive symptoms CLD.
Usage: python regenerate_depressive_symptoms.py <session_id>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules import export_edges_comparison_to_excel, CausalDiscovery

if len(sys.argv) < 2:
    print("Usage: python regenerate_depressive_symptoms.py <session_id>")
    sys.exit(1)

session_id = sys.argv[1]
print(f"🆔 Session ID: {session_id}")

# Use Depressive symptoms validation files
validation_vars = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_vars_data.json"
validation_edges = "/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/ground_truth_clds_for_experiments/Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_edges_data.json"
output_excel = f"/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/results/session_{session_id[:8]}_depressive_symptoms_REGENERATED.xlsx"

print(f"📂 Output: {output_excel}")

# Create discovery object
disco = CausalDiscovery(
    target_variable="dummy",
    temporal_scale="dummy",
    spatial_scale="dummy",
    yaml_path="/home/nitai/code/causalix.ai/data_science/parameter_tuning_experiments/prompts_Nitai_C.yaml",
    generator_config={"provider": "openai", "model": "gpt-4o"}
)

# Export
print("📊 Exporting...")
export_edges_comparison_to_excel(
    discovery=disco,
    session_ids=session_id,
    validation_vars_json_path=validation_vars,
    validation_edges_json_path=validation_edges,
    output_filename=output_excel
)

print(f"✅ Done! File: {output_excel}")
