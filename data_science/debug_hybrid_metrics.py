#!/usr/bin/env python3
"""Debug script to check if hybrid metrics are in the returned dictionary"""

import sys
sys.path.insert(0, '/home/nitai/code/causalix.ai/data_science')

from modules import compare_session_graph_to_validation, CausalDiscovery
from backend.graph_db import Neo4jConnection

# Create a minimal discovery object
graph_db = Neo4jConnection()
discovery = CausalDiscovery(
    graph_db=graph_db,
    target_variable="Test",
    target_definition="Test definition",
    target_units="units"
)

# Set the session ID from the verification run
discovery.session_id = "b1e75d2e-379c-4187-a5b5-e2726bb86151"

# Call the function with the same parameters as the verification run
results = compare_session_graph_to_validation(
    discovery=discovery,
    session_ids="b1e75d2e-379c-4187-a5b5-e2726bb86151",
    validation_vars_json_path="/tmp/tmp477v5spx_vars_data.json",
    validation_edges_json_path="/tmp/tmp8hwkxeqy_edges_data.json",
    plot_session_graph=False,
    plot_validation_graph=False,
    only_cited_edges=False,
    only_consistent_edges=False,
    node_match_mode="llm_batch",
)

print("\n" + "="*60)
print("CHECKING HYBRID METRICS IN RETURNED DICTIONARY")
print("="*60)
print(f"hybrid_node_precision: {results.get('hybrid_node_precision')}")
print(f"hybrid_node_recall: {results.get('hybrid_node_recall')}")
print(f"hybrid_node_f1: {results.get('hybrid_node_f1')}")
print(f"avg_similarity_gen_to_val: {results.get('avg_similarity_gen_to_val')}")
print(f"avg_similarity_val_to_gen: {results.get('avg_similarity_val_to_gen')}")
print("\nAll keys in results:")
for key in sorted(results.keys()):
    value = results[key]
    if key.startswith(('hybrid', 'cosine', 'node_', 'avg_')):
        print(f"  {key}: {value}")