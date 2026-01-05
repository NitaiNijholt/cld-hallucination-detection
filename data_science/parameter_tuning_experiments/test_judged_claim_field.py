#!/usr/bin/env python3
"""
Quick test to verify judged_claim field is populated
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery
import logging

logging.basicConfig(level=logging.WARNING)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results"

# Load PubMed session
session_file = OUTPUT_DIR / "citation_provider_sessions_20251106_051504.json"
with open(session_file, 'r') as f:
    sessions = json.load(f)

session_id = sessions["provider_sessions"]["pubmed"]

# Create discovery instance
discovery = CausalDiscovery(
    target_variable="Alzheimer's disease risk",
    temporal_scale="Chronic, aging-related",
    spatial_scale="Older adults",
    yaml_path=str(BASE_DIR / "prompts_Nitai_C.yaml"),
    generator_config={"provider": "openai", "model": "gpt-4o"},
    judge_config={"provider": "openai", "model": "gpt-4.1"},
    judge_temperature=0.3
)

discovery.session_id = session_id

# Load variables
with discovery.graph_db._get_session() as neo4j_session:
    result = neo4j_session.run("""
        MATCH (n:variable {session_id: $session_id})
        RETURN n.name as name
    """, {"session_id": session_id})
    discovery.variables = [record["name"] for record in result]

# Judge only first edge
with discovery.graph_db._get_session() as neo4j_session:
    query = """
        MATCH (source:variable {session_id: $session_id})-[r:CAUSAL_RELATION]->(target:variable {session_id: $session_id})
        RETURN source.name AS source, target.name AS target, r.type AS rel_type, 
               properties(r) AS props
        LIMIT 1
    """
    result = neo4j_session.run(query, {"session_id": session_id})
    records = list(result)

# Build judge
from llm_client_openai_working import OpenAIClient
judge = OpenAIClient(model="gpt-4.1", temperature=0.3, enable_websearch=False)

# Process
result = discovery._process_single_edge_judgment(
    idx=1,
    total_edges=1,
    record=records[0],
    ephemeral_judges=[judge],
    models=["gpt-4.1"],
    approach="per_citation_aggregate",
    use_jina_ai=True,
    jina_timeout=60,
    jina_retries=2
)

print("\n" + "="*80)
print("TEST RESULT")
print("="*80)
print(f"\nEdge: {result['source']} --[{result['type']}]--> {result['target']}")
print(f"Original motivation: {result.get('motivation', 'N/A')}")
print(f"Judged claim: {result.get('judged_claim', 'NOT FOUND!')}")
print(f"Verdict: {result.get('aggregate_verdict', 'N/A')}")
print(f"Score: {result.get('aggregate_score', 'N/A')}")

if 'judged_claim' in result:
    print("\n✅ SUCCESS - judged_claim field is now populated!")
else:
    print("\n❌ FAIL - judged_claim field is missing!")

print("="*80 + "\n")



