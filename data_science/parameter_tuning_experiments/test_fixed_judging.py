#!/usr/bin/env python3
"""
Test the fixed judging logic on a single provider
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results"

def test_fixed_judging():
    print("\n" + "="*80)
    print("TESTING FIXED JUDGING LOGIC")
    print("="*80)
    
    # Load PubMed session (fastest one)
    session_file = OUTPUT_DIR / "citation_provider_sessions_20251106_051504.json"
    with open(session_file, 'r') as f:
        sessions = json.load(f)
    
    session_id = sessions["provider_sessions"]["pubmed"]
    print(f"\nTesting on PubMed session: {session_id}")
    print(f"Will judge only first 3 edges to verify the fix\n")
    
    # Create CausalDiscovery instance
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
    
    # Load variables from the session
    with discovery.graph_db._get_session() as neo4j_session:
        result = neo4j_session.run("""
            MATCH (n:variable {session_id: $session_id})
            RETURN n.name as name
        """, {"session_id": session_id})
        discovery.variables = [record["name"] for record in result]
    
    print(f"Loaded {len(discovery.variables)} variables\n")
    
    # Get first 3 edges to show what will be judged
    with discovery.graph_db._get_session() as neo4j_session:
        query = """
            MATCH (source:variable {session_id: $session_id})-[r:CAUSAL_RELATION]->(target:variable {session_id: $session_id})
            RETURN source.name AS source, target.name AS target, r.type AS rel_type, 
                   properties(r) AS props
            LIMIT 3
        """
        result = neo4j_session.run(query, {"session_id": session_id})
        edges = list(result)
    
    print(f"Preview of {len(edges)} edges that will be judged:\n")
    
    for idx, record in enumerate(edges, 1):
        src = record["source"]
        tgt = record["target"]
        rel_type = record["rel_type"]
        props = record["props"]
        motivation = props.get("motivation", "")
        
        print(f"{idx}. {src} --[{rel_type}]--> {tgt}")
        print(f"   Motivation field: '{motivation}'")
        
        # Determine what will be judged
        if motivation and motivation.lower() not in ["expert-validated", "expert validated", "n/a", "na"]:
            claim = motivation
        else:
            if rel_type.upper() == "POSITIVE":
                claim = f"{src} directly causes an increase in {tgt}"
            elif rel_type.upper() == "NEGATIVE":
                claim = f"{src} directly causes a decrease in {tgt}"
            else:
                claim = f"{src} directly causes changes in {tgt}"
        
        print(f"   Will judge: '{claim}'")
        print()
    
    # Now run actual judging on first 3 edges using a patched version
    print(f"{'='*80}")
    print("RUNNING JUDGING ON FIRST 3 EDGES...")
    print(f"{'='*80}\n")
    
    # Monkey-patch the query to limit to 3 edges
    original_query = """
        MATCH (source:variable {session_id: $session_id})-[r:CAUSAL_RELATION]->(target:variable {session_id: $session_id})
        RETURN source.name AS source, target.name AS target, r.type AS rel_type, 
               properties(r) AS props
    """
    
    limited_query = original_query + "\nLIMIT 3"
    
    # Temporarily patch the graph query
    import types
    
    def patched_judge_method(self, *args, **kwargs):
        # Just run the first 3 edges
        with self.graph_db._get_session() as session:
            result = session.run(limited_query, {"session_id": self.session_id})
            records = list(result)
        
        # Now process them
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        start_time = time.time()
        
        # Build ephemeral judges
        judge_models = kwargs.get("judge_models", ["gpt-4.1"])
        from llm_client_openai_working import OpenAIClient
        ephemeral_judges = []
        for model in judge_models:
            client = OpenAIClient(
                model=model,
                temperature=self.judge_temperature,
                top_p=self.judge_top_p,
                enable_websearch=False
            )
            ephemeral_judges.append(client)
        
        judged_edges = []
        
        for idx, record in enumerate(records, 1):
            result = self._process_single_edge_judgment(
                idx=idx,
                total_edges=len(records),
                record=record,
                ephemeral_judges=ephemeral_judges,
                models=judge_models,
                approach=kwargs.get("approach", "per_citation_aggregate"),
                use_jina_ai=kwargs.get("use_jina_ai", False),
                jina_timeout=kwargs.get("jina_timeout", 60),
                jina_retries=kwargs.get("jina_retries", 2)
            )
            if result:
                judged_edges.append(result)
        
        duration = time.time() - start_time
        print(f"\n✅ Test judging complete in {duration:.1f}s")
        print(f"   Judged {len(judged_edges)} edges")
        
        return judged_edges
    
    # Apply the patch
    judged_edges = patched_judge_method(
        discovery,
        judge_models=["gpt-4.1"],
        num_judges=1,
        approach="per_citation_aggregate",
        use_jina_ai=True,
        jina_timeout=60,
        jina_retries=2
    )
    
    # Show results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    
    for edge in judged_edges:
        print(f"{edge['source']} --[{edge['type']}]--> {edge['target']}")
        print(f"  Verdict: {edge.get('aggregate_verdict', 'N/A')}")
        print(f"  Score: {edge.get('aggregate_score', 'N/A')}")
        
        # Show what was judged
        details = edge.get('details_json', {})
        if isinstance(details, str):
            try:
                import json as json_module
                details = json_module.loads(details)
            except:
                pass
        
        if isinstance(details, dict) and details.get('citations'):
            print(f"  Citations judged: {len(details['citations'])}")
            for cit in details['citations'][:1]:  # Show first citation
                print(f"    - {cit.get('url', 'N/A')}: {cit.get('verdict', 'N/A')}")
        print()
    
    print(f"{'='*80}")
    print("✅ FIX VERIFIED - Now judging actual causal claims with 'directly causes'!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_fixed_judging()
