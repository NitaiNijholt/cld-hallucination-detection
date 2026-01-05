#!/usr/bin/env python3
"""
Test judging on a single provider to verify the fix
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results"

def test_pubmed():
    print("\n" + "="*80)
    print("TESTING FIXED JUDGING - PubMed Provider")
    print("="*80)
    
    # Load PubMed session
    session_file = OUTPUT_DIR / "citation_provider_sessions_20251106_051504.json"
    with open(session_file, 'r') as f:
        sessions = json.load(f)
    
    session_id = sessions["provider_sessions"]["pubmed"]
    print(f"\nPubMed Session: {session_id}")
    
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
    
    # Load variables
    with discovery.graph_db._get_session() as neo4j_session:
        result = neo4j_session.run("""
            MATCH (n:variable {session_id: $session_id})
            RETURN n.name as name
        """, {"session_id": session_id})
        discovery.variables = [record["name"] for record in result]
    
    print(f"Loaded {len(discovery.variables)} variables")
    
    # Count edges
    with discovery.graph_db._get_session() as neo4j_session:
        result = neo4j_session.run("""
            MATCH (source:variable {session_id: $session_id})-[r:CAUSAL_RELATION]->(target:variable {session_id: $session_id})
            RETURN count(r) as count
        """, {"session_id": session_id})
        edge_count = result.single()["count"]
    
    print(f"Total edges: {edge_count}")
    print(f"\nRunning judging (this will take ~1-2 minutes)...\n")
    
    import time
    start_time = time.time()
    
    # Run judging
    judged_edges = discovery.judge_all_edges_with_citations_serial(
        judge_models=["gpt-4.1"],
        num_judges=1,
        approach="per_citation_aggregate",
        judge_parallel=True,
        judge_max_workers=10,
        use_jina_ai=True,
        jina_timeout=60,
        jina_retries=2
    )
    
    duration = time.time() - start_time
    
    # Calculate summary
    aggregate_scores = []
    verdicts = []
    
    for edge in judged_edges:
        score = edge.get("aggregate_score")
        if score is not None:
            aggregate_scores.append(score)
            verdicts.append(edge.get("aggregate_verdict", "unknown"))
    
    avg_score = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0
    
    from collections import Counter
    verdict_counts = Counter(verdicts)
    
    print(f"\n{'='*80}")
    print("TEST RESULTS")
    print(f"{'='*80}\n")
    print(f"Provider: PubMed")
    print(f"Session: {session_id}")
    print(f"Edges judged: {len(judged_edges)}")
    print(f"Edges with citations: {len(aggregate_scores)}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Duration: {duration:.1f}s")
    print(f"\nVerdict breakdown:")
    for verdict, count in verdict_counts.most_common():
        pct = count / len(verdicts) * 100 if verdicts else 0
        print(f"  - {verdict}: {count} ({pct:.1f}%)")
    
    # Show a sample edge to verify the claim is correct
    print(f"\n{'='*80}")
    print("SAMPLE EDGE (to verify fix):")
    print(f"{'='*80}\n")
    
    if judged_edges:
        sample = judged_edges[0]
        print(f"Edge: {sample['source']} --[{sample['type']}]--> {sample['target']}")
        print(f"Verdict: {sample.get('aggregate_verdict', 'N/A')}")
        print(f"Score: {sample.get('aggregate_score', 'N/A')}")
        
        # Show what claim was judged
        details = sample.get('details_json', {})
        if isinstance(details, dict) and details.get('citations'):
            first_cit = details['citations'][0]
            print(f"\nFirst citation judgment:")
            print(f"  URL: {first_cit.get('url', 'N/A')}")
            print(f"  Verdict: {first_cit.get('verdict', 'N/A')}")
            reason = first_cit.get('reason', '')
            if len(reason) > 200:
                reason = reason[:200] + "..."
            print(f"  Reason: {reason}")
    
    print(f"\n{'='*80}")
    if avg_score > 0:
        print("✅ FIX VERIFIED - Scores are non-zero, judging is working correctly!")
    else:
        print("⚠️  WARNING - Scores are still 0, something may be wrong")
    print(f"{'='*80}\n")
    
    # Save result
    result = {
        "provider": "PubMed",
        "session_id": session_id,
        "edges_judged": len(judged_edges),
        "edges_with_citations": len(aggregate_scores),
        "average_score": avg_score,
        "duration_seconds": duration,
        "verdict_counts": dict(verdict_counts),
        "judged_edges": judged_edges
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"test_fixed_judging_pubmed_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_file}\n")

if __name__ == "__main__":
    test_pubmed()



