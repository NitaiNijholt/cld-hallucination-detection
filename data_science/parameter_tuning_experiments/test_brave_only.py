#!/usr/bin/env python3
"""
Quick test of Brave Search citation provider after fixing import.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery
from neo4j_session_cloner import clone_session
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
OUTPUT_DIR = Path(__file__).parent / "results"

def load_citation_hints():
    """Load citation hints from JSON."""
    hint_file = BASE_DIR / "Alzheimers_disease_citation_hints.json"
    with open(hint_file, 'r') as f:
        return json.load(f)

def main():
    print("\n" + "="*80)
    print("TESTING BRAVE SEARCH (FIXED IMPORT)")
    print("="*80)
    
    # Load citation hints
    hint_map = load_citation_hints()
    logger.info(f"Loaded {len(hint_map)} edges with citation hints")
    
    # Load the existing session mapping to get the base session
    session_file = OUTPUT_DIR / "citation_provider_sessions_20251106_051504.json"
    with open(session_file, 'r') as f:
        sessions = json.load(f)
    
    base_session_id = sessions["base_session_id"]
    logger.info(f"Base session: {base_session_id}")
    
    # Clone the base session for Brave test
    load_dotenv(Path(__file__).parent.parent / ".env.dev")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    logger.info("Cloning base CLD for Brave test...")
    brave_session_id = clone_session(
        original_session_id=base_session_id,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        include_judge_data=False
    )
    logger.info(f"Cloned session ID: {brave_session_id}")
    
    # Create CausalDiscovery instance with the cloned session
    discovery = CausalDiscovery(
        target_variable="Alzheimer's disease risk",
        temporal_scale="Chronic, aging-related",
        spatial_scale="Older adults",
        yaml_path=str(Path(__file__).parent / "prompts_Nitai_C.yaml"),
        generator_config={"provider": "openai", "model": "gpt-4o"}
    )
    discovery.session_id = brave_session_id
    
    # Test Brave Search
    logger.info("\n🔍 Testing Brave Search with fixed import...")
    import time
    start_time = time.time()
    
    discovery.fill_in_missing_citations_with_hints(
        hint_map=hint_map,
        search_provider="brave",
        use_trusted_domains=False,
        max_urls_per_citation=1
    )
    
    duration = time.time() - start_time
    
    # Count results
    with discovery.graph_db._get_session() as session:
        result = session.run("""
            MATCH (s:variable)-[r]->(t:variable)
            WHERE r.session_id = $session_id
            AND EXISTS(r.citations)
            RETURN count(r) as edges_with_citations,
                   reduce(total = 0, c IN [r IN collect(r) | size(r.citations)] | total + c) as total_urls
        """, {"session_id": brave_session_id})
        
        record = result.single()
        edges_with_citations = record["edges_with_citations"]
        total_urls = record["total_urls"]
    
    print("\n" + "="*80)
    print("BRAVE SEARCH RESULTS (FIXED)")
    print("="*80)
    print(f"Session ID: {brave_session_id}")
    print(f"Edges with citations: {edges_with_citations}/150")
    print(f"Coverage: {edges_with_citations/150*100:.1f}%")
    print(f"Total URLs: {total_urls}")
    print(f"Duration: {duration:.1f}s")
    print("="*80)
    
    # Save updated session info
    sessions["provider_sessions"]["brave_fixed"] = brave_session_id
    output_file = OUTPUT_DIR / "citation_provider_sessions_brave_fixed.json"
    with open(output_file, 'w') as f:
        json.dump(sessions, f, indent=2)
    
    logger.info(f"Session info saved to: {output_file}")

if __name__ == "__main__":
    main()



