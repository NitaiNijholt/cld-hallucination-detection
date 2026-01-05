#!/usr/bin/env python3
"""
Check judging progress for Neo4j sessions
"""
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env.dev")

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def check_session_progress(session_ids):
    """Check judging progress for given session IDs."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    results = []
    with driver.session() as session:
        for sid in session_ids:
            # Query to count total edges and judged edges
            query = """
            MATCH (s:variable {session_id: $session_id})-[r]->(t:variable {session_id: $session_id})
            RETURN 
                count(r) as total_edges,
                count(CASE WHEN r.judge_verdict IS NOT NULL THEN 1 END) as judged_edges
            """
            result = session.run(query, session_id=sid)
            record = result.single()
            
            if record:
                total = record["total_edges"]
                judged = record["judged_edges"]
                percentage = (judged / total * 100) if total > 0 else 0
                results.append({
                    "session_id": sid,
                    "total": total,
                    "judged": judged,
                    "percentage": percentage
                })
            else:
                results.append({
                    "session_id": sid,
                    "total": 0,
                    "judged": 0,
                    "percentage": 0
                })
    
    driver.close()
    return results

if __name__ == "__main__":
    # Session IDs from the multirunner log
    session_ids = [
        "4fd98d73-f801-476f-85be-84119938cec6",  # run_1 cot
        "517b9b0a-036d-4c81-ab56-585db00641f0",  # run_1 mechanistic
        "fef78998-dee3-40ec-baf0-7521fb2760ac",  # run_2 cot
        "c1ac48ee-c315-40a1-80d8-741e61e1d93d",  # run_2 mechanistic
        "fe16f921-eb09-4a7a-9c46-0c9b19e2f9f3",  # run_3 cot (assumed)
    ]
    
    # Allow override from command line
    if len(sys.argv) > 1:
        session_ids = sys.argv[1:]
    
    print("=" * 80)
    print("CHECKING JUDGING PROGRESS")
    print("=" * 80)
    
    results = check_session_progress(session_ids)
    
    for i, result in enumerate(results, 1):
        sid = result["session_id"]
        total = result["total"]
        judged = result["judged"]
        pct = result["percentage"]
        
        print(f"\n{i}. Session: {sid[:8]}...")
        print(f"   Total edges: {total}")
        print(f"   Judged edges: {judged}")
        print(f"   Progress: {judged}/{total} ({pct:.1f}%)")
        
        if pct < 100:
            remaining = total - judged
            print(f"   ⚠️  Remaining: {remaining} edges")
        else:
            print(f"   ✅ COMPLETE")
    
    print("\n" + "=" * 80)
