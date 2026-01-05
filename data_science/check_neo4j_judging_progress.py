#!/usr/bin/env python3
"""
Check judging progress directly from Neo4j at relationship level
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

# Session IDs to check (from multirunner logs and backup file)
SESSIONS = {
    "run_1_cot": "4fd98d73-f801-476f-85be-84119938cec6",
    "run_1_mechanistic": "517b9b0a-036d-4c81-ab56-585db00641f0",
    "run_2_cot": "fef78998-dee3-40ec-baf0-7521fb2760ac",
    "run_2_mechanistic": "c1ac48ee-c315-40a1-80d8-741e61e1d93d",
    "run_3_cot": "fe16f921-eb09-4a7a-9c46-0c9b19e2f9f3",
}

def check_session_progress(driver, session_id, name):
    """Check judging progress for a specific session."""
    with driver.session() as session:
        # Query 1: Count total edges and judged edges
        query1 = """
        MATCH (s:variable {session_id: $session_id})-[r]->(t:variable {session_id: $session_id})
        RETURN 
            count(r) as total_edges,
            count(CASE WHEN r.judge_verdict IS NOT NULL THEN 1 END) as judged_edges,
            count(CASE WHEN r.judge_verdict = 'NO_CITATION' THEN 1 END) as no_citation_edges,
            count(CASE WHEN r.judge_verdict IS NOT NULL AND r.judge_verdict <> 'NO_CITATION' THEN 1 END) as properly_judged_edges
        """
        result1 = session.run(query1, session_id=session_id)
        record1 = result1.single()
        
        # Query 2: Check if nodes exist
        query2 = """
        MATCH (n:variable {session_id: $session_id})
        RETURN count(n) as total_nodes
        """
        result2 = session.run(query2, session_id=session_id)
        record2 = result2.single()
        
        # Query 3: Sample some edge verdicts
        query3 = """
        MATCH (s:variable {session_id: $session_id})-[r]->(t:variable {session_id: $session_id})
        WHERE r.judge_verdict IS NOT NULL
        RETURN r.judge_verdict as verdict, count(*) as count
        ORDER BY count DESC
        LIMIT 10
        """
        result3 = session.run(query3, session_id=session_id)
        verdict_counts = list(result3)
        
        return {
            "name": name,
            "session_id": session_id,
            "nodes": record2["total_nodes"] if record2 else 0,
            "total_edges": record1["total_edges"] if record1 else 0,
            "judged_edges": record1["judged_edges"] if record1 else 0,
            "no_citation": record1["no_citation_edges"] if record1 else 0,
            "properly_judged": record1["properly_judged_edges"] if record1 else 0,
            "verdict_distribution": verdict_counts
        }

def main():
    print("=" * 80)
    print("NEO4J JUDGING PROGRESS CHECK")
    print("=" * 80)
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Verify connection
        driver.verify_connectivity()
        print("✅ Connected to Neo4j\n")
        
        results = []
        for name, session_id in SESSIONS.items():
            print(f"Checking {name} ({session_id[:8]}...)...")
            result = check_session_progress(driver, session_id, name)
            results.append(result)
        
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)
        
        for result in results:
            total = result["total_edges"]
            judged = result["judged_edges"]
            no_cit = result["no_citation"]
            proper = result["properly_judged"]
            nodes = result["nodes"]
            
            print(f"\n📊 {result['name'].upper()}")
            print(f"   Session ID: {result['session_id']}")
            print(f"   Nodes: {nodes}")
            print(f"   Total edges: {total}")
            print(f"   Judged edges: {judged} ({judged/total*100:.1f}%)" if total > 0 else "   Judged edges: 0 (N/A)")
            print(f"   ├─ NO_CITATION: {no_cit}")
            print(f"   └─ Properly judged: {proper}")
            
            if judged > 0:
                print(f"\n   Verdict distribution:")
                for vd in result["verdict_distribution"]:
                    print(f"      - {vd['verdict']}: {vd['count']} edges")
            
            if total == 0:
                print(f"   ❌ SESSION NOT FOUND IN NEO4J")
            elif judged == 0:
                print(f"   ⚠️  NO EDGES JUDGED YET")
            elif judged < total:
                remaining = total - judged
                print(f"   ⚠️  INCOMPLETE: {remaining} edges not judged ({remaining/total*100:.1f}%)")
            else:
                print(f"   ✅ ALL EDGES JUDGED")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_sessions = len(results)
        complete_sessions = sum(1 for r in results if r["judged_edges"] == r["total_edges"] and r["total_edges"] > 0)
        partial_sessions = sum(1 for r in results if 0 < r["judged_edges"] < r["total_edges"])
        empty_sessions = sum(1 for r in results if r["total_edges"] == 0)
        
        print(f"Total sessions checked: {total_sessions}")
        print(f"Complete sessions: {complete_sessions}")
        print(f"Partially judged sessions: {partial_sessions}")
        print(f"Sessions not found: {empty_sessions}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()
