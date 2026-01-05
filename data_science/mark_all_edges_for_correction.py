#!/usr/bin/env python3
"""
Mark All Edges for Correction

This script forces all edges in a session to be marked as candidates for correction
by setting a temporary property. This is useful for testing the corrector on edges
that haven't been judged yet or were judged as CORRECT but we want to test correction anyway.

Usage:
    python mark_all_edges_for_correction.py <session_id>
"""

import os
import sys
import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python mark_all_edges_for_correction.py <session_id>")
        return 1
    
    session_id = sys.argv[1]
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    print(f"Connecting to Neo4j at {uri}...")
    
    with driver.session() as session:
        # Verify session exists
        result = session.run(
            "MATCH (n:variable {session_id: $sid}) RETURN count(n) as c", 
            {"sid": session_id}
        ).single()
        
        if not result or result["c"] == 0:
            print(f"❌ Session {session_id} not found or has no nodes.")
            return 1
            
        print(f"✅ Session found with {result['c']} nodes.")
        
        # Mark all edges as 'INCORRECT' judge verdict to force correction
        # We use 'judge_verdict' because that's what the corrector query looks for
        print("Marking all edges as candidates for correction...")
        
        update_query = """
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id = $sid AND t.session_id = $sid
        SET r.judge_verdict = 'INCORRECT',
            r.judge_message = 'Forced correction test'
        RETURN count(r) as updated_count
        """
        
        result = session.run(update_query, {"sid": session_id}).single()
        updated = result["updated_count"]
        
        print(f"✅ Marked {updated} edges for correction.")
        
    driver.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())




