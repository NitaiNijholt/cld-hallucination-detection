#!/usr/bin/env python3
"""
Import Judge Verdicts from Excel to Neo4j

This script reads judge verdicts from an RQ1a judged Excel file and writes them
as properties on the corresponding edges in Neo4j.

Usage:
    python import_verdicts_to_neo4j.py <excel_file> <session_id>
"""

import sys
import os
import pandas as pd
from neo4j import GraphDatabase

def main():
    if len(sys.argv) < 3:
        print("Usage: python import_verdicts_to_neo4j.py <excel_file> <session_id>")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    session_id = sys.argv[2]
    
    print("="*80)
    print("IMPORTING JUDGE VERDICTS TO NEO4J")
    print("="*80)
    print(f"Excel file: {excel_file}")
    print(f"Session ID: {session_id}")
    print()
    
    # Read Excel
    df = pd.read_excel(excel_file, sheet_name="All Edges")
    print(f"Loaded {len(df)} edges from Excel")
    
    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    updated = 0
    not_found = 0
    
    with driver.session() as session:
        for idx, row in df.iterrows():
            source = row['Source']
            target = row['Target']
            rel_type = row['Relationship Type']
            
            # Get judge verdict info from Excel
            judge_verdict = row.get('Judge Verdict 1')
            judge_message = row.get('Judge Message 1')
            aggregate_verdict = row.get('Aggregate Verdict')
            aggregate_score = row.get('Aggregate Score')
            
            # Update Neo4j edge
            query = f"""
            MATCH (s:variable {{name: $source, session_id: $sid}})-[r:{rel_type}]->(t:variable {{name: $target, session_id: $sid}})
            SET r.judge_verdict = $verdict,
                r.judge_message = $message,
                r.aggregate_verdict = $agg_verdict,
                r.aggregate_score = $agg_score
            RETURN count(r) as updated
            """
            
            result = session.run(query, {
                'source': source,
                'target': target,
                'sid': session_id,
                'verdict': judge_verdict,
                'message': str(judge_message) if judge_message and pd.notna(judge_message) else None,
                'agg_verdict': aggregate_verdict if pd.notna(aggregate_verdict) else None,
                'agg_score': float(aggregate_score) if pd.notna(aggregate_score) else None
            }).single()
            
            if result and result['updated'] > 0:
                updated += 1
            else:
                not_found += 1
                if not_found <= 5:  # Print first few misses
                    print(f"⚠️  Edge not found in Neo4j: {source} -[{rel_type}]-> {target}")
    
    driver.close()
    
    print(f"\n✅ Updated {updated} edges in Neo4j")
    print(f"⚠️  {not_found} edges not found in Neo4j")
    print("="*80)

if __name__ == "__main__":
    main()




