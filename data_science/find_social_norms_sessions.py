#!/usr/bin/env python3
"""Find Social Norms session IDs in Neo4j."""

from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'bvdGGOJxqQp3Wdz'))

with driver.session() as session:
    # Find sessions with Social Norms variables
    result = session.run('''
        MATCH (v:variable)
        WHERE v.session_id IS NOT NULL 
        AND (v.name CONTAINS 'BMI' OR v.name CONTAINS 'Obesity')
        WITH v.session_id AS sid, count(v) AS var_count, max(v.created_at) AS last_update
        ORDER BY last_update DESC
        LIMIT 5
        RETURN sid, var_count, last_update
    ''')
    
    print("Recent Social Norms sessions:")
    print("-" * 80)
    for rec in result:
        sid = rec['sid']
        var_count = rec['var_count']
        last_update = rec['last_update']
        print(f"Session: {sid}")
        print(f"  Variables: {var_count}")
        print(f"  Last update: {last_update}")
        
        # Check what data is available
        sample = session.run('''
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid
            RETURN count(r) AS edge_count,
                   sum(CASE WHEN r.aggregate_score IS NOT NULL THEN 1 ELSE 0 END) AS has_agg,
                   sum(CASE WHEN r.judge_perplexity IS NOT NULL THEN 1 ELSE 0 END) AS has_j_perp,
                   sum(CASE WHEN r.generator_perplexity IS NOT NULL THEN 1 ELSE 0 END) AS has_g_perp,
                   sum(CASE WHEN r.gen_cosine_sim IS NOT NULL THEN 1 ELSE 0 END) AS has_cos
        ''', sid=sid).single()
        
        if sample:
            print(f"  Edges: {sample['edge_count']}")
            print(f"  With aggregate_score: {sample['has_agg']}")
            print(f"  With judge_perplexity: {sample['has_j_perp']}")
            print(f"  With generator_perplexity: {sample['has_g_perp']}")
            print(f"  With cosine_sim: {sample['has_cos']}")
        print()

driver.close()
