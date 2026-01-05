#!/usr/bin/env python3
"""Find recent Neo4j sessions with edge data."""

from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'bvdGGOJxqQp3Wdz'))

with driver.session() as session:
    result = session.run('''
        MATCH (s:variable)-[r]->(t:variable)
        WHERE s.session_id IS NOT NULL
        WITH s.session_id AS sid, count(r) AS edge_count
        WHERE edge_count > 0
        RETURN sid, edge_count
        ORDER BY edge_count DESC
        LIMIT 20
    ''')
    
    print("Recent sessions with edges:")
    print("-" * 80)
    for rec in result:
        sid = rec['sid']
        edge_count = rec['edge_count']
        print(f"  Session: {sid}")
        print(f"    Edges: {edge_count}")
        
        # Get a sample edge to check what data is available
        sample = session.run('''
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid
            RETURN s.name AS src, t.name AS tgt, type(r) AS rel_type,
                   r.aggregate_score AS agg_score,
                   r.judge_perplexity AS j_perp,
                   r.generator_perplexity AS g_perp,
                   r.gen_cosine_sim AS g_cos,
                   r.judge_cosine_sim AS j_cos,
                   r.classification AS classif
            LIMIT 1
        ''', sid=sid).single()
        
        if sample:
            print(f"    Sample edge: {sample['src']} -[{sample['rel_type']}]-> {sample['tgt']}")
            print(f"    Has agg_score: {sample['agg_score'] is not None}")
            print(f"    Has judge_perp: {sample['j_perp'] is not None}")
            print(f"    Has gen_perp: {sample['g_perp'] is not None}")
            print(f"    Has gen_cosine: {sample['g_cos'] is not None}")
            print(f"    Has judge_cosine: {sample['j_cos'] is not None}")
            print(f"    Has classification: {sample['classif'] is not None}")
        print()

driver.close()
