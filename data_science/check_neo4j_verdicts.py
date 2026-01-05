import os
from modules import Neo4jClient

# Connect to Neo4j using environment variables
db = Neo4jClient(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
)

# The latest corrected session ID from session_info.txt
session_id = '568d4702-0634-4f31-87df-bbab0c6feb5a'

print('=' * 100)
print('CHECKING NEO4J DATABASE FOR CORRECTOR SELECTION CRITERIA')
print('=' * 100)

# 1. Check what the corrector query would return
corrector_query = '''
MATCH (s:variable)-[r]->(t:variable)
WHERE s.session_id = $sid AND t.session_id = $sid
AND (
    r.judge_verdict IN ['INCONSISTENT','Contradicted','Not supported','NO_MOTIVATION','NO_CITATION'] OR
    coalesce(r.aggregate_verdict,'') IN ['Not supported', 'INCORRECT']
)
RETURN s.name AS src, t.name AS tgt, type(r) AS rel_type, 
       r.judge_verdict AS judge_verdict,
       r.aggregate_verdict AS aggregate_verdict
'''

print('\n🔍 Running corrector selection query...')
with db._get_session() as session:
    corrector_results = list(session.run(corrector_query, {'sid': session_id}))

print(f'\n✅ Edges matching corrector criteria: {len(corrector_results)}')
if len(corrector_results) > 0:
    print('\nFirst 10 matching edges:')
    for i, rec in enumerate(corrector_results[:10]):
        print(f'  {i+1}. {rec["src"]} -> {rec["tgt"]} | judge_verdict={rec["judge_verdict"]} | aggregate_verdict={rec["aggregate_verdict"]}')

# 2. Check all verdict distributions in Neo4j
all_verdicts_query = '''
MATCH (s:variable)-[r]->(t:variable)
WHERE s.session_id = $sid AND t.session_id = $sid
RETURN r.judge_verdict AS judge_verdict, 
       r.aggregate_verdict AS aggregate_verdict,
       count(*) AS count
'''

print('\n\n📊 ALL VERDICT DISTRIBUTION IN NEO4J:')
with db._get_session() as session:
    verdict_dist = list(session.run(all_verdicts_query, {'sid': session_id}))

for rec in verdict_dist:
    print(f'  judge_verdict={rec["judge_verdict"]}, aggregate_verdict={rec["aggregate_verdict"]}, count={rec["count"]}')

# 3. Sample edges with INCORRECT verdict
incorrect_query = '''
MATCH (s:variable)-[r]->(t:variable)
WHERE s.session_id = $sid AND t.session_id = $sid
AND r.judge_verdict = 'INCORRECT'
RETURN s.name AS src, t.name AS tgt,
       r.judge_verdict AS judge_verdict,
       r.aggregate_verdict AS aggregate_verdict,
       r.is_corrupted AS is_corrupted
LIMIT 5
'''

print('\n\n📋 SAMPLE EDGES WITH INCORRECT VERDICT IN NEO4J:')
with db._get_session() as session:
    incorrect_edges = list(session.run(incorrect_query, {'sid': session_id}))

if len(incorrect_edges) > 0:
    for i, rec in enumerate(incorrect_edges):
        print(f'  {i+1}. {rec["src"]} -> {rec["tgt"]}')
        print(f'      judge_verdict: {rec["judge_verdict"]}')
        print(f'      aggregate_verdict: {rec["aggregate_verdict"]}')
        print(f'      is_corrupted: {rec["is_corrupted"]}')
else:
    print('  No edges with INCORRECT verdict found!')

# 4. Check if aggregate_verdict exists for edges
aggregate_check_query = '''
MATCH (s:variable)-[r]->(t:variable)
WHERE s.session_id = $sid AND t.session_id = $sid
RETURN 
    count(*) AS total_edges,
    sum(CASE WHEN r.aggregate_verdict IS NOT NULL THEN 1 ELSE 0 END) AS has_aggregate_verdict,
    sum(CASE WHEN r.judge_verdict IS NOT NULL THEN 1 ELSE 0 END) AS has_judge_verdict
'''

print('\n\n🔍 CHECKING PROPERTY EXISTENCE:')
with db._get_session() as session:
    prop_check = list(session.run(aggregate_check_query, {'sid': session_id}))
    
for rec in prop_check:
    print(f'  Total edges: {rec["total_edges"]}')
    print(f'  Edges with aggregate_verdict: {rec["has_aggregate_verdict"]}')
    print(f'  Edges with judge_verdict: {rec["has_judge_verdict"]}')

db.close()

print('\n' + '=' * 100)
print('ANALYSIS COMPLETE')
print('=' * 100)
