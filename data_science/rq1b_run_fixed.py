import sys
sys.path.insert(0, '.')
from modules import CausalDiscovery
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('='*80)
logger.info('RQ1b EXPERIMENT: Tool-based Corrector Agent (FINAL FIX)')
logger.info('='*80)

# Create discovery instance with gpt-4o-2024-11-20 using Social Norms CLD
discovery = CausalDiscovery(
    target_variable='Obesity Prevalence',
    temporal_scale='Years',
    spatial_scale='Population',
    yaml_path='../backend/configs/prompts.yaml',
    generator_config={'provider': 'openai', 'model': 'gpt-4o-2024-11-20'},
    corruptor_config={'provider': 'openai', 'model': 'gpt-4o-2024-11-20'},
    judge_config={'provider': 'openai', 'model': 'gpt-4o-mini'}
)

session_id = discovery.session_id
logger.info(f'Session: {session_id}')

# Phase 1
logger.info('\nPHASE 1: Generate base CLD')
discovery.generate_variables()
discovery.discover_relationships()

# Check edges
with discovery.graph_db._get_session() as session:
    result = session.run('''
        MATCH ()-[r]->()
        WHERE r.session_id = $sid
        RETURN count(r) as cnt,
               sum(CASE WHEN type(r)='POSITIVE' THEN 1 ELSE 0 END) as pos,
               sum(CASE WHEN type(r)='NEGATIVE' THEN 1 ELSE 0 END) as neg,
               sum(CASE WHEN type(r)='NONE' THEN 1 ELSE 0 END) as none
    ''', {'sid': session_id}).single()
    logger.info(f'Base edges: {result["cnt"]} (POS:{result["pos"]}, NEG:{result["neg"]}, NONE:{result["none"]})')

# Phase 2
logger.info('\nPHASE 2: Corrupt (0.3 rate)')
corrupted = discovery._corrupt_relationships(0.3)
logger.info(f'Corrupted: {corrupted}')

# Get corruption breakdown - FIXED QUERY
with discovery.graph_db._get_session() as session:
    result = session.run('''
        MATCH ()-[r]->()
        WHERE r.session_id = $sid AND r.is_corrupted = true
        RETURN count(r) as total,
               sum(CASE WHEN r.corruption_subtype = 'new_edge' THEN 1 ELSE 0 END) as new_edges,
               sum(CASE WHEN r.corruption_subtype = 'polarity_flip' THEN 1 ELSE 0 END) as flips,
               sum(CASE WHEN r.corruption_type IS NULL OR r.corruption_type = 'motivation' THEN 1 ELSE 0 END) as motivation_only
    ''', {'sid': session_id}).single()
    logger.info(f'  Breakdown: {result["total"]} total ({result["new_edges"]} new, {result["flips"]} flips, {result["motivation_only"]} motivation)')

# Phase 3a: Judge
logger.info('\nPHASE 3a: Judge all edges')
discovery.judge_all_edges()  
logger.info('Phase 3a complete')

# Phase 3b: Correct with agent - FIXED PARAMETER
logger.info('\nPHASE 3b: Correct with agent')
try:
    corrected = discovery.correct_edges_serial(corrector_models='gpt-4o-2024-11-20')  # FIXED
    logger.info(f'Processed: {corrected} edges')
except Exception as e:
    logger.error(f'Correction error: {e}')
    import traceback
    traceback.print_exc()

# Get correction stats
with discovery.graph_db._get_session() as session:
    corrections = session.run('''
        MATCH ()-[r]->()
        WHERE r.session_id = $sid AND r.corrected = true
        RETURN r.correction_action as action, count(*) as cnt
        ORDER BY cnt DESC
    ''', {'sid': session_id}).data()
    logger.info('Correction actions:')
    for c in corrections:
        logger.info(f'  {c["action"]}: {c["cnt"]}')

# Phase 3c: Re-judge after correction
logger.info('\nPHASE 3c: Re-judge after correction')
discovery.judge_all_edges()
logger.info('Phase 3c complete')

# Final analysis
with discovery.graph_db._get_session() as session:
    edges = session.run('MATCH ()-[r]->() WHERE r.session_id = $sid RETURN count(r) as cnt', {'sid': session_id}).single()
    logger.info(f'\nFinal edge count: {edges["cnt"]}')

logger.info(f'\n{"="*80}')
logger.info(f'EXPERIMENT COMPLETE! Session: {session_id}')
logger.info(f'{"="*80}')
print(f'DONE:{session_id}')

