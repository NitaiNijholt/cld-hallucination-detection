#!/usr/bin/env python3
"""
Run Corrector Only Script

This script takes an existing judged session ID, clones it, runs correction,
then re-judges ALL edges to measure motivation quality improvement.

Usage:
    python run_corrector_only.py <session_id> [corrector_model]

Example:
    python run_corrector_only.py 1234-5678-90ab gpt-4.1
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env.dev
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env.dev')

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent))

from modules import CausalDiscovery
from neo4j_session_cloner import SessionCloner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_corrector_only.py <session_id> [corrector_model] [judge_approach] [prompt_variant]")
        return 1
    
    session_id = sys.argv[1]
    corrector_model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1"
    judge_approach = sys.argv[3] if len(sys.argv) > 3 else "correctness"
    corrector_prompt_variant = sys.argv[4] if len(sys.argv) > 4 else "baseline"
    
    print("="*80)
    print(f"🧪 RUNNING CORRECTOR ON SESSION: {session_id}")
    print(f"🤖 Model: {corrector_model}")
    print(f"📝 Prompt Variant: {corrector_prompt_variant}")
    print("="*80)
    
    prompts_path = Path("data_science/parameter_tuning_experiments/alternative_prompts/prompts_correctness_baseline.yaml")

    try:
        # 1. Clone the session first (to preserve original)
        print("\n1. Cloning judged session for correction...")
        cloner = SessionCloner()
        corrected_session_id = cloner.clone_session(
            original_session_id=session_id,
            include_judge_data=True  # Keep judge verdicts for correction
        )
        print(f"✅ Cloned: {session_id[:8]}... → {corrected_session_id[:8]}...")
        
        # Generate descriptive timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save session mapping with full details
        mapping_file = f"RQ1b_correction_session_mapping_{session_id[:8]}_to_{corrected_session_id[:8]}_{timestamp}.txt"
        with open(mapping_file, "w") as f:
            f.write(f"# RQ1b Correction Experiment - Session Mapping\n")
            f.write(f"# Experiment: LLM-as-a-Corrector on Judged CLD\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Model: {corrector_model}\n")
            f.write(f"# Prompt Variant: original\n")
            f.write(f"\n")
            f.write(f"original_judged_session_id={session_id}\n")
            f.write(f"corrected_session_id={corrected_session_id}\n")
            f.write(f"corrector_model={corrector_model}\n")
            f.write(f"corrector_prompt_variant=original\n")
            f.write(f"timestamp={timestamp}\n")
            f.write(f"\n")
            f.write(f"# Full Session IDs:\n")
            f.write(f"# Original (Judged): {session_id}\n")
            f.write(f"# Corrected:         {corrected_session_id}\n")
        print(f"✅ Session mapping saved to {mapping_file}")
        
        # Also save short mapping for batch controller compatibility
        short_mapping = f"corrector_session_mapping_{session_id[:8]}.txt"
        with open(short_mapping, "w") as f:
            f.write(f"original_session_id={session_id}\n")
            f.write(f"corrected_session_id={corrected_session_id}\n")
            f.write(f"model={corrector_model}\n")
        print()
        
        # 2. Initialize CausalDiscovery
        print("2. Initializing corrector...")
        discovery = CausalDiscovery(
            target_variable="Unknown", 
            temporal_scale="Unknown", 
            spatial_scale="Unknown",
            yaml_path=str(prompts_path),
            dev_mode=False,
            generator_config={"provider": "openai", "model": corrector_model},
            corruptor_config={"provider": "openai", "model": corrector_model},
            judge_config={"provider": "openai", "model": corrector_model, "temperature": 0.0, "seed": 42},
            judge_enable_web_search=False,
            generator_enable_web_search=False,
            corruptor_enable_web_search=False,
            judge_temperature=0.0,
            judge_seed=42
        )
        
        # Set to corrected session
        discovery.session_id = corrected_session_id
        
        # Verify
        with discovery.graph_db._get_session() as session:
            result = session.run(
                "MATCH (n)-[r]->(m) WHERE n.session_id=$sid RETURN count(r) as c", 
                {"sid": corrected_session_id}
            ).single()
            count = result["c"] if result else 0
            print(f"✅ Cloned session has {count} edges")
        print()

        # 2b. Export judged (pre-corrected) session to Excel for comparison
        print("2b. Exporting judged (pre-corrected) session to Excel...")
        try:
            from modules import export_edges_comparison_to_excel
            
            judged_excel = f"judged_{session_id[:8]}_before_correction_{timestamp}.xlsx"
            
            # Temporarily set discovery to the original judged session
            original_session_id = discovery.session_id
            discovery.session_id = session_id
            
            # Export to Excel
            export_edges_comparison_to_excel(
                discovery=discovery,
                output_filename=judged_excel,
                embedding_enable=False,  # Skip expensive embeddings
                ci_compute_embeddings=False,  # Skip CI metrics
                experiment_params={
                    'session_id': session_id,
                    'timestamp': timestamp,
                    'judge_approach': judge_approach,
                    'experiment_type': 'RQ1b_Pre_Correction_Judged'
                }
            )
            print(f"   ✅ Judged session exported to: {judged_excel}")
            
            # Restore session ID to corrected session
            discovery.session_id = original_session_id
            
        except Exception as e:
            print(f"   ⚠️  Judged session Excel export skipped: {e}")
            import traceback
            traceback.print_exc()
        print()

        # 3. Run Correction
        print("3. Running correction...")
        outcomes = discovery.correct_edges_serial(
            corrector_models=[corrector_model],
            max_rounds=1,
            action_order=("revise", "recite", "remove"),
            rejudge_after=False,  # Don't re-judge during correction
            use_simple_corrector=False,
            corrector_prompt_variant=corrector_prompt_variant,  # Use command-line argument
            correction_parallel=True,       # Enable parallel correction
            correction_max_workers=5        # 5 concurrent workers
        )
        
        print(f"   ✅ Correction complete: {len(outcomes)} edges processed")
        
        # 4. Re-judge corrected edges using proper judge function
        print("4. Re-judging corrected edges...")
        # Use outcomes from THIS run (not Neo4j query which may include old corrections)
        # Filter for edges that were actually corrected in this run
        corrected_outcomes = [o for o in outcomes if o.get('corrected', False)]
        print(f"   Found {len(corrected_outcomes)} edges corrected in THIS run to re-judge")
        
        if corrected_outcomes:
            print(f"   Re-judging ONLY {len(corrected_outcomes)} corrected edges (not entire session) with approach: {judge_approach}")
            
            # Build edge filter list for selective judging from outcomes
            edge_filter = []
            for outcome in corrected_outcomes:
                # Get new_type if change_type was used, otherwise use original type
                edge_type = outcome.get('new_type', outcome.get('type', 'UNKNOWN'))
                edge_filter.append({
                    'source': outcome['source'],
                    'target': outcome['target'],
                    'type': edge_type
                })
            
            # Use the proper judge function with edge_filter to judge ONLY corrected edges
            judged_edges = discovery.judge_all_edges_with_citations_serial(
                judge_models=[corrector_model],
                num_judges=1,
                approach=judge_approach,
                judge_parallel=True,       # Enable parallel re-judging
                judge_max_workers=10,      # 10 concurrent workers
                edge_filter=edge_filter    # Only judge corrected edges
            )
            
            print(f"   ✅ Re-judged {len(corrected_outcomes)} corrected edges (saved cost by skipping unchanged edges)")
            
            # Calculate average score for corrected edges only using outcomes
            rejudge_scores = []
            for outcome in corrected_outcomes:
                src = outcome['source']
                tgt = outcome['target']
                rel_type = outcome.get('new_type', outcome.get('type', 'UNKNOWN'))
                
                # Query Neo4j for the re-judged score
                with discovery.graph_db._get_session() as neo_session:
                    query = f"""
                    MATCH (s:variable {{name:$src}})-[r:{rel_type}]->(t:variable {{name:$tgt}})
                    WHERE s.session_id=$sid
                    RETURN r.aggregate_score as score
                    """
                    result = neo_session.run(query, {'src': src, 'tgt': tgt, 'sid': corrected_session_id}).single()
                    if result and result['score'] is not None:
                        rejudge_scores.append(result['score'])
            
            avg_rejudge_score = sum(rejudge_scores) / len(rejudge_scores) if rejudge_scores else 0.0
            print(f"   Average re-judge score for corrected edges: {avg_rejudge_score:.3f}")
        else:
            print(f"   No edges were corrected, skipping re-judging")
            avg_rejudge_score = 0.0
            rejudge_scores = []  # Initialize empty for reporting
        
        print()
        
        # 6. Output Results
        # Get judge scores before correction
        from neo4j import GraphDatabase
        neo_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz"))
        )
        
        with neo_driver.session() as neo_session:
            query = '''
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND r.aggregate_score IS NOT NULL
            RETURN avg(r.aggregate_score) as avg_score,
                   count(*) as edges_with_scores
            '''
            result_before = neo_session.run(query, {'sid': session_id}).single()
            avg_score_before = result_before['avg_score'] if result_before and result_before['avg_score'] else 0.0
            edges_before = result_before['edges_with_scores'] if result_before else 0
        
        neo_driver.close()
        
        print("="*80)
        print("✅ CORRECTION + RE-JUDGING COMPLETE")
        print("="*80)
        print(f"Original (Judged) Session: {session_id}")
        print(f"Corrected Session:         {corrected_session_id}")
        print()
        print(f"Judge Scores:")
        print(f"  Before Correction: {avg_score_before:.3f} (avg of {edges_before} edges)")
        print(f"  After Correction:  {avg_rejudge_score:.3f} (avg of {len(rejudge_scores)} corrected edges)")
        print(f"  Delta: {avg_rejudge_score - avg_score_before:+.3f}")
        print()
        print(f"Total actions attempted: {len(outcomes)}")
        
        corrected_count = sum(1 for o in outcomes if o.get("corrected"))
        print(f"Successfully corrected edges: {corrected_count}")
        
        # Save detailed log with descriptive name
        outcomes_file = f"RQ1b_correction_outcomes_{session_id[:8]}_to_{corrected_session_id[:8]}_{timestamp}.json"
        with open(outcomes_file, "w") as f:
            json.dump(outcomes, f, indent=2)
        print(f"Detailed outcomes saved to: {outcomes_file}")
        
        # Also save short name for batch controller compatibility
        short_outcomes = f"correction_outcomes_{corrected_session_id[:8]}.json"
        with open(short_outcomes, "w") as f:
            json.dump(outcomes, f, indent=2)
        
        # 7. Export corrected session to Excel (same format as RQ1a judged files)
        print("5. Exporting corrected session to Excel...")
        try:
            from modules import export_edges_comparison_to_excel
            
            corrected_excel = f"corrected_{session_id[:8]}_to_{corrected_session_id[:8]}_{timestamp}.xlsx"
            
            # Export to Excel using the discovery object
            export_edges_comparison_to_excel(
                discovery=discovery,
                output_filename=corrected_excel,
                embedding_enable=False,  # Skip expensive embeddings
                ci_compute_embeddings=False,  # Skip CI metrics
                experiment_params={
                    'session_id': corrected_session_id,
                    'original_session_id': session_id,
                    'timestamp': timestamp,
                    'corrector_model': corrector_model,
                    'experiment_type': 'RQ1b_Correction'
                }
            )
            print(f"   ✅ Corrected session exported to: {corrected_excel}")
            
        except Exception as e:
            print(f"   ⚠️  Excel export skipped: {e}")
            import traceback
            traceback.print_exc()
        
        # Print summary of actions
        actions = {}
        for o in outcomes:
            act = o.get("action")
            actions[act] = actions.get(act, 0) + 1
            
        print("\nAction Breakdown:")
        for act, count in actions.items():
            print(f"  - {act}: {count}")
        
        print(f"\n📝 OUTPUT FILES:")
        print(f"   Session Mapping:  {mapping_file}")
        print(f"   Outcomes JSON:    {outcomes_file}")
        print(f"   Short Mapping:    {short_mapping} (for batch controller)")
        print(f"   Short Outcomes:   {short_outcomes} (for batch controller)")
        print()
        print(f"📝 SESSION IDs (Full):")
        print(f"   Original (Judged): {session_id}")
        print(f"   Corrected:         {corrected_session_id}")
        print()
        print(f"📝 EXPERIMENT DESCRIPTION:")
        print(f"   Type: RQ1b Correction (LLM-as-a-Corrector)")
        print(f"   Model: {corrector_model}")
        print(f"   Timestamp: {timestamp}")
        print()
        print(f"Use corrected session ID for F1 calculation.")
        
        cloner.close()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
