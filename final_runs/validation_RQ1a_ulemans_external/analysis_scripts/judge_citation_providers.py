#!/usr/bin/env python3
"""
Judge Citation Provider Sessions

This script runs citation-based judging on each provider's CLD session
to evaluate how well the citations support the causal relationships.

Usage:
    python judge_citation_providers.py
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "results"

def judge_provider_session(provider_name: str, session_id: str, base_session_id: str):
    """
    Run citation-based judging on a provider's session.
    
    Args:
        provider_name: Name of the citation provider
        session_id: Neo4j session ID for this provider's CLD
        base_session_id: Base CLD session ID (for context)
    
    Returns:
        Dict with judging results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"JUDGING PROVIDER: {provider_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"Session ID: {session_id}")
    
    # Create CausalDiscovery instance
    discovery = CausalDiscovery(
        target_variable="Alzheimer's disease risk",
        temporal_scale="Chronic, aging-related",
        spatial_scale="Older adults",
        yaml_path=str(BASE_DIR / "prompts_Nitai_C.yaml"),
        generator_config={"provider": "openai", "model": "gpt-4o"},
        judge_config={"provider": "openai", "model": "gpt-4.1"},
        judge_temperature=0.3  # Low temperature for consistent judgments
    )
    
    # Set to the provider's session
    discovery.session_id = session_id
    
    # Load variables from the session
    with discovery.graph_db._get_session() as neo4j_session:
        result = neo4j_session.run("""
            MATCH (n:variable {session_id: $session_id})
            RETURN n.name as name
        """, {"session_id": session_id})
        discovery.variables = [record["name"] for record in result]
    
    logger.info(f"Loaded {len(discovery.variables)} variables from session")
    
    # Run citation-based judging
    logger.info(f"Running citation-based judging with approach='per_citation_aggregate'...")
    import time
    start_time = time.time()
    
    try:
        # Use the per_citation_aggregate approach with parallel processing
        # Get Jina API key from environment
        import os
        jina_api_key = os.getenv("JINA_API_KEY", None)
        
        # Use Jina AI for citation fetching (NO fallback)
        judged_edges = discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"],  # Using GPT-4.1 for judging
            num_judges=1,  # Single judge for speed
            approach="per_citation_aggregate",  # Detailed scoring per citation
            judge_parallel=True,  # Enable parallel processing
            judge_max_workers=2,  # Process only 2 edges in parallel (reduced to avoid Jina rate limits)
            citation_fetcher="jina",  # Use Jina AI ONLY (no fallback)
            jina_timeout=60,  # Jina AI timeout
            jina_retries=2,  # Jina AI retries
            jina_api_key=jina_api_key  # Pass API key
        )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ Judging complete in {duration:.1f}s")
        logger.info(f"   Judged {len(judged_edges)} edges")
        
        # Calculate summary statistics
        if judged_edges:
            # Extract aggregate scores
            aggregate_scores = []
            verdicts = []
            
            for edge in judged_edges:
                # Get aggregate_score from top level (not nested in judge_details)
                score = edge.get("aggregate_score")
                if score is not None:  # Only include edges with scores (excludes NO_CITATION)
                    aggregate_scores.append(score)
                    verdicts.append(edge.get("aggregate_verdict", "unknown"))
            
            avg_score = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0
            
            # Count verdict types
            from collections import Counter
            verdict_counts = Counter(verdicts)
            
            logger.info(f"\n📊 Judging Summary:")
            logger.info(f"   Average aggregate score: {avg_score:.3f}")
            logger.info(f"   Verdict breakdown:")
            for verdict, count in verdict_counts.most_common():
                logger.info(f"     - {verdict}: {count} edges ({count/len(verdicts)*100:.1f}%)")
        
        return {
            "provider": provider_name,
            "session_id": session_id,
            "edges_judged": len(judged_edges),
            "duration_seconds": duration,
            "average_aggregate_score": avg_score if judged_edges else 0,
            "verdict_counts": dict(verdict_counts) if judged_edges else {},
            "judged_edges": judged_edges,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error judging {provider_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "provider": provider_name,
            "session_id": session_id,
            "edges_judged": 0,
            "duration_seconds": 0,
            "average_aggregate_score": 0,
            "verdict_counts": {},
            "judged_edges": [],
            "success": False,
            "error": str(e)
        }

def main():
    print("\n" + "="*80)
    print("CITATION PROVIDER JUDGING EXPERIMENT")
    print("="*80)
    
    # Load provider sessions
    logger.info("\n📚 Loading provider sessions...")
    
    # Load the original comparison results
    session_file = OUTPUT_DIR / "citation_provider_sessions_20251106_051504.json"
    with open(session_file, 'r') as f:
        sessions = json.load(f)
    
    # Load the fixed Brave session
    brave_fixed_file = OUTPUT_DIR / "citation_provider_sessions_brave_fixed.json"
    with open(brave_fixed_file, 'r') as f:
        brave_fixed = json.load(f)
    
    base_session_id = sessions["base_session_id"]
    logger.info(f"Base session: {base_session_id}")
    
    # Providers to judge (only Brave Fixed for comparison)
    providers_to_judge = [
        ("Brave (Fixed)", brave_fixed["provider_sessions"]["brave_fixed"]),
    ]
    
    logger.info(f"\n📋 Will judge {len(providers_to_judge)} providers:")
    for name, session_id in providers_to_judge:
        logger.info(f"   - {name}: {session_id}")
    
    # Judge each provider
    results = []
    
    for provider_name, session_id in providers_to_judge:
        result = judge_provider_session(provider_name, session_id, base_session_id)
        results.append(result)
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = OUTPUT_DIR / f"judge_results_{provider_name.replace(' ', '_')}_{timestamp}.json"
        with open(intermediate_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"💾 Intermediate results saved to: {intermediate_file}")
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        if r["success"]:
            verdict_breakdown = " | ".join([f"{k}: {v}" for k, v in r["verdict_counts"].items()])
            summary_data.append({
                "Provider": r["provider"],
                "Session ID": r["session_id"][:8] + "...",
                "Edges Judged": r["edges_judged"],
                "Avg Score": f"{r['average_aggregate_score']:.3f}",
                "Duration (s)": f"{r['duration_seconds']:.1f}",
                "Verdict Breakdown": verdict_breakdown
            })
        else:
            summary_data.append({
                "Provider": r["provider"],
                "Session ID": r["session_id"][:8] + "...",
                "Edges Judged": 0,
                "Avg Score": "ERROR",
                "Duration (s)": 0,
                "Verdict Breakdown": r.get("error", "Unknown error")
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = OUTPUT_DIR / f"judge_citation_providers_summary_{timestamp}.csv"
    df.to_csv(summary_file, index=False)
    
    # Save full results
    full_results_file = OUTPUT_DIR / f"judge_citation_providers_full_{timestamp}.json"
    with open(full_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("JUDGING COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(df.to_string(index=False))
    print(f"\n💾 Results saved to:")
    print(f"   - Summary: {summary_file}")
    print(f"   - Full results: {full_results_file}")
    print("="*80)

if __name__ == "__main__":
    main()

