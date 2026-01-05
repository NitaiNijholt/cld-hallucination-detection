#!/usr/bin/env python3
"""
Comparison experiment for citation search providers.

Tests different search providers (Semantic Scholar, PubMed, Perplexity, Brave)
and measures their coverage and URL quality for the Alzheimer's expert CLD.

Output: CSV with comparison metrics for each provider.
"""

import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery
from neo4j_session_cloner import clone_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
CITATION_HINTS_JSON = BASE_DIR / "Alzheimers_disease_citation_hints.json"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_citation_hints():
    """Load citation hints from JSON."""
    with open(CITATION_HINTS_JSON, 'r', encoding='utf-8') as f:
        hints_json = json.load(f)
    
    # Convert from "Source -> Target" strings to tuples
    hint_map = {}
    for edge_key, citations in hints_json.items():
        parts = edge_key.split(' -> ')
        if len(parts) == 2:
            hint_map[(parts[0], parts[1])] = citations
    
    return hint_map


def count_log_metrics(log_file: str, provider: str) -> dict:
    """
    Parse log file to count rate limits, API calls, and failures.
    
    Returns:
        Dict with: rate_limit_hits, api_calls_made, failed_searches
    """
    metrics = {
        "rate_limit_hits": 0,
        "api_calls_made": 0,
        "failed_searches": 0,
        "successful_searches": 0
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Count rate limit hits
        for line in lines:
            if "Rate limit hit" in line or "429" in line:
                metrics["rate_limit_hits"] += 1
            if "No URLs found for" in line:
                metrics["failed_searches"] += 1
            if f"Found via {provider.replace('_', ' ').title()}" in line or \
               f"Found via {provider.upper()}" in line or \
               f"[{provider.upper()}]" in line and "Found" in line:
                metrics["successful_searches"] += 1
        
        # Estimate API calls (each edge with citations = 1-2 API calls)
        # Plus failed searches also made API calls
        metrics["api_calls_made"] = metrics["successful_searches"] + metrics["failed_searches"]
        
    except Exception as e:
        logger.warning(f"Could not parse log file: {e}")
    
    return metrics


def get_provider_info(provider: str) -> dict:
    """
    Get actual provider information - whether we have API keys, if it's free, etc.
    
    Returns:
        Dict with: has_api_key, is_free, billing_model
    """
    import os
    
    info = {
        "semantic_scholar": {
            "has_api_key": False,
            "is_free": True,
            "billing_model": "Free (rate-limited to ~100 requests/5min)",
            "cost_proxy": "api_calls"
        },
        "pubmed": {
            "has_api_key": False,
            "is_free": True,
            "billing_model": "Free (NCBI guidelines: max 3 req/sec without key)",
            "cost_proxy": "api_calls"
        },
        "openalex": {
            "has_api_key": False,
            "is_free": True,
            "billing_model": "Free (polite pool: 10 req/sec, 100k req/day)",
            "cost_proxy": "api_calls"
        },
        "perplexity": {
            "has_api_key": bool(os.getenv("PERPLEXITY_API_KEY") and os.getenv("PERPLEXITY_API_KEY") != "your-perplexity-api-key"),
            "is_free": False,
            "billing_model": "Token-based ($1 per 1M input tokens, $3 per 1M output tokens for sonar model)",
            "cost_proxy": "tokens"
        },
        "brave": {
            "has_api_key": bool(os.getenv("BRAVE_SEARCH_API_KEY") and os.getenv("BRAVE_SEARCH_API_KEY") != "your-brave-search-api-key"),
            "is_free": False,
            "billing_model": "Per-query ($5/month for 2000 queries, then $0.005/query)",
            "cost_proxy": "api_calls"
        }
    }
    
    return info.get(provider, {
        "has_api_key": False,
        "is_free": True,
        "billing_model": "Unknown",
        "cost_proxy": "api_calls"
    })


def test_provider(provider_name: str, hint_map: dict, discovery: CausalDiscovery, log_file: str) -> dict:
    """
    Test a single provider and return comprehensive metrics.
    
    Returns:
        Dict with metrics: edges_updated, total_urls, cost, rate_limits, etc.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING PROVIDER: {provider_name}")
    logger.info(f"{'='*80}")
    
    start_time = datetime.now()
    
    # Clear log before starting
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== Testing {provider_name} ===\n")
    except:
        pass
    
    try:
        # Run citation refiller with this provider
        edges_updated = discovery.fill_in_missing_citations_with_hints(
            hint_map=hint_map,
            search_provider=provider_name,
            max_urls_per_citation=1,  # Only get the best URL per citation
            use_trusted_domains=False
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Query Neo4j to count URLs found
        with discovery.graph_db._get_session() as sess:
            result = sess.run("""
                MATCH (s:variable)-[r]->(t:variable)
                WHERE s.session_id = $sid AND t.session_id = $sid
                AND r.citations IS NOT NULL AND size(r.citations) > 0
                RETURN count(r) as edges_with_citations,
                       sum(size(r.citations)) as total_urls
            """, {"sid": discovery.session_id})
            
            record = result.single()
            edges_with_citations = record["edges_with_citations"] if record else 0
            total_urls = record["total_urls"] if record else 0
        
        # Parse log for detailed metrics
        log_metrics = count_log_metrics(log_file, provider_name)
        
        # Get provider info (API key status, billing model)
        provider_info = get_provider_info(provider_name)
        
        # Calculate efficiency metrics
        edges_without_citations = len(hint_map) - edges_with_citations
        retrieval_failure_rate = log_metrics["failed_searches"] / len(hint_map) if len(hint_map) > 0 else 0
        rate_limit_impact = log_metrics["rate_limit_hits"] / log_metrics["api_calls_made"] if log_metrics["api_calls_made"] > 0 else 0
        urls_per_api_call = total_urls / log_metrics["api_calls_made"] if log_metrics["api_calls_made"] > 0 else 0
        
        metrics = {
            "provider": provider_name,
            "edges_processed": len(hint_map),
            "edges_updated": edges_updated,
            "edges_with_citations": edges_with_citations,
            "edges_without_citations": edges_without_citations,
            "coverage_pct": round(edges_with_citations / len(hint_map) * 100, 1) if len(hint_map) > 0 else 0,
            "total_urls": total_urls,
            "duration_seconds": round(duration, 1),
            "api_calls_made": log_metrics["api_calls_made"],
            "rate_limit_hits": log_metrics["rate_limit_hits"],
            "failed_searches": log_metrics["failed_searches"],
            "successful_searches": log_metrics["successful_searches"],
            "retrieval_failure_rate": round(retrieval_failure_rate, 3),
            "rate_limit_impact_pct": round(rate_limit_impact * 100, 1),
            "urls_per_api_call": round(urls_per_api_call, 2),
            "has_api_key": provider_info["has_api_key"],
            "is_free": provider_info["is_free"],
            "billing_model": provider_info["billing_model"],
            "cost_proxy_metric": provider_info["cost_proxy"],
            "success": True,
            "error": None
        }
        
        logger.info(f"✅ {provider_name}: {edges_with_citations}/{len(hint_map)} edges ({metrics['coverage_pct']}%)")
        logger.info(f"   Total URLs: {total_urls}")
        logger.info(f"   Time: {duration:.1f}s, API Calls: {log_metrics['api_calls_made']}, Rate Limits: {log_metrics['rate_limit_hits']}")
        logger.info(f"   Billing: {provider_info['billing_model']}")
        logger.info(f"   Efficiency: {urls_per_api_call:.2f} URLs per API call")
        
    except Exception as e:
        logger.error(f"❌ {provider_name} failed: {e}")
        import traceback
        traceback.print_exc()
        
        provider_info = get_provider_info(provider_name)
        
        metrics = {
            "provider": provider_name,
            "edges_processed": len(hint_map),
            "edges_updated": 0,
            "edges_with_citations": 0,
            "edges_without_citations": len(hint_map),
            "coverage_pct": 0,
            "total_urls": 0,
            "duration_seconds": 0,
            "api_calls_made": 0,
            "rate_limit_hits": 0,
            "failed_searches": len(hint_map),
            "successful_searches": 0,
            "retrieval_failure_rate": 1.0,
            "rate_limit_impact_pct": 0,
            "urls_per_api_call": 0,
            "has_api_key": provider_info["has_api_key"],
            "is_free": provider_info["is_free"],
            "billing_model": provider_info["billing_model"],
            "cost_proxy_metric": provider_info["cost_proxy"],
            "success": False,
            "error": str(e)
        }
    
    return metrics


def clear_citations(discovery: CausalDiscovery):
    """Clear all citations from Neo4j edges to prepare for next test."""
    with discovery.graph_db._get_session() as sess:
        sess.run("""
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND t.session_id = $sid
            SET r.citations = NULL, r.relevance = NULL
        """, {"sid": discovery.session_id})
    logger.info("✅ Cleared all citations for next test")


def main():
    """
    Main comparison experiment.
    """
    print("\n" + "="*80)
    print("CITATION PROVIDER COMPARISON EXPERIMENT")
    print("="*80)
    
    # Load citation hints
    logger.info("\n📚 Loading citation hints...")
    hint_map = load_citation_hints()
    logger.info(f"   Loaded {len(hint_map)} edges with citation hints")
    
    # Create a single CausalDiscovery instance that we'll reuse
    logger.info("\n🔧 Initializing CausalDiscovery...")
    discovery = CausalDiscovery(
        target_variable="Alzheimer's disease risk",
        temporal_scale="Chronic, aging-related",
        spatial_scale="Older adults",
        yaml_path=str(Path(__file__).parent / "prompts_Nitai_C.yaml"),
        generator_config={"provider": "openai", "model": "gpt-4o"}
    )
    
    # Load the CLD into Neo4j (variables and edges)
    logger.info("\n📊 Loading CLD into Neo4j...")
    VARS_JSON = BASE_DIR / "Alzheimers_disease_vars_data.json"
    EDGES_JSON = BASE_DIR / "Alzheimers_disease_edges_data.json"
    
    with open(VARS_JSON, 'r') as f:
        variables_data = json.load(f)
    with open(EDGES_JSON, 'r') as f:
        edges_data = json.load(f)['edges']
    
    # Create variables
    with discovery.graph_db._get_session() as session:
        for var_data in variables_data:
            session.run("""
                CREATE (n:variable {
                    name: $name,
                    description: $description,
                    session_id: $session_id,
                    target: false,
                    deleted: false
                })
            """, {
                "name": var_data["name"],
                "description": var_data.get("definition", ""),
                "session_id": discovery.session_id
            })
            discovery.variables.append(var_data["name"])
        
        # Create edges
        for edge in edges_data:
            rel_type = edge["type"].upper()  # POSITIVE or NEGATIVE
            session.run(f"""
                MATCH (s:variable {{name: $source, session_id: $session_id}})
                MATCH (t:variable {{name: $target, session_id: $session_id}})
                CREATE (s)-[r:{rel_type}]->(t)
                SET r.session_id = $session_id,
                    r.motivation = 'Expert-validated',
                    r.expert_validated = true
            """, {
                "source": edge["source"],
                "target": edge["target"],
                "session_id": discovery.session_id
            })
    
    logger.info(f"✅ Loaded {len(variables_data)} variables and {len(edges_data)} edges")
    
    # Test each provider
    providers_to_test = [
        "semantic_scholar",  # Free, academic-focused, slow (3s/call)
        "pubmed",            # Free, medical papers, fast (0.35s/call)
        "openalex",          # Free, academic, fast (0.11s/call, 10 req/sec)
        "perplexity",        # Paid, requires API key - will skip if not available
        "brave",             # Paid, requires API key - will skip if not available
    ]
    
    # Filter providers based on API key availability
    available_providers = []
    for provider in providers_to_test:
        provider_info = get_provider_info(provider)
        if provider_info["is_free"] or provider_info["has_api_key"]:
            available_providers.append(provider)
            logger.info(f"✓ {provider}: {'FREE' if provider_info['is_free'] else 'API key found'}")
        else:
            logger.warning(f"✗ {provider}: Skipping (no API key)")
    
    providers_to_test = available_providers
    logger.info(f"\n📋 Testing {len(providers_to_test)} providers: {', '.join(providers_to_test)}\n")
    
    # Store base session ID
    base_session_id = discovery.session_id
    logger.info(f"Base CLD session ID: {base_session_id}")
    
    # Load .env for Neo4j credentials  
    load_dotenv(Path(__file__).parent.parent / ".env.dev")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    results = []
    session_mapping = {
        "base_session_id": base_session_id,
        "timestamp": datetime.now().isoformat(),
        "provider_sessions": {}
    }
    log_file = "/tmp/citation_provider_test.log"
    
    for provider in providers_to_test:
        logger.info(f"\n🔄 Cloning base CLD for provider: {provider}")
        
        # Clone the base session for this provider
        cloned_session_id = clone_session(
            original_session_id=base_session_id,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            include_judge_data=False
        )
        logger.info(f"   Cloned session ID: {cloned_session_id}")
        
        # Update discovery instance to use the cloned session
        discovery.session_id = cloned_session_id
        session_mapping["provider_sessions"][provider] = cloned_session_id
        
        # Test provider on the clone
        metrics = test_provider(provider, hint_map, discovery, log_file)
        metrics["session_id"] = cloned_session_id
        results.append(metrics)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Add any additional derived metrics
    df['avg_urls_per_edge'] = (df['total_urls'] / df['edges_with_citations'].replace(0, 1)).round(2)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"citation_provider_comparison_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Save session mapping for judging
    session_mapping_file = OUTPUT_DIR / f"citation_provider_sessions_{timestamp}.json"
    with open(session_mapping_file, 'w') as f:
        json.dump(session_mapping, f, indent=2)
    logger.info(f"\n💾 Session mapping saved to: {session_mapping_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\nTested {len(hint_map)} edges with citation hints\n")
    
    # Show main metrics
    summary_cols = ['provider', 'coverage_pct', 'total_urls', 'duration_seconds', 'is_free']
    print(df[summary_cols].to_string(index=False))
    
    print("\n" + "-"*80)
    print("API USAGE & EFFICIENCY METRICS")
    print("-"*80)
    
    efficiency_cols = ['provider', 'api_calls_made', 'rate_limit_hits', 'urls_per_api_call', 
                      'retrieval_failure_rate', 'rate_limit_impact_pct']
    print(df[efficiency_cols].to_string(index=False))
    
    print("\n" + "-"*80)
    print("BILLING & COST METRICS")
    print("-"*80)
    
    billing_cols = ['provider', 'billing_model', 'has_api_key', 'cost_proxy_metric']
    print(df[billing_cols].to_string(index=False))
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Determine winner (by coverage)
    best_provider = df.loc[df['coverage_pct'].idxmax()]
    print(f"\n🏆 BEST COVERAGE: {best_provider['provider']}")
    print(f"   Coverage: {best_provider['coverage_pct']}%")
    print(f"   Total URLs: {int(best_provider['total_urls'])}")
    print(f"   API Calls: {int(best_provider['api_calls_made'])}")
    print(f"   Rate Limits Hit: {int(best_provider['rate_limit_hits'])}")
    print(f"   Time: {best_provider['duration_seconds']:.1f}s")
    print(f"   Billing: {best_provider['billing_model']}")
    
    # Most efficient (highest URLs per API call)
    if df['urls_per_api_call'].max() > 0:
        most_efficient = df.loc[df['urls_per_api_call'].idxmax()]
        print(f"\n⚡ MOST EFFICIENT: {most_efficient['provider']}")
        print(f"   URLs per API call: {most_efficient['urls_per_api_call']:.2f}")
        print(f"   Coverage: {most_efficient['coverage_pct']}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("❌ Experiment failed:")
        sys.exit(1)

