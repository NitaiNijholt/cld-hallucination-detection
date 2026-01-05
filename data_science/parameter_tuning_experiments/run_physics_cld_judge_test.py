#!/usr/bin/env python3
"""
Run Judge Test on Physics CLD (Thermostat Heating System)

This script:
1. Loads the thermostat heating system CLD into Neo4j
2. Runs the citation-based judge on all edges
3. Compares results against Alzheimer's CLD performance

Usage:
    python run_physics_cld_judge_test.py
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging
import uuid

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import CausalDiscovery, run_discovery_experiment

# Paths
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
PHYSICS_CLD = {
    "name": "thermostat_heating_system",
    "excel": BASE_DIR / "thermostat_heating_system.xlsx",
    "vars_json": BASE_DIR / "thermostat_heating_system_vars_data.json",
    "edges_json": BASE_DIR / "thermostat_heating_system_edges_data.json",
    "context_json": BASE_DIR / "thermostat_heating_system_context_data.json",
    "citation_hints": BASE_DIR / "thermostat_heating_system_citation_hints.json"
}


def load_physics_cld_to_neo4j(approach: str = "correctness") -> str:
    """
    Load the physics CLD directly into Neo4j as a base session.
    
    Args:
        approach: 'correctness' or 'citation' - determines which yaml file to use
    
    Returns:
        session_id: The Neo4j session ID
    """
    logger.info("="*80)
    logger.info("LOADING PHYSICS CLD INTO NEO4J")
    logger.info("="*80)
    
    # Load JSON data
    with open(PHYSICS_CLD["vars_json"], 'r') as f:
        vars_data = json.load(f)
    
    with open(PHYSICS_CLD["edges_json"], 'r') as f:
        edges_data = json.load(f)
    
    with open(PHYSICS_CLD["context_json"], 'r') as f:
        context_data = json.load(f)
    
    with open(PHYSICS_CLD["citation_hints"], 'r') as f:
        citation_hints = json.load(f)
    
    logger.info(f"Loaded: {len(vars_data)} variables, {len(edges_data)} edges")
    
    # Select yaml file based on approach
    if approach == "citation":
        yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_citation_baseline.yaml"
    else:
        yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_correctness_baseline.yaml"
    
    logger.info(f"Using yaml: {yaml_path.name}")
    
    # Create CausalDiscovery instance
    discovery = CausalDiscovery(
        target_variable=context_data.get("Target", "Room Temperature"),
        temporal_scale=context_data.get("Temporal Scale", "Minutes to hours"),
        spatial_scale=context_data.get("Spatial Scale", "Single room"),
        yaml_path=str(yaml_path),
        generator_config={"provider": "openai", "model": "gpt-4.1"},
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.0}
    )
    
    session_id = discovery.session_id
    logger.info(f"Created session: {session_id}")
    
    # Load variables into Neo4j
    logger.info("Loading variables...")
    for var in vars_data:
        with discovery.graph_db._get_session() as session:
            session.run("""
                CREATE (n:variable {
                    name: $name,
                    description: $definition,
                    session_id: $session_id,
                    target: false,
                    deleted: false
                })
            """, {
                "name": var["name"],
                "definition": var.get("definition", ""),
                "session_id": session_id
            })
        discovery.variables.append(var["name"])
    
    # Physics-based motivations for each edge
    PHYSICS_MOTIVATIONS = {
        "Room Temperature -> Temperature Gap": 
            "According to the definition of Temperature Gap (the difference between Desired Temperature and Room Temperature), "
            "an increase in Room Temperature directly causes a decrease in the Temperature Gap. This is a mathematical relationship: "
            "Gap = Desired - Actual, so higher Room Temperature means smaller gap. This negative causal relationship is fundamental to "
            "thermostat feedback control systems.",
        
        "Desired Temperature -> Temperature Gap":
            "The Temperature Gap is defined as the difference between the Desired Temperature (setpoint) and the actual Room Temperature. "
            "An increase in the Desired Temperature directly causes an increase in the Temperature Gap, assuming Room Temperature stays constant. "
            "This positive causal relationship drives the heating system to work harder to reach the higher setpoint.",
        
        "Temperature Gap -> Heating Rate":
            "In thermostat control systems, the Temperature Gap triggers the heating system. When the gap is positive (room is too cold), "
            "the thermostat activates the furnace, causing an increase in Heating Rate. Larger temperature gaps cause more aggressive heating "
            "(either through longer on-cycles in bang-bang control or proportionally higher output in proportional control). "
            "This positive causal relationship is the core of negative feedback temperature regulation.",
        
        "Heating Rate -> Room Temperature":
            "Based on the First Law of Thermodynamics (conservation of energy), heat added to a system increases its internal energy. "
            "An increase in Heating Rate directly causes an increase in Room Temperature. The relationship is governed by: "
            "dT/dt = Q/(m×cp), where higher heat input Q raises temperature T. This fundamental physics principle underlies all heating systems.",
        
        "Room Temperature -> Heat Loss Rate":
            "According to Newton's Law of Cooling, the rate of heat transfer is proportional to the temperature difference between the "
            "room and outside. An increase in Room Temperature directly causes an increase in Heat Loss Rate because: Q = hA(T_room - T_outside). "
            "Higher room temperature means greater temperature differential, driving faster heat flow to the cooler exterior.",
        
        "Outside Temperature -> Heat Loss Rate":
            "Newton's Law of Cooling states that heat transfer rate depends on the temperature differential: Q = hA(T_room - T_outside). "
            "An increase in Outside Temperature directly causes a decrease in Heat Loss Rate because the temperature difference shrinks. "
            "Warmer outside conditions reduce the driving force for heat to escape from the building.",
        
        "Heat Loss Rate -> Room Temperature":
            "Based on the First Law of Thermodynamics, energy leaving a system reduces its internal energy. "
            "An increase in Heat Loss Rate directly causes a decrease in Room Temperature. As heat escapes through walls and windows, "
            "the room's thermal energy decreases, lowering its temperature. This is why buildings cool down when heating is off.",
        
        "Thermal Resistance -> Heat Loss Rate":
            "According to Fourier's Law of Heat Conduction, heat transfer rate is inversely proportional to thermal resistance: Q = ΔT/R. "
            "An increase in Thermal Resistance (better insulation, higher R-value) directly causes a decrease in Heat Loss Rate. "
            "This is why well-insulated buildings lose heat more slowly and require less energy to maintain temperature."
    }
    
    # Load edges into Neo4j
    logger.info("Loading edges...")
    for edge in edges_data:
        rel_type = edge["relationship"]  # POSITIVE or NEGATIVE
        edge_key = f"{edge['source']} -> {edge['target']}"
        
        # Get proper physics motivation
        motivation = PHYSICS_MOTIVATIONS.get(edge_key, f"Physics-based causal relationship between {edge['source']} and {edge['target']}")
        
        with discovery.graph_db._get_session() as session:
            session.run(f"""
                MATCH (s:variable {{name: $source, session_id: $session_id}})
                MATCH (t:variable {{name: $target, session_id: $session_id}})
                CREATE (s)-[r:{rel_type}]->(t)
                SET r.session_id = $session_id,
                    r.motivation = $motivation,
                    r.expert_validated = true,
                    r.physics_based = true
            """, {
                "source": edge["source"],
                "target": edge["target"],
                "session_id": session_id,
                "motivation": motivation
            })
    
    logger.info(f"✅ Loaded {len(vars_data)} variables and {len(edges_data)} edges")
    logger.info(f"   Session ID: {session_id}")
    
    return session_id, discovery


def fill_citations_from_hints(discovery: CausalDiscovery, citation_hints: dict):
    """
    Fill in citation URLs on edges using the citation hints.
    Uses Brave search to find URLs for each citation.
    """
    logger.info("\n" + "="*80)
    logger.info("FILLING CITATIONS FROM HINTS")
    logger.info("="*80)
    
    # Convert citation hints to the format expected by fill_in_missing_citations_with_hints
    # hint_map keys are (source, target) tuples, values are the citations list directly
    hint_map = {}
    for edge_key, citations in citation_hints.items():
        # Parse "Source -> Target" format
        parts = edge_key.split(' -> ')
        if len(parts) == 2:
            source, target = parts
            hint_map[(source, target)] = citations  # Pass citations directly (list of dicts)
    
    logger.info(f"Processing {len(hint_map)} edges with citation hints")
    
    # Use Brave search to find URLs for citations
    edges_updated = discovery.fill_in_missing_citations_with_hints(
        hint_map=hint_map,
        search_provider="brave",
        max_urls_per_citation=1,
        use_trusted_domains=False
    )
    
    logger.info(f"✅ Updated {edges_updated} edges with citation URLs")
    return edges_updated


def run_judge_on_physics_cld(discovery: CausalDiscovery, session_id: str, approach: str = "correctness", citation_hints: dict = None):
    """
    Run the judge on the physics CLD.
    
    Args:
        discovery: CausalDiscovery instance
        session_id: Neo4j session ID
        approach: 'correctness' (no citations needed) or 'citation' (requires citations)
        citation_hints: Citation hints dict for citation-based judging
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING JUDGE ON PHYSICS CLD")
    logger.info("="*80)
    logger.info(f"Approach: {approach}")
    
    import time
    start_time = time.time()
    
    try:
        if approach == "correctness":
            # Use correctness judging - evaluates causal relationships directly
            logger.info("Running correctness judging (evaluates causal relationships directly)...")
            
            judged_edges = discovery.judge_all_edges_serial(
                judge_models=["gpt-4.1"],
                num_judges=1
            )
        else:
            # Citation-based judging requires citations on edges
            # First, fill in citations from hints
            if citation_hints:
                fill_citations_from_hints(discovery, citation_hints)
            
            logger.info("Running citation-based judging (per_citation_aggregate approach)...")
            
            judged_edges = discovery.judge_all_edges_with_citations_serial(
                judge_models=["gpt-4.1"],
                num_judges=1,
                approach="per_citation_aggregate",
                judge_parallel=True,
                judge_max_workers=5,
                citation_fetcher="scraper",  # Use 'scraper' or 'jina'
                jina_timeout=60,
                jina_retries=2
            )
        
        duration = time.time() - start_time
        
        logger.info(f"✅ Judging complete in {duration:.1f}s")
        logger.info(f"   Judged {len(judged_edges)} edges")
        
        return judged_edges, duration
        
    except Exception as e:
        logger.error(f"❌ Judging failed: {e}")
        import traceback
        traceback.print_exc()
        return [], 0


def analyze_results(judged_edges: list, duration: float):
    """
    Analyze and display the judging results.
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS: PHYSICS CLD JUDGE RESULTS")
    logger.info("="*80)
    
    if not judged_edges:
        logger.warning("No edges were judged!")
        return
    
    # Collect verdicts (correctness judging uses OK/INCONSISTENT verdicts)
    verdicts = []
    for edge in judged_edges:
        verdict = edge.get("judge_verdict") or edge.get("verdict") or edge.get("aggregate_verdict", "unknown")
        verdicts.append(verdict)
    
    from collections import Counter
    verdict_counts = Counter(verdicts)
    
    # Calculate approval rate (OK = valid, INCONSISTENT = invalid)
    ok_count = verdict_counts.get("OK", 0)
    inconsistent_count = verdict_counts.get("INCONSISTENT", 0)
    total = len(verdicts)
    approval_rate = ok_count / total if total > 0 else 0
    
    # Display results
    logger.info(f"\n📊 RESULTS SUMMARY")
    logger.info(f"   Total edges judged: {len(judged_edges)}")
    logger.info(f"   Edges approved (OK): {ok_count}")
    logger.info(f"   Edges rejected (INCONSISTENT): {inconsistent_count}")
    logger.info(f"   Approval rate: {approval_rate:.1%}")
    logger.info(f"   Duration: {duration:.1f}s")
    
    logger.info(f"\n📋 VERDICT BREAKDOWN:")
    for verdict, count in verdict_counts.most_common():
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"   - {verdict}: {count} ({pct:.1f}%)")
    
    # Display individual edges
    logger.info(f"\n📝 INDIVIDUAL EDGE RESULTS:")
    for edge in judged_edges:
        source = edge.get("source", "?")
        target = edge.get("target", "?")
        verdict = edge.get("judge_verdict") or edge.get("verdict") or edge.get("aggregate_verdict", "N/A")
        reason = edge.get("judge_reason", "")[:50] + "..." if edge.get("judge_reason") else ""
        
        symbol = "✅" if verdict == "OK" else "❌"
        logger.info(f"   {symbol} {source} -> {target}: {verdict}")
        if reason and verdict != "OK":
            logger.info(f"      Reason: {reason}")
    
    # Expected vs actual
    logger.info(f"\n🎯 VALIDATION:")
    logger.info(f"   Expected: All 8 edges should be OK (physics principles)")
    logger.info(f"   Actual: {ok_count}/8 edges approved ({approval_rate:.1%})")
    
    if approval_rate >= 0.9:
        logger.info(f"   ✅ SUCCESS: Judge correctly validates physics-based edges")
    elif approval_rate >= 0.6:
        logger.info(f"   ⚠️  PARTIAL: Judge partially validates physics-based edges ({ok_count}/8)")
    else:
        logger.info(f"   ❌ ISSUE: Judge fails to validate well-established physics")
    
    return {
        "total_edges": len(judged_edges),
        "ok_count": ok_count,
        "inconsistent_count": inconsistent_count,
        "approval_rate": approval_rate,
        "verdict_counts": dict(verdict_counts),
        "duration_seconds": duration
    }


def save_results(session_id: str, judged_edges: list, metrics: dict):
    """Save results to JSON for later comparison."""
    output_dir = Path(__file__).parent / "results" / "physics_cld_judge_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"thermostat_judge_results_{timestamp}.json"
    
    results = {
        "session_id": session_id,
        "cld_name": "thermostat_heating_system",
        "timestamp": timestamp,
        "metrics": metrics,
        "edges": judged_edges
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run judge test on physics CLD")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judging, just load CLD")
    parser.add_argument("--approach", type=str, default="correctness", 
                       choices=["correctness", "citation"],
                       help="Judging approach: 'correctness' (no citations needed) or 'citation' (requires citations)")
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("PHYSICS CLD JUDGE VALIDATION TEST")
    logger.info("="*80)
    logger.info("Purpose: Validate judge works on well-established physics principles")
    logger.info("CLD: Thermostat Heating System (Newton's law of cooling, thermodynamics)")
    logger.info("")
    
    # Check required files exist
    for key, path in PHYSICS_CLD.items():
        if key != "name" and not Path(path).exists():
            logger.error(f"Missing file: {path}")
            return 1
    
    # Load CLD (pass approach to use correct yaml file)
    session_id, discovery = load_physics_cld_to_neo4j(approach=args.approach)
    
    if args.skip_judge:
        logger.info("\n⏭️  Skipping judging (--skip-judge flag)")
        logger.info(f"   Session ID for manual testing: {session_id}")
        return 0
    
    # Load citation hints if using citation approach
    citation_hints = None
    if args.approach == "citation":
        with open(PHYSICS_CLD["citation_hints"], 'r') as f:
            citation_hints = json.load(f)
        logger.info(f"Loaded citation hints for {len(citation_hints)} edges")
    
    # Run judge
    judged_edges, duration = run_judge_on_physics_cld(discovery, session_id, approach=args.approach, citation_hints=citation_hints)
    
    # Analyze results
    metrics = analyze_results(judged_edges, duration)
    
    # Save results
    if metrics:
        save_results(session_id, judged_edges, metrics)
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"\nNext step: Compare with Alzheimer's CLD results to validate hypothesis")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

