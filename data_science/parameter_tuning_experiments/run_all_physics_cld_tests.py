#!/usr/bin/env python3
"""
Run Judge Tests on Multiple Physics CLDs

Tests the judge on physics-based CLDs derived from published equations:
1. Thermostat Heating System (Newton's Law of Cooling, Thermodynamics)
2. Predator-Prey (Lotka-Volterra Equations)
3. RC Circuit (Ohm's Law, Kirchhoff's Laws)
4. Water Tank (Conservation of Mass, Torricelli's Theorem)

Each CLD's edges are derived from and cited to original physics sources.

Usage:
    python run_all_physics_cld_tests.py [--approach correctness|citation]
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging
from collections import Counter
from typing import Dict, List, Tuple

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

from modules import CausalDiscovery

# Base directory for CLDs
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
RESULTS_DIR = Path(__file__).parent / "results" / "physics_cld_judge_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define all physics CLDs
PHYSICS_CLDS = {
    "thermostat_heating_system": {
        "name": "Thermostat Heating System",
        "physics_basis": "Newton's Law of Cooling (1701), First Law of Thermodynamics, Feedback Control Theory",
        "key_equations": ["Q = hA(T_room - T_outside)", "dU = Q - W", "Gap = T_desired - T_actual"],
        "vars_json": BASE_DIR / "thermostat_heating_system_vars_data.json",
        "edges_json": BASE_DIR / "thermostat_heating_system_edges_data.json",
        "context_json": BASE_DIR / "thermostat_heating_system_context_data.json",
        "citation_hints": BASE_DIR / "thermostat_heating_system_citation_hints.json"
    },
    "predator_prey": {
        "name": "Predator-Prey (Lotka-Volterra)",
        "physics_basis": "Lotka (1925), Volterra (1926) - Population Dynamics Equations",
        "key_equations": ["dR/dt = αR - βRP", "dP/dt = δRP - γP"],
        "vars_json": BASE_DIR / "predator_prey_vars_data.json",
        "edges_json": BASE_DIR / "predator_prey_edges_data.json",
        "context_json": BASE_DIR / "predator_prey_context_data.json",
        "citation_hints": BASE_DIR / "predator_prey_citation_hints.json"
    },
    "rc_circuit": {
        "name": "RC Circuit (Capacitor Charging)",
        "physics_basis": "Ohm's Law (1827), Kirchhoff's Voltage Law (1845), Capacitor Equation",
        "key_equations": ["V = IR", "ΣV = 0", "Q = CV", "I = dQ/dt"],
        "vars_json": BASE_DIR / "rc_circuit_vars_data.json",
        "edges_json": BASE_DIR / "rc_circuit_edges_data.json",
        "context_json": BASE_DIR / "rc_circuit_context_data.json",
        "citation_hints": BASE_DIR / "rc_circuit_citation_hints.json"
    },
    "water_tank": {
        "name": "Water Tank (Stock-Flow)",
        "physics_basis": "Conservation of Mass, Pascal's Law (1663), Torricelli's Theorem (1643)",
        "key_equations": ["dV/dt = Qin - Qout", "P = ρgh", "v = √(2gh)"],
        "vars_json": BASE_DIR / "water_tank_vars_data.json",
        "edges_json": BASE_DIR / "water_tank_edges_data.json",
        "context_json": BASE_DIR / "water_tank_context_data.json",
        "citation_hints": BASE_DIR / "water_tank_citation_hints.json"
    }
}


def check_cld_files(cld_key: str) -> bool:
    """Check if all required files exist for a CLD."""
    cld = PHYSICS_CLDS[cld_key]
    required_files = ["vars_json", "edges_json", "context_json", "citation_hints"]
    
    for file_key in required_files:
        if not cld[file_key].exists():
            logger.warning(f"Missing file for {cld_key}: {cld[file_key]}")
            return False
    return True


def load_cld_to_neo4j(cld_key: str, approach: str = "correctness") -> Tuple[str, CausalDiscovery]:
    """
    Load a physics CLD into Neo4j.
    
    Returns:
        Tuple of (session_id, discovery_instance)
    """
    cld = PHYSICS_CLDS[cld_key]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading: {cld['name']}")
    logger.info(f"Physics basis: {cld['physics_basis']}")
    logger.info(f"{'='*60}")
    
    # Load JSON data
    with open(cld["vars_json"], 'r') as f:
        vars_data = json.load(f)
    
    with open(cld["edges_json"], 'r') as f:
        edges_data = json.load(f)
    
    with open(cld["context_json"], 'r') as f:
        context_data = json.load(f)
    
    logger.info(f"Variables: {len(vars_data)}, Edges: {len(edges_data)}")
    
    # Select yaml file based on approach
    if approach == "citation":
        yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_citation_baseline.yaml"
    else:
        yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_correctness_baseline.yaml"
    
    # Create CausalDiscovery instance
    discovery = CausalDiscovery(
        target_variable=context_data.get("Target", "Unknown"),
        temporal_scale=context_data.get("Temporal Scale", "Unknown"),
        spatial_scale=context_data.get("Spatial Scale", "Unknown"),
        yaml_path=str(yaml_path),
        generator_config={"provider": "openai", "model": "gpt-4.1"},
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.0}
    )
    
    session_id = discovery.session_id
    
    # Load variables into Neo4j
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
    
    # Load edges into Neo4j
    for edge in edges_data:
        rel_type = edge["type"].upper()
        motivation = edge.get("motivation", "Physics-based relationship.")
        
        with discovery.graph_db._get_session() as session:
            session.run(f"""
                MATCH (s:variable {{name: $source, session_id: $session_id}})
                MATCH (t:variable {{name: $target, session_id: $session_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.session_id = $session_id,
                    r.motivation = $motivation,
                    r.expert_validated = true
            """, {
                "source": edge["source"],
                "target": edge["target"],
                "session_id": session_id,
                "motivation": motivation
            })
    
    logger.info(f"✅ Loaded into Neo4j: session_id={session_id}")
    return session_id, discovery


def fill_citations_from_hints(discovery: CausalDiscovery, citation_hints_path: Path):
    """
    Fill in citation URLs on edges using the citation hints.
    Uses Brave search to find URLs for each citation.
    """
    logger.info("Filling citations from hints...")
    
    with open(citation_hints_path, 'r') as f:
        citation_hints = json.load(f)
    
    # Convert citation hints to the format expected by fill_in_missing_citations_with_hints
    hint_map = {}
    for edge_key, citations in citation_hints.items():
        parts = edge_key.split(' -> ')
        if len(parts) == 2:
            source, target = parts
            hint_map[(source, target)] = citations
    
    if not hint_map:
        logger.warning("No citation hints found")
        return 0
    
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


def run_judge_test(discovery: CausalDiscovery, approach: str = "correctness", 
                   citation_hints_path: Path = None) -> Tuple[List[Dict], float]:
    """
    Run the judge on a CLD.
    
    Args:
        approach: "correctness" uses judgeCorrectness prompt (CORRECT/PARTIALLY_CORRECT/INCORRECT)
                  "citation" uses per_citation_aggregate approach (Fully/Partially/Not supported)
        citation_hints_path: Path to citation hints file (needed for citation approach)
    
    Returns:
        Tuple of (judged_edges, duration_seconds)
    """
    import time
    start_time = time.time()
    
    if approach == "correctness":
        # Use judge_all_edges_with_citations_serial with approach="correctness"
        # This uses the judgeCorrectness prompt which outputs CORRECT/PARTIALLY_CORRECT/INCORRECT
        judged_edges = discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"],
            num_judges=1,
            approach="correctness",  # This triggers judgeCorrectness prompt
            judge_parallel=True,
            judge_max_workers=5,
            citation_fetcher="scraper",
            jina_timeout=60,
            jina_retries=2
        )
    else:
        # Citation-based judging (per_citation_aggregate)
        # First, fill in citations from hints
        if citation_hints_path and citation_hints_path.exists():
            fill_citations_from_hints(discovery, citation_hints_path)
        
        judged_edges = discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"],
            num_judges=1,
            approach="per_citation_aggregate",
            judge_parallel=True,
            judge_max_workers=5,
            citation_fetcher="scraper",
            jina_timeout=60,
            jina_retries=2
        )
    
    duration = time.time() - start_time
    return judged_edges, duration


def analyze_results(cld_key: str, judged_edges: List[Dict], duration: float, approach: str) -> Dict:
    """Analyze judge results for a CLD."""
    cld = PHYSICS_CLDS[cld_key]
    
    # Extract verdicts - try multiple possible keys
    verdicts = []
    for e in judged_edges:
        verdict = (e.get("judge_verdict") or 
                   e.get("verdict") or 
                   e.get("aggregate_verdict") or 
                   "N/A")
        verdicts.append(verdict)
    
    verdict_counts = Counter(verdicts)
    total = len(verdicts)
    
    # Calculate approval rate based on approach
    if approach == "correctness":
        # judgeCorrectness outputs: CORRECT, PARTIALLY_CORRECT, INCORRECT
        correct = verdict_counts.get("CORRECT", 0)
        partial = verdict_counts.get("PARTIALLY_CORRECT", 0)
        approval_rate = (correct + 0.5 * partial) / total if total > 0 else 0
    else:
        # Citation-based outputs: Fully supported, Partially supported, Not supported
        fully = verdict_counts.get("Fully supported", 0)
        partially = verdict_counts.get("Partially supported", 0)
        approval_rate = (fully + 0.5 * partially) / total if total > 0 else 0
    
    return {
        "cld_key": cld_key,
        "cld_name": cld["name"],
        "physics_basis": cld["physics_basis"],
        "total_edges": total,
        "verdict_counts": dict(verdict_counts),
        "approval_rate": approval_rate,
        "duration_seconds": duration,
        "approach": approach,
        "edges": judged_edges
    }


def print_results_summary(all_results: List[Dict]):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("PHYSICS CLD JUDGE TEST - SUMMARY")
    print(f"{'='*80}")
    
    approach = all_results[0]["approach"] if all_results else "unknown"
    print(f"Approach: {approach}")
    print(f"Judge Model: gpt-4.1")
    
    if approach == "correctness":
        print("Expected verdicts: CORRECT, PARTIALLY_CORRECT, INCORRECT")
    else:
        print("Expected verdicts: Fully supported, Partially supported, Not supported")
    print()
    
    # Table header
    print(f"{'CLD Name':<35} {'Edges':>6} {'Approval':>10} {'Time':>8}")
    print("-" * 65)
    
    total_edges = 0
    total_approved = 0
    
    for result in all_results:
        name = result["cld_name"][:33]
        edges = result["total_edges"]
        approval = result["approval_rate"]
        duration = result["duration_seconds"]
        
        total_edges += edges
        total_approved += edges * approval
        
        status = "✅" if approval >= 0.8 else "⚠️" if approval >= 0.5 else "❌"
        print(f"{status} {name:<33} {edges:>6} {approval:>9.1%} {duration:>7.1f}s")
    
    print("-" * 65)
    overall_rate = total_approved / total_edges if total_edges > 0 else 0
    print(f"{'TOTAL':<35} {total_edges:>6} {overall_rate:>9.1%}")
    
    # Detailed breakdown per CLD
    print(f"\n{'='*80}")
    print("DETAILED VERDICT BREAKDOWN")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n📊 {result['cld_name']}")
        print(f"   Physics: {result['physics_basis'][:60]}...")
        for verdict, count in sorted(result["verdict_counts"].items(), key=lambda x: -x[1]):
            pct = count / result["total_edges"] * 100 if result["total_edges"] > 0 else 0
            # Add emoji based on verdict
            if verdict in ["CORRECT", "Fully supported"]:
                emoji = "✅"
            elif verdict in ["PARTIALLY_CORRECT", "Partially supported"]:
                emoji = "⚠️"
            else:
                emoji = "❌"
            print(f"   {emoji} {verdict}: {count} ({pct:.1f}%)")


def save_all_results(all_results: List[Dict], approach: str):
    """Save all results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"all_physics_clds_{approach}_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "approach": approach,
        "judge_model": "gpt-4.1",
        "total_clds": len(all_results),
        "results": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run judge tests on all physics CLDs")
    parser.add_argument("--approach", type=str, default="correctness",
                       choices=["correctness", "citation"],
                       help="Judging approach")
    parser.add_argument("--cld", type=str, default=None,
                       choices=list(PHYSICS_CLDS.keys()),
                       help="Test only a specific CLD (default: all)")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("PHYSICS CLD JUDGE VALIDATION TEST")
    print(f"{'='*80}")
    print(f"Testing judge on physics-based CLDs with {args.approach} approach")
    print(f"Each CLD derived from published physics equations")
    print()
    
    # Determine which CLDs to test
    if args.cld:
        clds_to_test = [args.cld]
    else:
        clds_to_test = list(PHYSICS_CLDS.keys())
    
    # Check which CLDs have all required files
    valid_clds = []
    for cld_key in clds_to_test:
        if check_cld_files(cld_key):
            valid_clds.append(cld_key)
        else:
            logger.warning(f"Skipping {cld_key} - missing files")
    
    if not valid_clds:
        logger.error("No valid CLDs found!")
        return 1
    
    print(f"CLDs to test: {', '.join(valid_clds)}")
    print()
    
    # Run tests
    all_results = []
    
    for cld_key in valid_clds:
        try:
            # Load CLD
            session_id, discovery = load_cld_to_neo4j(cld_key, approach=args.approach)
            
            # Run judge
            logger.info(f"Running {args.approach} judge...")
            citation_hints_path = PHYSICS_CLDS[cld_key]["citation_hints"] if args.approach == "citation" else None
            judged_edges, duration = run_judge_test(discovery, approach=args.approach, citation_hints_path=citation_hints_path)
            
            # Analyze results
            result = analyze_results(cld_key, judged_edges, duration, args.approach)
            result["session_id"] = session_id
            all_results.append(result)
            
            logger.info(f"✅ {cld_key}: {result['approval_rate']:.1%} approval ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"❌ Error testing {cld_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_results:
        print_results_summary(all_results)
        save_all_results(all_results, args.approach)
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

