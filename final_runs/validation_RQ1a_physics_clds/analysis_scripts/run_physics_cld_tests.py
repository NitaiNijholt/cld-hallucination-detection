#!/usr/bin/env python3
"""
Run Judge Tests on Physics CLDs

Runs correctness and citation-based judging on physics CLDs and saves raw results to JSON.
Results can then be analyzed separately with analyze_physics_cld_results.py

Usage:
    python run_physics_cld_tests.py                    # Run both approaches
    python run_physics_cld_tests.py --approach correctness  # Correctness only
    python run_physics_cld_tests.py --approach citation     # Citation only
    python run_physics_cld_tests.py --skip-citation         # Skip citation (faster)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev", override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import CausalDiscovery

# Paths
BASE_DIR = Path(__file__).parent / "ground_truth_clds_for_experiments"
RESULTS_DIR = Path(__file__).parent / "results" / "physics_cld_batch_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Physics CLD definitions
PHYSICS_CLDS = {
    "thermostat_heating_system": {
        "name": "Thermostat Heating System",
        "short_name": "Thermostat",
        "physics_basis": "Newton's Law of Cooling (1701), First Law of Thermodynamics",
        "primary_sources": ["Newton (1701)", "Sterman (2000)", "Incropera et al. (2007)"],
        "vars_json": BASE_DIR / "thermostat_heating_system_vars_data.json",
        "edges_json": BASE_DIR / "thermostat_heating_system_edges_data.json",
        "context_json": BASE_DIR / "thermostat_heating_system_context_data.json",
        "citation_hints": BASE_DIR / "thermostat_heating_system_citation_hints.json"
    },
    "predator_prey": {
        "name": "Predator-Prey (Lotka-Volterra)",
        "short_name": "Predator-Prey",
        "physics_basis": "Lotka (1925), Volterra (1926) - Population Dynamics",
        "primary_sources": ["Lotka (1925)", "Volterra (1926)", "Murray (2002)"],
        "vars_json": BASE_DIR / "predator_prey_vars_data.json",
        "edges_json": BASE_DIR / "predator_prey_edges_data.json",
        "context_json": BASE_DIR / "predator_prey_context_data.json",
        "citation_hints": BASE_DIR / "predator_prey_citation_hints.json"
    },
    "rc_circuit": {
        "name": "RC Circuit (Capacitor Charging)",
        "short_name": "RC Circuit",
        "physics_basis": "Ohm's Law (1827), Kirchhoff's Laws (1845)",
        "primary_sources": ["Ohm (1827)", "Kirchhoff (1845)", "Nilsson & Riedel (2015)"],
        "vars_json": BASE_DIR / "rc_circuit_vars_data.json",
        "edges_json": BASE_DIR / "rc_circuit_edges_data.json",
        "context_json": BASE_DIR / "rc_circuit_context_data.json",
        "citation_hints": BASE_DIR / "rc_circuit_citation_hints.json"
    },
    "water_tank": {
        "name": "Water Tank (Stock-Flow)",
        "short_name": "Water Tank",
        "physics_basis": "Conservation of Mass, Pascal (1663), Torricelli (1643)",
        "primary_sources": ["Pascal (1663)", "Torricelli (1643)", "Bernoulli (1738)"],
        "vars_json": BASE_DIR / "water_tank_vars_data.json",
        "edges_json": BASE_DIR / "water_tank_edges_data.json",
        "context_json": BASE_DIR / "water_tank_context_data.json",
        "citation_hints": BASE_DIR / "water_tank_citation_hints.json"
    }
}


def check_files(cld_key: str) -> bool:
    """Check if all required files exist."""
    cld = PHYSICS_CLDS[cld_key]
    for key in ["vars_json", "edges_json", "context_json", "citation_hints"]:
        if not cld[key].exists():
            return False
    return True


def load_cld_to_neo4j(cld_key: str, approach: str) -> Tuple[str, CausalDiscovery]:
    """Load CLD into Neo4j and return discovery instance."""
    cld = PHYSICS_CLDS[cld_key]
    
    with open(cld["vars_json"]) as f:
        vars_data = json.load(f)
    with open(cld["edges_json"]) as f:
        edges_data = json.load(f)
    with open(cld["context_json"]) as f:
        context_data = json.load(f)
    
    yaml_file = "prompts_citation_baseline.yaml" if approach == "citation" else "prompts_correctness_baseline.yaml"
    yaml_path = Path(__file__).parent / "alternative_prompts" / yaml_file
    
    discovery = CausalDiscovery(
        target_variable=context_data.get("Target", "Unknown"),
        temporal_scale=context_data.get("Temporal Scale", "Unknown"),
        spatial_scale=context_data.get("Spatial Scale", "Unknown"),
        yaml_path=str(yaml_path),
        generator_config={"provider": "openai", "model": "gpt-4.1"},
        judge_config={"provider": "openai", "model": "gpt-4.1", "temperature": 0.0}
    )
    
    session_id = discovery.session_id
    
    # Load variables
    for var in vars_data:
        with discovery.graph_db._get_session() as session:
            session.run("""
                CREATE (n:variable {name: $name, description: $definition,
                    session_id: $session_id, target: false, deleted: false})
            """, {"name": var["name"], "definition": var.get("definition", ""), "session_id": session_id})
        discovery.variables.append(var["name"])
    
    # Load edges
    for edge in edges_data:
        rel_type = edge["type"].upper()
        motivation = edge.get("motivation", "")
        with discovery.graph_db._get_session() as session:
            session.run(f"""
                MATCH (s:variable {{name: $source, session_id: $session_id}})
                MATCH (t:variable {{name: $target, session_id: $session_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.session_id = $session_id, r.motivation = $motivation, r.expert_validated = true
            """, {"source": edge["source"], "target": edge["target"], "session_id": session_id, "motivation": motivation})
    
    return session_id, discovery


def fill_citations(discovery: CausalDiscovery, hints_path: Path) -> int:
    """Fill citation URLs from hints."""
    with open(hints_path) as f:
        hints = json.load(f)
    
    hint_map = {}
    for key, citations in hints.items():
        parts = key.split(' -> ')
        if len(parts) == 2:
            hint_map[(parts[0], parts[1])] = citations
    
    if not hint_map:
        return 0
    
    return discovery.fill_in_missing_citations_with_hints(
        hint_map=hint_map, search_provider="brave",
        max_urls_per_citation=1, use_trusted_domains=False
    )


def run_judge(discovery: CausalDiscovery, approach: str, hints_path: Path = None) -> List[Dict]:
    """Run judge and return results."""
    if approach == "correctness":
        return discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"], num_judges=1, approach="correctness",
            judge_parallel=True, judge_max_workers=5, citation_fetcher="scraper"
        )
    else:
        if hints_path and hints_path.exists():
            fill_citations(discovery, hints_path)
        return discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"], num_judges=1, approach="per_citation_aggregate",
            judge_parallel=True, judge_max_workers=5, citation_fetcher="scraper"
        )


def run_all_tests(approaches: List[str]) -> Dict:
    """Run all tests and return results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": "gpt-4.1",
        "cld_metadata": {},
        "correctness_results": [],
        "citation_results": []
    }
    
    # Get valid CLDs
    valid_clds = [k for k in PHYSICS_CLDS if check_files(k)]
    
    # Store CLD metadata
    for cld_key in valid_clds:
        cld = PHYSICS_CLDS[cld_key]
        with open(cld["vars_json"]) as f:
            vars_data = json.load(f)
        with open(cld["edges_json"]) as f:
            edges_data = json.load(f)
        
        results["cld_metadata"][cld_key] = {
            "name": cld["name"],
            "short_name": cld["short_name"],
            "physics_basis": cld["physics_basis"],
            "primary_sources": cld["primary_sources"],
            "num_variables": len(vars_data),
            "num_edges": len(edges_data),
            "variables": [v["name"] for v in vars_data],
            "edges": [{"source": e["source"], "target": e["target"], "type": e["type"]} for e in edges_data]
        }
    
    # Run correctness tests
    if "correctness" in approaches:
        print("\n" + "="*60)
        print("RUNNING CORRECTNESS JUDGING")
        print("="*60)
        
        for cld_key in valid_clds:
            print(f"\n>>> {PHYSICS_CLDS[cld_key]['name']}...")
            try:
                import time
                start = time.time()
                session_id, discovery = load_cld_to_neo4j(cld_key, "correctness")
                judged = run_judge(discovery, "correctness")
                duration = time.time() - start
                
                results["correctness_results"].append({
                    "cld_key": cld_key,
                    "session_id": session_id,
                    "duration_seconds": duration,
                    "edges": judged
                })
                print(f"    ✅ Done in {duration:.1f}s")
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    # Run citation tests
    if "citation" in approaches:
        print("\n" + "="*60)
        print("RUNNING CITATION JUDGING")
        print("="*60)
        
        for cld_key in valid_clds:
            print(f"\n>>> {PHYSICS_CLDS[cld_key]['name']}...")
            try:
                import time
                start = time.time()
                session_id, discovery = load_cld_to_neo4j(cld_key, "citation")
                hints_path = PHYSICS_CLDS[cld_key]["citation_hints"]
                judged = run_judge(discovery, "citation", hints_path)
                duration = time.time() - start
                
                results["citation_results"].append({
                    "cld_key": cld_key,
                    "session_id": session_id,
                    "duration_seconds": duration,
                    "edges": judged
                })
                print(f"    ✅ Done in {duration:.1f}s")
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Physics CLD Judge Tests")
    parser.add_argument("--approach", choices=["correctness", "citation", "both"], default="both")
    parser.add_argument("--skip-citation", action="store_true", help="Skip citation judging")
    args = parser.parse_args()
    
    # Determine which approaches to run
    if args.skip_citation:
        approaches = ["correctness"]
    elif args.approach == "both":
        approaches = ["correctness", "citation"]
    else:
        approaches = [args.approach]
    
    print("\n" + "="*60)
    print("PHYSICS CLD JUDGE TESTS")
    print("="*60)
    print(f"Approaches: {approaches}")
    print(f"Judge Model: gpt-4.1")
    
    # Run tests
    results = run_all_tests(approaches)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"physics_cld_raw_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"📁 {output_file}")
    print(f"\nRun analyze_physics_cld_results.py to generate tables.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())









