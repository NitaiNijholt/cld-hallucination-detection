#!/usr/bin/env python3
"""
Physics CLD Batch Analysis Script

Runs comprehensive judge validation on physics-based CLDs derived from published equations.
Produces:
1. Summary statistics for correctness and citation judging
2. LaTeX tables for thesis
3. Detailed JSON results
4. CLD structure overview

Physics CLDs included:
1. Thermostat Heating System - Newton's Law of Cooling, Thermodynamics
2. Predator-Prey - Lotka-Volterra Equations (1925-1926)
3. RC Circuit - Ohm's Law, Kirchhoff's Laws
4. Water Tank - Conservation of Mass, Torricelli's Theorem

Usage:
    python run_physics_cld_batch_analysis.py
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
from typing import Dict, List, Tuple, Optional

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
RESULTS_DIR = Path(__file__).parent / "results" / "physics_cld_batch_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define all physics CLDs with full metadata
PHYSICS_CLDS = {
    "thermostat_heating_system": {
        "name": "Thermostat Heating System",
        "short_name": "Thermostat",
        "physics_basis": "Newton's Law of Cooling (1701), First Law of Thermodynamics, Feedback Control Theory",
        "key_equations": [
            r"$Q = hA(T_{room} - T_{outside})$ (Newton's Cooling)",
            r"$dU = Q_{in} - Q_{out}$ (Energy Balance)",
            r"$Gap = T_{desired} - T_{actual}$ (Control Error)"
        ],
        "primary_sources": [
            "Newton (1701) - Scala graduum Caloris",
            "Sterman (2000) - Business Dynamics Ch. 5-6",
            "Incropera et al. (2007) - Heat and Mass Transfer"
        ],
        "vars_json": BASE_DIR / "thermostat_heating_system_vars_data.json",
        "edges_json": BASE_DIR / "thermostat_heating_system_edges_data.json",
        "context_json": BASE_DIR / "thermostat_heating_system_context_data.json",
        "citation_hints": BASE_DIR / "thermostat_heating_system_citation_hints.json"
    },
    "predator_prey": {
        "name": "Predator-Prey (Lotka-Volterra)",
        "short_name": "Predator-Prey",
        "physics_basis": "Lotka (1925), Volterra (1926) - Population Dynamics Equations",
        "key_equations": [
            r"$\frac{dR}{dt} = \alpha R - \beta RP$ (Prey dynamics)",
            r"$\frac{dP}{dt} = \delta RP - \gamma P$ (Predator dynamics)"
        ],
        "primary_sources": [
            "Lotka (1925) - Elements of Physical Biology",
            "Volterra (1926) - Fluctuations in Abundance",
            "Murray (2002) - Mathematical Biology"
        ],
        "vars_json": BASE_DIR / "predator_prey_vars_data.json",
        "edges_json": BASE_DIR / "predator_prey_edges_data.json",
        "context_json": BASE_DIR / "predator_prey_context_data.json",
        "citation_hints": BASE_DIR / "predator_prey_citation_hints.json"
    },
    "rc_circuit": {
        "name": "RC Circuit (Capacitor Charging)",
        "short_name": "RC Circuit",
        "physics_basis": "Ohm's Law (1827), Kirchhoff's Voltage Law (1845), Capacitor Equation",
        "key_equations": [
            r"$V = IR$ (Ohm's Law)",
            r"$\sum V = 0$ (Kirchhoff's Voltage Law)",
            r"$Q = CV$, $I = \frac{dQ}{dt}$ (Capacitor)"
        ],
        "primary_sources": [
            "Ohm (1827) - Die galvanische Kette",
            "Kirchhoff (1845) - Circuit Laws",
            "Nilsson & Riedel (2015) - Electric Circuits"
        ],
        "vars_json": BASE_DIR / "rc_circuit_vars_data.json",
        "edges_json": BASE_DIR / "rc_circuit_edges_data.json",
        "context_json": BASE_DIR / "rc_circuit_context_data.json",
        "citation_hints": BASE_DIR / "rc_circuit_citation_hints.json"
    },
    "water_tank": {
        "name": "Water Tank (Stock-Flow)",
        "short_name": "Water Tank",
        "physics_basis": "Conservation of Mass, Pascal's Law (1663), Torricelli's Theorem (1643)",
        "key_equations": [
            r"$\frac{dV}{dt} = Q_{in} - Q_{out}$ (Mass Conservation)",
            r"$P = \rho g h$ (Pascal's Law)",
            r"$v = \sqrt{2gh}$ (Torricelli)"
        ],
        "primary_sources": [
            "Pascal (1663) - Traité de l'équilibre",
            "Torricelli (1643) - De Motu Gravium",
            "Bernoulli (1738) - Hydrodynamica"
        ],
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


def load_cld_structure(cld_key: str) -> Dict:
    """Load CLD structure (variables and edges) for display."""
    cld = PHYSICS_CLDS[cld_key]
    
    with open(cld["vars_json"], 'r') as f:
        variables = json.load(f)
    
    with open(cld["edges_json"], 'r') as f:
        edges = json.load(f)
    
    with open(cld["context_json"], 'r') as f:
        context = json.load(f)
    
    return {
        "name": cld["name"],
        "short_name": cld["short_name"],
        "physics_basis": cld["physics_basis"],
        "key_equations": cld["key_equations"],
        "primary_sources": cld["primary_sources"],
        "variables": variables,
        "edges": edges,
        "context": context,
        "num_variables": len(variables),
        "num_edges": len(edges)
    }


def print_cld_overview():
    """Print an overview of all physics CLDs."""
    print("\n" + "="*80)
    print("PHYSICS CLD OVERVIEW")
    print("="*80)
    
    for cld_key in PHYSICS_CLDS:
        if not check_cld_files(cld_key):
            continue
        
        structure = load_cld_structure(cld_key)
        
        print(f"\n{'─'*80}")
        print(f"📊 {structure['name']}")
        print(f"{'─'*80}")
        print(f"Physics Basis: {structure['physics_basis']}")
        print(f"Variables: {structure['num_variables']}, Edges: {structure['num_edges']}")
        
        print("\nKey Equations:")
        for eq in structure['key_equations']:
            print(f"  • {eq}")
        
        print("\nVariables:")
        for var in structure['variables']:
            print(f"  • {var['name']}")
        
        print("\nEdges:")
        for edge in structure['edges']:
            polarity = "+" if edge['type'] == "POSITIVE" else "−"
            print(f"  {edge['source']} ──({polarity})──> {edge['target']}")


def load_cld_to_neo4j(cld_key: str, approach: str = "correctness") -> Tuple[str, CausalDiscovery]:
    """Load a physics CLD into Neo4j."""
    cld = PHYSICS_CLDS[cld_key]
    
    with open(cld["vars_json"], 'r') as f:
        vars_data = json.load(f)
    
    with open(cld["edges_json"], 'r') as f:
        edges_data = json.load(f)
    
    with open(cld["context_json"], 'r') as f:
        context_data = json.load(f)
    
    # Select yaml file based on approach
    if approach == "citation":
        yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_citation_baseline.yaml"
    else:
        yaml_path = Path(__file__).parent / "alternative_prompts" / "prompts_correctness_baseline.yaml"
    
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
                CREATE (n:variable {
                    name: $name, description: $definition,
                    session_id: $session_id, target: false, deleted: false
                })
            """, {"name": var["name"], "definition": var.get("definition", ""), "session_id": session_id})
        discovery.variables.append(var["name"])
    
    # Load edges
    for edge in edges_data:
        rel_type = edge["type"].upper()
        motivation = edge.get("motivation", "Physics-based relationship.")
        
        with discovery.graph_db._get_session() as session:
            session.run(f"""
                MATCH (s:variable {{name: $source, session_id: $session_id}})
                MATCH (t:variable {{name: $target, session_id: $session_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.session_id = $session_id, r.motivation = $motivation, r.expert_validated = true
            """, {"source": edge["source"], "target": edge["target"], "session_id": session_id, "motivation": motivation})
    
    return session_id, discovery


def fill_citations_from_hints(discovery: CausalDiscovery, citation_hints_path: Path) -> int:
    """Fill in citation URLs on edges using hints."""
    with open(citation_hints_path, 'r') as f:
        citation_hints = json.load(f)
    
    hint_map = {}
    for edge_key, citations in citation_hints.items():
        parts = edge_key.split(' -> ')
        if len(parts) == 2:
            hint_map[(parts[0], parts[1])] = citations
    
    if not hint_map:
        return 0
    
    edges_updated = discovery.fill_in_missing_citations_with_hints(
        hint_map=hint_map, search_provider="brave",
        max_urls_per_citation=1, use_trusted_domains=False
    )
    return edges_updated


def run_judge_test(discovery: CausalDiscovery, approach: str, 
                   citation_hints_path: Path = None) -> Tuple[List[Dict], float]:
    """Run the judge on a CLD."""
    import time
    start_time = time.time()
    
    if approach == "correctness":
        judged_edges = discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"], num_judges=1, approach="correctness",
            judge_parallel=True, judge_max_workers=5,
            citation_fetcher="scraper", jina_timeout=60, jina_retries=2
        )
    else:
        if citation_hints_path and citation_hints_path.exists():
            fill_citations_from_hints(discovery, citation_hints_path)
        
        judged_edges = discovery.judge_all_edges_with_citations_serial(
            judge_models=["gpt-4.1"], num_judges=1, approach="per_citation_aggregate",
            judge_parallel=True, judge_max_workers=5,
            citation_fetcher="scraper", jina_timeout=60, jina_retries=2
        )
    
    duration = time.time() - start_time
    return judged_edges, duration


def analyze_results(cld_key: str, judged_edges: List[Dict], duration: float, approach: str) -> Dict:
    """Analyze judge results for a CLD."""
    cld = PHYSICS_CLDS[cld_key]
    
    verdicts = []
    for e in judged_edges:
        verdict = e.get("judge_verdict") or e.get("verdict") or e.get("aggregate_verdict") or "N/A"
        verdicts.append(verdict)
    
    verdict_counts = Counter(verdicts)
    total = len(verdicts)
    
    if approach == "correctness":
        correct = verdict_counts.get("CORRECT", 0)
        partial = verdict_counts.get("PARTIALLY_CORRECT", 0)
        approval_rate = (correct + 0.5 * partial) / total if total > 0 else 0
    else:
        fully = verdict_counts.get("Fully supported", 0)
        partially = verdict_counts.get("Partially supported", 0)
        approval_rate = (fully + 0.5 * partially) / total if total > 0 else 0
    
    return {
        "cld_key": cld_key,
        "cld_name": cld["name"],
        "short_name": cld["short_name"],
        "physics_basis": cld["physics_basis"],
        "total_edges": total,
        "verdict_counts": dict(verdict_counts),
        "approval_rate": approval_rate,
        "duration_seconds": duration,
        "approach": approach,
        "edges": judged_edges
    }


def generate_latex_tables(correctness_results: List[Dict], citation_results: List[Dict]) -> str:
    """Generate LaTeX tables for the thesis."""
    
    latex = []
    
    # Table 1: Physics CLDs Overview with References
    latex.append(r"""
% Table: Physics CLDs Overview
\begin{table}[htbp]
\centering
\caption{Physics-Based CLDs Used for Judge Validation}
\label{tab:physics-clds-overview}
\begin{tabular}{@{}llccp{5cm}@{}}
\toprule
\textbf{CLD Name} & \textbf{Physics Basis} & \textbf{Vars} & \textbf{Edges} & \textbf{Primary References} \\
\midrule""")
    
    for cld_key in PHYSICS_CLDS:
        if check_cld_files(cld_key):
            structure = load_cld_structure(cld_key)
            cld_info = PHYSICS_CLDS[cld_key]
            # Escape underscores and truncate physics basis
            physics_short = structure['physics_basis'].split(',')[0].replace('_', r'\_')
            name = structure['short_name'].replace('_', r'\_')
            # Get first 2 references, escape special chars
            refs = cld_info['primary_sources'][:2]
            refs_str = "; ".join(refs).replace('_', r'\_').replace('&', r'\&')
            latex.append(f"{name} & {physics_short} & {structure['num_variables']} & {structure['num_edges']} & {refs_str} \\\\")
    
    latex.append(r"""\midrule
\textbf{Total} & & & \textbf{""" + str(sum(load_cld_structure(k)['num_edges'] for k in PHYSICS_CLDS if check_cld_files(k))) + r"""} & \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 2: Correctness Judging Results
    latex.append(r"""
% Table: Correctness Judging Results
\begin{table}[htbp]
\centering
\caption{Correctness-Based Judge Validation Results on Physics CLDs}
\label{tab:physics-correctness-results}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{CLD} & \textbf{Edges} & \textbf{CORRECT} & \textbf{PARTIAL} & \textbf{INCORRECT} & \textbf{Approval} \\
\midrule""")
    
    total_edges = 0
    total_correct = 0
    total_partial = 0
    total_incorrect = 0
    
    for result in correctness_results:
        name = result['short_name'].replace('_', r'\_')
        edges = result['total_edges']
        correct = result['verdict_counts'].get('CORRECT', 0)
        partial = result['verdict_counts'].get('PARTIALLY_CORRECT', 0)
        incorrect = result['verdict_counts'].get('INCORRECT', 0)
        approval = result['approval_rate']
        
        total_edges += edges
        total_correct += correct
        total_partial += partial
        total_incorrect += incorrect
        
        latex.append(f"{name} & {edges} & {correct} & {partial} & {incorrect} & {approval:.1%} \\\\")
    
    # Calculate percentages for total row
    pct_correct = (total_correct / total_edges * 100) if total_edges > 0 else 0
    pct_partial = (total_partial / total_edges * 100) if total_edges > 0 else 0
    pct_incorrect = (total_incorrect / total_edges * 100) if total_edges > 0 else 0
    overall_approval = (total_correct + 0.5 * total_partial) / total_edges if total_edges > 0 else 0
    
    latex.append(r"""\midrule
\textbf{Total} & \textbf{""" + str(total_edges) + r"""} & \textbf{""" + f"{total_correct} ({pct_correct:.1f}\\%)" + r"""} & \textbf{""" + f"{total_partial} ({pct_partial:.1f}\\%)" + r"""} & \textbf{""" + f"{total_incorrect} ({pct_incorrect:.1f}\\%)" + r"""} & \textbf{""" + f"{overall_approval:.1%}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 3: Citation Judging Results
    latex.append(r"""
% Table: Citation Judging Results  
\begin{table}[htbp]
\centering
\caption{Citation-Based Judge Validation Results on Physics CLDs}
\label{tab:physics-citation-results}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{CLD} & \textbf{Edges} & \textbf{Fully} & \textbf{Partial} & \textbf{Not} & \textbf{No Cit.} & \textbf{Approval} \\
\midrule""")
    
    total_edges = 0
    total_fully = 0
    total_partial = 0
    total_not = 0
    total_nocit = 0
    
    for result in citation_results:
        name = result['short_name'].replace('_', r'\_')
        edges = result['total_edges']
        fully = result['verdict_counts'].get('Fully supported', 0)
        partial = result['verdict_counts'].get('Partially supported', 0)
        not_sup = result['verdict_counts'].get('Not supported', 0)
        no_cit = result['verdict_counts'].get('NO_CITATION', 0)
        approval = result['approval_rate']
        
        total_edges += edges
        total_fully += fully
        total_partial += partial
        total_not += not_sup
        total_nocit += no_cit
        
        latex.append(f"{name} & {edges} & {fully} & {partial} & {not_sup} & {no_cit} & {approval:.1%} \\\\")
    
    # Calculate percentages for total row
    pct_fully = (total_fully / total_edges * 100) if total_edges > 0 else 0
    pct_partial = (total_partial / total_edges * 100) if total_edges > 0 else 0
    pct_not = (total_not / total_edges * 100) if total_edges > 0 else 0
    pct_nocit = (total_nocit / total_edges * 100) if total_edges > 0 else 0
    overall_approval = (total_fully + 0.5 * total_partial) / total_edges if total_edges > 0 else 0
    
    latex.append(r"""\midrule
\textbf{Total} & \textbf{""" + str(total_edges) + r"""} & \textbf{""" + f"{total_fully} ({pct_fully:.1f}\\%)" + r"""} & \textbf{""" + f"{total_partial} ({pct_partial:.1f}\\%)" + r"""} & \textbf{""" + f"{total_not} ({pct_not:.1f}\\%)" + r"""} & \textbf{""" + f"{total_nocit} ({pct_nocit:.1f}\\%)" + r"""} & \textbf{""" + f"{overall_approval:.1%}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 4: Comparison Summary
    latex.append(r"""
% Table: Comparison Summary
\begin{table}[htbp]
\centering
\caption{Comparison of Correctness vs Citation-Based Judging on Physics CLDs}
\label{tab:physics-comparison}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{CLD} & \textbf{Correctness Approval} & \textbf{Citation Approval} \\
\midrule""")
    
    for corr, cit in zip(correctness_results, citation_results):
        name = corr['short_name'].replace('_', r'\_')
        latex.append(f"{name} & {corr['approval_rate']:.1%} & {cit['approval_rate']:.1%} \\\\")
    
    corr_overall = sum(r['approval_rate'] * r['total_edges'] for r in correctness_results) / sum(r['total_edges'] for r in correctness_results)
    cit_overall = sum(r['approval_rate'] * r['total_edges'] for r in citation_results) / sum(r['total_edges'] for r in citation_results)
    
    latex.append(r"""\midrule
\textbf{Overall} & \textbf{""" + f"{corr_overall:.1%}" + r"""} & \textbf{""" + f"{cit_overall:.1%}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    return '\n'.join(latex)


def print_summary(correctness_results: List[Dict], citation_results: List[Dict]):
    """Print summary to console."""
    
    print("\n" + "="*80)
    print("PHYSICS CLD BATCH ANALYSIS - SUMMARY")
    print("="*80)
    print(f"Judge Model: gpt-4.1")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Correctness Results
    print("\n" + "─"*80)
    print("CORRECTNESS-BASED JUDGING (CORRECT/PARTIALLY_CORRECT/INCORRECT)")
    print("─"*80)
    print(f"{'CLD':<30} {'Edges':>6} {'CORRECT':>8} {'PARTIAL':>8} {'INCORR':>8} {'Approval':>10}")
    print("-" * 80)
    
    for result in correctness_results:
        correct = result['verdict_counts'].get('CORRECT', 0)
        partial = result['verdict_counts'].get('PARTIALLY_CORRECT', 0)
        incorrect = result['verdict_counts'].get('INCORRECT', 0)
        print(f"{result['short_name']:<30} {result['total_edges']:>6} {correct:>8} {partial:>8} {incorrect:>8} {result['approval_rate']:>9.1%}")
    
    total_edges = sum(r['total_edges'] for r in correctness_results)
    overall = sum(r['approval_rate'] * r['total_edges'] for r in correctness_results) / total_edges
    print("-" * 80)
    print(f"{'TOTAL':<30} {total_edges:>6} {'':>8} {'':>8} {'':>8} {overall:>9.1%}")
    
    # Citation Results
    print("\n" + "─"*80)
    print("CITATION-BASED JUDGING (Fully/Partially/Not supported)")
    print("─"*80)
    print(f"{'CLD':<25} {'Edges':>6} {'Fully':>6} {'Part':>6} {'Not':>6} {'NoCit':>6} {'Approval':>10}")
    print("-" * 80)
    
    for result in citation_results:
        fully = result['verdict_counts'].get('Fully supported', 0)
        partial = result['verdict_counts'].get('Partially supported', 0)
        not_sup = result['verdict_counts'].get('Not supported', 0)
        no_cit = result['verdict_counts'].get('NO_CITATION', 0)
        print(f"{result['short_name']:<25} {result['total_edges']:>6} {fully:>6} {partial:>6} {not_sup:>6} {no_cit:>6} {result['approval_rate']:>9.1%}")
    
    total_edges = sum(r['total_edges'] for r in citation_results)
    overall = sum(r['approval_rate'] * r['total_edges'] for r in citation_results) / total_edges
    print("-" * 80)
    print(f"{'TOTAL':<25} {total_edges:>6} {'':>6} {'':>6} {'':>6} {'':>6} {overall:>9.1%}")


def main():
    parser = argparse.ArgumentParser(description="Physics CLD Batch Analysis")
    parser.add_argument("--overview-only", action="store_true", help="Only show CLD overview, don't run tests")
    parser.add_argument("--skip-citation", action="store_true", help="Skip citation-based judging (faster)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PHYSICS CLD BATCH ANALYSIS")
    print("="*80)
    print("Testing judge on physics-based CLDs derived from published equations")
    print()
    
    # Print CLD overview
    print_cld_overview()
    
    if args.overview_only:
        return 0
    
    # Validate all CLDs have required files
    valid_clds = [k for k in PHYSICS_CLDS if check_cld_files(k)]
    if not valid_clds:
        logger.error("No valid CLDs found!")
        return 1
    
    print(f"\n\n{'='*80}")
    print("RUNNING JUDGE TESTS")
    print("="*80)
    
    # Run correctness judging
    print("\n>>> Running CORRECTNESS judging...")
    correctness_results = []
    for cld_key in valid_clds:
        logger.info(f"Testing {cld_key} (correctness)...")
        try:
            session_id, discovery = load_cld_to_neo4j(cld_key, approach="correctness")
            judged_edges, duration = run_judge_test(discovery, approach="correctness")
            result = analyze_results(cld_key, judged_edges, duration, "correctness")
            result["session_id"] = session_id
            correctness_results.append(result)
            logger.info(f"  ✅ {result['approval_rate']:.1%} approval ({duration:.1f}s)")
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
    
    # Run citation judging
    citation_results = []
    if not args.skip_citation:
        print("\n>>> Running CITATION judging...")
        for cld_key in valid_clds:
            logger.info(f"Testing {cld_key} (citation)...")
            try:
                session_id, discovery = load_cld_to_neo4j(cld_key, approach="citation")
                citation_hints_path = PHYSICS_CLDS[cld_key]["citation_hints"]
                judged_edges, duration = run_judge_test(discovery, approach="citation", citation_hints_path=citation_hints_path)
                result = analyze_results(cld_key, judged_edges, duration, "citation")
                result["session_id"] = session_id
                citation_results.append(result)
                logger.info(f"  ✅ {result['approval_rate']:.1%} approval ({duration:.1f}s)")
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
    
    # Print summary
    if correctness_results:
        print_summary(correctness_results, citation_results if citation_results else [])
    
    # Generate LaTeX tables
    if correctness_results and citation_results:
        latex_content = generate_latex_tables(correctness_results, citation_results)
        latex_file = RESULTS_DIR / f"physics_cld_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        print(f"\n📄 LaTeX tables saved to: {latex_file}")
    
    # Save JSON results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_json = {
        "timestamp": timestamp,
        "judge_model": "gpt-4.1",
        "correctness_results": correctness_results,
        "citation_results": citation_results if citation_results else None
    }
    
    json_file = RESULTS_DIR / f"physics_cld_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"💾 JSON results saved to: {json_file}")
    
    print(f"\n{'='*80}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

