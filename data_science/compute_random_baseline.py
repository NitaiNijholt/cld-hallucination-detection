#!/usr/bin/env python3
"""
Compute Random Baseline for LLM Generator Evaluation

This script generates random graph baselines to contextualize LLM generator performance.
For each validation CLD, it:
1. Loads the ground truth nodes and edges
2. Generates N random directed graphs with same |V| and |E|
3. Computes edge precision/recall/F1 against validation
4. Reports mean ± std as random baseline

The random baseline demonstrates that LLM generators perform significantly
above chance at recovering validation CLD structure.

Usage:
    python compute_random_baseline.py [--n-simulations 1000] [--output results.json]
"""

import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import random


# Ground truth CLD paths
GROUND_TRUTH_DIR = Path(__file__).parent / "parameter_tuning_experiments" / "ground_truth_clds_for_experiments"

VALIDATION_CLDS = {
    "depressive": {
        "name": "Depressive Symptoms",
        "vars_file": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_vars_data.json",
        "edges_file": "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults_edges_data.json"
    },
    "social_norms": {
        "name": "Social Norms",
        "vars_file": "Social_norms_and_obesity_prevalence_vars_data.json",
        "edges_file": "Social_norms_and_obesity_prevalence_edges_data.json"
    },
    "emergency_dept": {
        "name": "Emergency Department",
        "vars_file": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_vars_data.json",
        "edges_file": "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet_edges_data.json"
    }
}


def load_validation_graph(cld_key: str) -> Tuple[nx.DiGraph, List[str]]:
    """
    Load a validation CLD from JSON files.
    
    Returns:
        Tuple of (NetworkX DiGraph, list of node names)
    """
    cld_info = VALIDATION_CLDS[cld_key]
    
    # Load variables
    vars_path = GROUND_TRUTH_DIR / cld_info["vars_file"]
    with open(vars_path, 'r') as f:
        vars_data = json.load(f)
    
    # Load edges
    edges_path = GROUND_TRUTH_DIR / cld_info["edges_file"]
    with open(edges_path, 'r') as f:
        edges_data = json.load(f)
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes
    node_names = [v["name"] for v in vars_data]
    for node in node_names:
        G.add_node(node)
    
    # Add edges
    for edge in edges_data:
        G.add_edge(edge["source"], edge["target"], 
                   polarity=edge.get("polarity", "unknown"))
    
    return G, node_names


def generate_random_graph_same_nodes(
    node_names: List[str], 
    n_edges: int, 
    seed: int = None
) -> nx.DiGraph:
    """
    Generate a random directed graph using the same node names as validation.
    
    This is the stricter comparison mode - random edges between exact same nodes.
    """
    if seed is not None:
        random.seed(seed)
    
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    
    # Generate all possible directed edges (excluding self-loops)
    all_possible_edges = [
        (src, tgt) for src in node_names for tgt in node_names if src != tgt
    ]
    
    # Randomly sample n_edges
    if n_edges > len(all_possible_edges):
        n_edges = len(all_possible_edges)
    
    selected_edges = random.sample(all_possible_edges, n_edges)
    G.add_edges_from(selected_edges)
    
    return G


def compute_edge_metrics(
    generated_edges: set, 
    validation_edges: set
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for edge overlap.
    """
    tp = len(generated_edges & validation_edges)
    fp = len(generated_edges - validation_edges)
    fn = len(validation_edges - generated_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_random_baseline_metrics(
    validation_graph: nx.DiGraph,
    node_names: List[str],
    n_simulations: int = 1000,
    base_seed: int = 42
) -> Dict[str, Any]:
    """
    Monte Carlo estimate of random baseline edge F1.
    
    Generates n_simulations random graphs with same nodes and edge count,
    computing edge metrics for each.
    """
    n = validation_graph.number_of_nodes()
    m = validation_graph.number_of_edges()
    validation_edges = set(validation_graph.edges())
    
    # Possible edges (excluding self-loops)
    max_possible_edges = n * (n - 1)
    density = m / max_possible_edges if max_possible_edges > 0 else 0
    
    # Monte Carlo simulation
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(n_simulations):
        seed = base_seed + i
        random_G = generate_random_graph_same_nodes(node_names, m, seed=seed)
        random_edges = set(random_G.edges())
        
        metrics = compute_edge_metrics(random_edges, validation_edges)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1"])
    
    # Analytical expected values
    # For random graph with m edges from n*(n-1) possible:
    # Expected TP = m * (m_val / max_possible) = m * density
    # Expected precision = TP / m = density (approximately)
    analytical_expected_precision = density
    analytical_expected_recall = density  # Same since m_random = m_validation
    analytical_expected_f1 = density  # F1 = precision when P = R
    
    return {
        "n_nodes": n,
        "n_edges": m,
        "max_possible_edges": max_possible_edges,
        "density": density,
        "n_simulations": n_simulations,
        "precision": {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
            "min": float(np.min(precisions)),
            "max": float(np.max(precisions)),
            "analytical_expected": analytical_expected_precision
        },
        "recall": {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls)),
            "min": float(np.min(recalls)),
            "max": float(np.max(recalls)),
            "analytical_expected": analytical_expected_recall
        },
        "f1": {
            "mean": float(np.mean(f1_scores)),
            "std": float(np.std(f1_scores)),
            "min": float(np.min(f1_scores)),
            "max": float(np.max(f1_scores)),
            "analytical_expected": analytical_expected_f1
        }
    }


def run_all_baselines(n_simulations: int = 1000) -> Dict[str, Any]:
    """
    Run random baseline computation for all validation CLDs.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_simulations": n_simulations,
        "clds": {}
    }
    
    for cld_key, cld_info in VALIDATION_CLDS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {cld_info['name']}")
        print(f"{'='*60}")
        
        # Load validation graph
        G, node_names = load_validation_graph(cld_key)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        density = m / (n * (n - 1)) if n > 1 else 0
        
        print(f"  Nodes: {n}")
        print(f"  Edges: {m}")
        print(f"  Density: {density:.4f}")
        
        # Compute random baseline
        print(f"  Running {n_simulations} simulations...")
        baseline = compute_random_baseline_metrics(G, node_names, n_simulations)
        
        print(f"  Random Baseline F1: {baseline['f1']['mean']:.4f} ± {baseline['f1']['std']:.4f}")
        print(f"  Analytical Expected: {baseline['f1']['analytical_expected']:.4f}")
        
        results["clds"][cld_key] = {
            "name": cld_info["name"],
            "baseline": baseline
        }
    
    return results


def print_latex_table(results: Dict[str, Any]) -> str:
    """
    Generate LaTeX table for thesis.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Random Baseline Comparison for Generator Edge Recovery}",
        r"\label{tab:random_baseline}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{CLD} & $|V|$ & $|E|$ & \textbf{Density} & \textbf{Random F1} & \textbf{Analytical} \\",
        r"\midrule"
    ]
    
    for cld_key in ["depressive", "social_norms", "emergency_dept"]:
        if cld_key in results["clds"]:
            data = results["clds"][cld_key]
            baseline = data["baseline"]
            name = data["name"]
            
            lines.append(
                f"{name} & {baseline['n_nodes']} & {baseline['n_edges']} & "
                f"{baseline['density']:.3f} & "
                f"{baseline['f1']['mean']:.3f}$\\pm${baseline['f1']['std']:.3f} & "
                f"{baseline['f1']['analytical_expected']:.3f} \\\\"
            )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Note.} Random F1 computed via Monte Carlo simulation (1000 iterations).",
        r"\item Density = $|E| / (|V| \times (|V|-1))$; Analytical = expected F1 based on graph density.",
        r"\item LLM generator must significantly exceed these baselines to demonstrate causal reasoning.",
        r"\end{tablenotes}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute random baseline for LLM generator evaluation"
    )
    parser.add_argument(
        "--n-simulations", "-n", type=int, default=1000,
        help="Number of Monte Carlo simulations (default: 1000)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON file path (default: stdout)"
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Print LaTeX table format"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RANDOM BASELINE COMPUTATION FOR LLM GENERATOR EVALUATION")
    print("=" * 60)
    print(f"Simulations per CLD: {args.n_simulations}")
    
    # Run baseline computation
    results = run_all_baselines(n_simulations=args.n_simulations)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'CLD':<25} {'|V|':>5} {'|E|':>5} {'Density':>8} {'Random F1':>15}")
    print("-" * 60)
    
    for cld_key in ["depressive", "social_norms", "emergency_dept"]:
        if cld_key in results["clds"]:
            data = results["clds"][cld_key]
            baseline = data["baseline"]
            print(
                f"{data['name']:<25} "
                f"{baseline['n_nodes']:>5} "
                f"{baseline['n_edges']:>5} "
                f"{baseline['density']:>8.3f} "
                f"{baseline['f1']['mean']:>7.3f} ± {baseline['f1']['std']:.3f}"
            )
    
    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    if args.latex:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(print_latex_table(results))
    
    return results


if __name__ == "__main__":
    main()







