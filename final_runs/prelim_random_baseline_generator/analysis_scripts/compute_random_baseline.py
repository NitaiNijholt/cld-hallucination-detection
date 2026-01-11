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
import shutil

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from scipy import stats


# Ground truth CLD paths
# From .../final_runs/prelim_random_baseline_generator/analysis_scripts/
#   parents[0] = analysis_scripts/, [1] = prelim_random_baseline_generator/, [2] = final_runs/, [3] = project root
REPO_ROOT = Path(__file__).resolve().parents[3]  # project root
HERE = Path(__file__).resolve().parent

# Prefer a self-contained copy of the GT CLDs under this final_runs module.
# Fallback locations: 1) analysis_lib (self-contained), 2) data_science (legacy)
LOCAL_GROUND_TRUTH_DIR = HERE / "ground_truth_clds_for_experiments"
ANALYSIS_LIB_GROUND_TRUTH_DIR = REPO_ROOT / "final_runs" / "analysis_lib" / "data" / "ground_truth_clds"
CANONICAL_GROUND_TRUTH_DIR = (
    REPO_ROOT
    / "data_science"
    / "parameter_tuning_experiments"
    / "ground_truth_clds_for_experiments"
)
GROUND_TRUTH_DIR = LOCAL_GROUND_TRUTH_DIR

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

def _ensure_local_ground_truth_dir() -> None:
    """
    Ensure GT CLD JSONs exist under final_runs/random_baseline_generator/ground_truth_clds_for_experiments/.

    This keeps the random-baseline generator self-contained while still allowing the repo to keep the
    canonical GT CLDs under data_science/ or analysis_lib/.
    """
    LOCAL_GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    to_copy: list[tuple[Path, Path]] = []
    for _, info in VALIDATION_CLDS.items():
        for key in ("vars_file", "edges_file"):
            local_fp = LOCAL_GROUND_TRUTH_DIR / info[key]
            # Check multiple source locations in priority order
            analysis_lib_fp = ANALYSIS_LIB_GROUND_TRUTH_DIR / info[key]
            canon_fp = CANONICAL_GROUND_TRUTH_DIR / info[key]
            # Use analysis_lib first (self-contained), then data_science (legacy)
            if analysis_lib_fp.exists():
                to_copy.append((analysis_lib_fp, local_fp))
            elif canon_fp.exists():
                to_copy.append((canon_fp, local_fp))
            else:
                to_copy.append((canon_fp, local_fp))  # Will fail with clear error

    for source_fp, local_fp in to_copy:
        if not source_fp.exists():
            raise FileNotFoundError(
                f"Missing GT CLD file. Checked: '{ANALYSIS_LIB_GROUND_TRUTH_DIR}', '{CANONICAL_GROUND_TRUTH_DIR}'."
            )
        shutil.copy2(source_fp, local_fp)


def _mean_ci_95(mean: float, std: float, n: int) -> tuple[float, float]:
    """Normal-approx 95% CI for a mean."""
    if n <= 1:
        return (mean, mean)
    se = std / np.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def _try_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_random_baseline_comparison_figure(
    *,
    random_baseline_results: dict,
    llm_f1_variability_results: dict,
    out_png: Path,
) -> None:
    """
    Thesis-friendly comparison plot:
      - Random baseline: mean ± 95% CI (Monte Carlo)
      - LLM: mean ± 95% CI across 3 runs (t-distribution, df=2)
    """
    cld_order = ["depressive", "social_norms", "emergency_dept"]
    cld_labels = ["Depressive\\nSymptoms", "Social\\nNorms", "Emergency\\nDept"]

    rb_means: list[float] = []
    rb_err_low: list[float] = []
    rb_err_high: list[float] = []
    llm_means: list[float] = []
    llm_err_low: list[float] = []
    llm_err_high: list[float] = []
    improv: list[float] = []

    for key in cld_order:
        rb = random_baseline_results["clds"][key]["baseline"]["f1"]
        rb_mean = float(rb["mean"])
        rb_std = float(rb["std"])
        rb_n = int(random_baseline_results.get("n_simulations", rb.get("n_simulations", 0)) or 0)
        rb_ci = _mean_ci_95(rb_mean, rb_std, rb_n)

        llm = llm_f1_variability_results["clds"][key]["statistics"]
        llm_mean = float(llm["mean"])
        llm_vals = [float(v) for v in llm.get("values", [])]
        if len(llm_vals) >= 2:
            llm_std = float(np.std(llm_vals, ddof=1))
            llm_se = llm_std / np.sqrt(len(llm_vals))
            t_crit = float(stats.t.ppf(0.975, len(llm_vals) - 1))
            llm_ci_hw = t_crit * llm_se
        else:
            llm_ci_hw = 0.0

        rb_means.append(rb_mean)
        rb_err_low.append(rb_mean - rb_ci[0])
        rb_err_high.append(rb_ci[1] - rb_mean)
        llm_means.append(llm_mean)
        llm_err_low.append(llm_ci_hw)
        llm_err_high.append(llm_ci_hw)

        improv.append(llm_mean / rb_mean if rb_mean > 0 else float("nan"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(cld_order))
    w = 0.36

    ax.bar(
        x - w / 2,
        rb_means,
        width=w,
        color="#7f8c8d",
        edgecolor="black",
        linewidth=1.0,
        label="Random baseline (mean ± 95% CI)",
        yerr=np.array([rb_err_low, rb_err_high]),
        capsize=5,
    )
    ax.bar(
        x + w / 2,
        llm_means,
        width=w,
        color="#2c7fb8",
        edgecolor="black",
        linewidth=1.0,
        label="LLM generator (mean ± 95% CI)",
        yerr=np.array([llm_err_low, llm_err_high]),
        capsize=5,
    )

    for i, r in enumerate(improv):
        if np.isnan(r):
            continue
        ax.text(
            x[i] + w / 2,
            llm_means[i] + llm_err_high[i] + 0.02,
            f"{r:.1f}×",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cld_labels)
    ax.set_ylabel("Edge F1")
    ax.set_ylim(0, max(max(llm_means) + max(llm_err_high) + 0.08, 0.65))
    ax.set_title("LLM generator vs. random baseline edge F1 across validation CLDs", fontweight="bold")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_validation_graph(cld_key: str) -> Tuple[nx.DiGraph, List[str]]:
    """
    Load a validation CLD from JSON files.
    
    Returns:
        Tuple of (NetworkX DiGraph, list of node names)
    """
    cld_info = VALIDATION_CLDS[cld_key]
    
    _ensure_local_ground_truth_dir()

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
    
    # Add edges (normalize names to avoid accidental node inflation from typos/casing)
    canonical_nodes = set(node_names)
    lower_to_canonical = {n.lower(): n for n in node_names}
    # Known dataset aliases/typos observed in GT edge JSONs
    alias_map = {
        # Social Norms
        "individual ideal bmi": "Individual Ideal BMI",
        # Older persons ED
        "multimorbidity": "Multi-morbidity",
        "incorrect medication use": "Incorrect use of medication",
    }

    def _normalize_node(name: str) -> str | None:
        if name in canonical_nodes:
            return name
        n = str(name).strip()
        if n in canonical_nodes:
            return n
        # Alias/typo mapping (case-insensitive)
        alias = alias_map.get(n.lower())
        if alias and alias in canonical_nodes:
            return alias
        # Case-insensitive match (covers e.g. 'individual ideal BMI' vs 'Individual Ideal BMI')
        cand = lower_to_canonical.get(n.lower())
        if cand:
            return cand
        # No safe normalization found
        return None

    for edge in edges_data:
        src = _normalize_node(edge.get("source", ""))
        tgt = _normalize_node(edge.get("target", ""))
        if src is None or tgt is None:
            # Skip edges that refer to non-existent nodes; otherwise NetworkX will silently
            # create new nodes and inflate |V|, breaking comparability to the intended GT node set.
            continue
        G.add_edge(src, tgt, polarity=edge.get("polarity", "unknown"))
    
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
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{CLD} & $|V|$ & $|E|$ & \textbf{Density} & \textbf{Random F1 (mean $\pm$ 95\% CI)} \\",
        r"\midrule"
    ]
    
    for cld_key in ["depressive", "social_norms", "emergency_dept"]:
        if cld_key in results["clds"]:
            data = results["clds"][cld_key]
            baseline = data["baseline"]
            name = data["name"]

            # Monte Carlo mean ± 95% CI for the mean (normal approximation; n=1000)
            n_sim = int(baseline.get("n_simulations", results.get("n_simulations", 0)) or 0)
            f1_mean = float(baseline["f1"]["mean"])
            f1_std = float(baseline["f1"]["std"])
            ci_l, ci_u = _mean_ci_95(f1_mean, f1_std, n_sim)
            ci_hw = f1_mean - ci_l  # symmetric under normal approximation
            
            lines.append(
                f"{name} & {baseline['n_nodes']} & {baseline['n_edges']} & "
                f"{baseline['density']:.3f} & "
                f"{f1_mean:.3f}$\\pm${ci_hw:.3f} \\\\"
            )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textit{Note.} Random F1 computed via Monte Carlo simulation (1000 iterations) and reported as mean $\pm$ 95\% CI for the Monte Carlo mean.",
        r"\item Density = $|E| / (|V| \times (|V|-1))$.",
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

    # Always write thesis-facing outputs under final_runs/prelim_random_baseline_generator/Output/
    output_dir = REPO_ROOT / "final_runs" / "prelim_random_baseline_generator" / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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

    # Write canonical outputs for thesis/repro
    with open(output_dir / "random_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "random_baseline_table.tex", "w") as f:
        f.write(print_latex_table(results))

    # Generate the random-baseline comparison figure as a PNG (thesis uses PNGs).
    llm_path = HERE / "llm_f1_variability_results.json"
    llm = _try_load_json(llm_path)
    if llm is None:
        print(f"Note: {llm_path} not found; skipping random-baseline comparison figure generation.")
    else:
        out_png = HERE / "random_baseline_comparison.png"
        generate_random_baseline_comparison_figure(
            random_baseline_results=results,
            llm_f1_variability_results=llm,
            out_png=out_png,
        )
        print(f"Saved figure: {out_png}")
        shutil.copy2(out_png, output_dir / "random_baseline_comparison.png")
    
    if args.latex:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(print_latex_table(results))
    
    return results


if __name__ == "__main__":
    main()
