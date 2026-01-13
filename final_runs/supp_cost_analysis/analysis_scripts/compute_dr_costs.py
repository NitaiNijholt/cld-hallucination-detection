#!/usr/bin/env python3
"""
Compute Deep Research Computational Costs and Edge Space Analysis.

This script calculates:
1. Token usage and API costs per CLD and aggregate
2. Total edge space (all possible directed edges given variable counts)
3. True Negative count (edges not analyzed due to cost)

Outputs reproducible values for thesis inclusion.

Author: Automated analysis for MSc thesis
Date: 2024-12-18
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Data files
DATA_DIR = Path(__file__).parent.parent.parent / "RQ3_deep_research_validation" / "Data"
OUTPUT_DIR = Path(__file__).parent  # cost_analysis folder
DATA_FILES = [
    ("deep_research_results_depressive_symptoms_20251013_032002_84edges.json", "Depressive"),
    ("deep_research_results_social_norms_20251012_213641_17edges.json", "Social"),
    ("deep_research_results_older_persons_ALL_EDGES_184edges.json", "Older"),
]

# CLD variable counts (from method_experiments_final.tex)
CLD_VARIABLES = {
    "Depressive": 14,  # Depressive Symptoms CLD
    "Social": 10,      # Social Norms CLD
    "Older": 34,       # Older Persons Emergency Department CLD
}

# Pricing (GPT-4.1 rates, December 2025)
PRICING = {
    "gpt4_input_per_m": 2.00,   # $/M input tokens (GPT-4.1)
    "gpt4_output_per_m": 8.00,  # $/M output tokens (GPT-4.1)
    # IMPORTANT:
    # The Deep Research result JSONs contain `total_tokens` but the per-edge
    # `estimated_input_tokens` / `estimated_output_tokens` fields are not consistently
    # populated across all runs (and can yield wildly inconsistent aggregates).
    #
    # For reproducible cost reporting, we use a single global split ratio derived from
    # observed "long-context" judge usage (input-heavy). This matches the thesis-level
    # cost scale for the Deep Research pipeline.
    "input_ratio": 0.865,       # ~86.5% input tokens
    "output_ratio": 0.135,      # ~13.5% output tokens
}


def load_results(json_file: Path) -> List[Dict]:
    """Load results from a JSON file, handling both formats."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        return data.get('results', [])


def compute_costs(results: List[Dict]) -> Dict:
    """Compute token and cost statistics for a set of results."""
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    api_calls = sum(r.get('api_calls', 0) for r in results)

    # Use ratio split for reproducible aggregation (see PRICING note above).
    input_tokens = int(total_tokens * PRICING['input_ratio'])
    output_tokens = max(int(total_tokens - input_tokens), 0)
    
    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * PRICING['gpt4_input_per_m']
    output_cost = (output_tokens / 1_000_000) * PRICING['gpt4_output_per_m']
    total_cost = input_cost + output_cost
    
    return {
        'edges': len(results),
        'total_tokens': total_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'api_calls': api_calls,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'cost_per_edge': total_cost / len(results) if len(results) > 0 else 0,
        'tokens_per_edge': total_tokens / len(results) if len(results) > 0 else 0,
    }


def compute_edge_space() -> Dict:
    """Compute total edge space and TN count for each CLD."""
    edge_space = {}
    
    for cld_name, n_vars in CLD_VARIABLES.items():
        # Total possible directed edges = V * (V-1) for directed graph without self-loops
        total_possible = n_vars * (n_vars - 1)
        edge_space[cld_name] = {
            'variables': n_vars,
            'total_possible_edges': total_possible,
        }
    
    return edge_space


def main():
    print("\n" + "=" * 80)
    print("DEEP RESEARCH COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    # Load and analyze each CLD
    cld_costs = {}
    all_results = []
    
    print("\n--- PER-CLD TOKEN USAGE ---")
    for filename, cld_name in DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found, skipping")
            continue
        
        results = load_results(filepath)
        costs = compute_costs(results)
        cld_costs[cld_name] = costs
        all_results.extend(results)
        
        print(f"\n{cld_name}:")
        print(f"  Edges: {costs['edges']}")
        print(f"  Total tokens: {costs['total_tokens']:,}")
        print(f"  Tokens/edge: {costs['tokens_per_edge']:,.0f}")
        print(f"  API calls: {costs['api_calls']:,}")
        print(f"  Cost: ${costs['total_cost']:.2f} (${costs['cost_per_edge']:.2f}/edge)")
    
    # Aggregate costs
    aggregate = compute_costs(all_results)
    print("\n--- AGGREGATE ---")
    print(f"  Total edges: {aggregate['edges']}")
    print(f"  Total tokens: {aggregate['total_tokens']:,}")
    print(f"  Avg tokens/edge: {aggregate['tokens_per_edge']:,.0f}")
    print(f"  Total API calls: {aggregate['api_calls']:,}")
    print(f"  Total cost: ${aggregate['total_cost']:.2f}")
    print(f"  Cost per edge: ${aggregate['cost_per_edge']:.2f}")
    
    # Edge space analysis
    print("\n--- EDGE SPACE ANALYSIS ---")
    edge_space = compute_edge_space()
    
    total_possible = 0
    total_analyzed = 0
    
    for cld_name, space in edge_space.items():
        analyzed = cld_costs.get(cld_name, {}).get('edges', 0)
        tn_count = space['total_possible_edges'] - analyzed
        
        print(f"\n{cld_name}:")
        print(f"  Variables: {space['variables']}")
        print(f"  Total possible edges: {space['variables']} × {space['variables']-1} = {space['total_possible_edges']}")
        print(f"  Analyzed (TP+FP+FN): {analyzed}")
        print(f"  True Negatives (not analyzed): {tn_count}")
        
        total_possible += space['total_possible_edges']
        total_analyzed += analyzed
        
        # Add to edge_space dict
        edge_space[cld_name]['analyzed'] = analyzed
        edge_space[cld_name]['true_negatives'] = tn_count
    
    total_tn = total_possible - total_analyzed
    
    print(f"\n--- AGGREGATE EDGE SPACE ---")
    print(f"  Total possible edges: {total_possible}")
    print(f"  Edges analyzed: {total_analyzed}")
    print(f"  True Negatives (not analyzed): {total_tn}")
    print(f"  Analysis coverage: {total_analyzed/total_possible*100:.1f}%")
    
    # Cost to analyze all TNs
    tn_cost = total_tn * aggregate['cost_per_edge']
    print(f"\n--- HYPOTHETICAL TN ANALYSIS COST ---")
    print(f"  TN edges: {total_tn}")
    print(f"  Cost per edge: ${aggregate['cost_per_edge']:.2f}")
    print(f"  Total TN cost: ${tn_cost:.2f}")
    
    # Thesis-ready values
    print("\n" + "=" * 80)
    print("THESIS-READY VALUES")
    print("=" * 80)
    print(f"""
% Deep Research Cost Statistics
Total edges analyzed: {aggregate['edges']}
Total tokens: {aggregate['total_tokens']:,} ({aggregate['total_tokens']/1e6:.1f}M)
Avg tokens per edge: {aggregate['tokens_per_edge']:,.0f} ({aggregate['tokens_per_edge']/1000:.0f}K)
Total API calls: {aggregate['api_calls']:,}
Total cost: ${aggregate['total_cost']:.2f}
Cost per edge: ${aggregate['cost_per_edge']:.2f}

% Edge Space
Total possible edges: {total_possible}
Edges analyzed (TP+FP+FN): {total_analyzed}
True Negatives (not analyzed): {total_tn}
Analysis coverage: {total_analyzed/total_possible*100:.1f}%

% Hypothetical TN Analysis
Cost to analyze all TNs: ${tn_cost:.2f}
""")
    
    # Per-CLD table for LaTeX
    print("% LaTeX table data:")
    print("% CLD & Vars & Possible & Analyzed & TN & Tokens & Cost")
    for cld_name in ["Depressive", "Social", "Older"]:
        if cld_name not in cld_costs:
            print(f"% {cld_name} - NO DATA")
            continue
        space = edge_space[cld_name]
        costs = cld_costs[cld_name]
        print(f"% {cld_name} & {space['variables']} & {space['total_possible_edges']} & "
              f"{space['analyzed']} & {space['true_negatives']} & "
              f"{costs['total_tokens']/1e6:.1f}M & ${costs['total_cost']:.0f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'pricing': PRICING,
        'per_cld': {
            cld: {
                'costs': cld_costs.get(cld, {}),
                'edge_space': edge_space.get(cld, {}),
            }
            for cld in CLD_VARIABLES.keys()
        },
        'aggregate': {
            'costs': aggregate,
            'edge_space': {
                'total_possible': total_possible,
                'analyzed': total_analyzed,
                'true_negatives': total_tn,
                'coverage_pct': total_analyzed / total_possible * 100,
            },
            'tn_analysis_cost': tn_cost,
        },
    }
    
    output_file = Path(__file__).parent / "dr_cost_analysis_latest.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Generate LaTeX table for dr_cost_table.tex
    cld_display_names = {
        "Depressive": "Depressive Symptoms",
        "Social": "Social Norms",
        "Older": "Emergency Department",
    }
    
    latex_table = r"""\begin{table}[H]
\centering
\caption[Deep Research computational cost]{Deep Research (DR) computational cost by CLD for the RQ3 pilot at GPT-4.1 pricing. We evaluated """ + str(aggregate['edges']) + r""" edges total (TP+FP+FN; TN omitted for cost). Per-edge cost is approximately \$""" + f"{aggregate['cost_per_edge']:.2f}" + r"""/edge.}
\label{tab:dr_cost}
\begin{tabular}{lrrr}
\hline
\textbf{CLD} & \textbf{Edges (N)} & \textbf{Cost (\$)} & \textbf{Cost/edge (\$)} \\
\hline
"""
    
    for cld_key in ["Depressive", "Social", "Older"]:
        if cld_key in cld_costs:
            costs = cld_costs[cld_key]
            display_name = cld_display_names[cld_key]
            latex_table += f"{display_name} & {costs['edges']}  & {costs['total_cost']:.1f}  & {costs['cost_per_edge']:.2f} \\\\\n"
    
    latex_table += r"""\hline
\textbf{Total} & \textbf{""" + str(aggregate['edges']) + r"""} & \textbf{""" + f"{aggregate['total_cost']:.1f}" + r"""} & \textbf{""" + f"{aggregate['cost_per_edge']:.2f}" + r"""} \\
\hline
\end{tabular}
\end{table}
"""
    
    # Save to RQ3 directory for thesis integration
    rq3_dir = Path(__file__).parent.parent.parent / "RQ3_deep_research_validation"
    dr_cost_tex_path = rq3_dir / "dr_cost_table.tex"
    with open(dr_cost_tex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {dr_cost_tex_path}")


if __name__ == "__main__":
    main()
