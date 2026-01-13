#!/usr/bin/env python3
"""
Generate enhanced comparison table for physics CLD validation.

This script creates a table comparing correctness vs citation approval,
including citation retrieval rates and approval for retrieved-only edges.

Output: Markdown and LaTeX formatted tables.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_results(results_path: str) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_cld_results(data: dict) -> list:
    """Analyze results and compute all metrics per CLD."""
    metadata = data.get("cld_metadata", {})
    
    # Handle both array and dict formats for results
    correctness_raw = data.get("correctness_results", [])
    citation_raw = data.get("citation_results", [])
    
    # Convert arrays to dicts keyed by cld_key
    if isinstance(correctness_raw, list):
        correctness = {r['cld_key']: r for r in correctness_raw}
    else:
        correctness = correctness_raw
        
    if isinstance(citation_raw, list):
        citation = {r['cld_key']: r for r in citation_raw}
    else:
        citation = citation_raw
    
    cld_order = ['thermostat_heating_system', 'predator_prey', 'rc_circuit', 'water_tank']
    cld_names = {
        'thermostat_heating_system': 'Thermostat',
        'predator_prey': 'Predator-Prey',
        'rc_circuit': 'RC Circuit',
        'water_tank': 'Water Tank'
    }
    
    results = []
    
    for cld_key in cld_order:
        if cld_key not in correctness or cld_key not in citation:
            continue
        
        corr_edges = correctness[cld_key].get("edges", [])
        cit_edges = citation[cld_key].get("edges", [])
        
        # Correctness metrics
        total_edges = len(corr_edges)
        correct_count = sum(1 for e in corr_edges if e.get("judge_verdict") == "CORRECT")
        correctness_approval = correct_count / total_edges * 100 if total_edges else 0
        
        # Citation metrics
        # Check for NO_CITATION first (from judge_verdict), then check aggregate_verdict
        def get_citation_verdict(e):
            # If judge_verdict is NO_CITATION, that takes precedence
            if e.get("judge_verdict") == "NO_CITATION":
                return "NO_CITATION"
            # Otherwise use aggregate_verdict (normalized to uppercase with underscores)
            agg = e.get("aggregate_verdict", "")
            if agg:
                return agg.upper().replace(" ", "_")
            return e.get("judge_verdict", "").upper().replace(" ", "_")
        
        fully = sum(1 for e in cit_edges if get_citation_verdict(e) == "FULLY_SUPPORTED")
        partial = sum(1 for e in cit_edges if get_citation_verdict(e) == "PARTIALLY_SUPPORTED")
        not_supported = sum(1 for e in cit_edges if get_citation_verdict(e) == "NOT_SUPPORTED")
        no_citation = sum(1 for e in cit_edges if get_citation_verdict(e) == "NO_CITATION")
        
        # Citation approval (all edges)
        citation_approval_all = (fully + partial) / total_edges * 100 if total_edges else 0
        
        # Citations retrieved
        retrieved = total_edges - no_citation
        retrieval_rate = retrieved / total_edges * 100 if total_edges else 0
        
        # Citation approval (retrieved only)
        citation_approval_retrieved = (fully + partial) / retrieved * 100 if retrieved > 0 else 0
        
        # Gap (correctness - citation retrieved only)
        gap = correctness_approval - citation_approval_retrieved
        
        results.append({
            'cld_key': cld_key,
            'name': cld_names.get(cld_key, cld_key),
            'total_edges': total_edges,
            'correctness_approval': correctness_approval,
            'citation_approval_all': citation_approval_all,
            'retrieved': retrieved,
            'retrieval_rate': retrieval_rate,
            'citation_approval_retrieved': citation_approval_retrieved,
            'gap': gap,
            'fully': fully,
            'partial': partial,
            'not_supported': not_supported,
            'no_citation': no_citation
        })
    
    return results


def print_markdown_table(results: list):
    """Print the comparison table in markdown format."""
    print("\n## TABLE: CORRECTNESS vs CITATION APPROVAL COMPARISON\n")
    print("| CLD | Correctness | Citation (All) | Citations Retrieved | Citation (Retrieved Only) | Gap |")
    print("|-----|-------------|----------------|---------------------|---------------------------|-----|")
    
    total_edges = sum(r['total_edges'] for r in results)
    total_retrieved = sum(r['retrieved'] for r in results)
    total_fully = sum(r['fully'] for r in results)
    total_partial = sum(r['partial'] for r in results)
    
    for r in results:
        print(f"| {r['name']} | {r['correctness_approval']:.1f}% | {r['citation_approval_all']:.1f}% | "
              f"{r['retrieval_rate']:.1f}% ({r['retrieved']}/{r['total_edges']}) | "
              f"{r['citation_approval_retrieved']:.1f}% | {r['gap']:.1f}% |")
    
    # Totals
    total_corr = sum(r['correctness_approval'] * r['total_edges'] for r in results) / total_edges
    total_cit_all = (total_fully + total_partial) / total_edges * 100
    total_retrieval = total_retrieved / total_edges * 100
    total_cit_retrieved = (total_fully + total_partial) / total_retrieved * 100 if total_retrieved > 0 else 0
    total_gap = total_corr - total_cit_retrieved
    
    print(f"| **TOTAL** | **{total_corr:.1f}%** | **{total_cit_all:.1f}%** | "
          f"**{total_retrieval:.1f}% ({total_retrieved}/{total_edges})** | "
          f"**{total_cit_retrieved:.1f}%** | **{total_gap:.1f}%** |")
    
    print("\n**Key Insight**: When citations are successfully retrieved, the approval rate jumps from "
          f"{total_cit_all:.1f}% to **{total_cit_retrieved:.1f}%** — demonstrating the gap is primarily "
          "due to URL accessibility issues, not judge quality.\n")


def generate_latex_table(results: list) -> str:
    """Generate LaTeX table."""
    total_edges = sum(r['total_edges'] for r in results)
    total_retrieved = sum(r['retrieved'] for r in results)
    total_fully = sum(r['fully'] for r in results)
    total_partial = sum(r['partial'] for r in results)
    
    total_corr = sum(r['correctness_approval'] * r['total_edges'] for r in results) / total_edges
    total_cit_all = (total_fully + total_partial) / total_edges * 100
    total_retrieval = total_retrieved / total_edges * 100
    total_cit_retrieved = (total_fully + total_partial) / total_retrieved * 100 if total_retrieved > 0 else 0
    total_gap = total_corr - total_cit_retrieved
    
    latex = [r"""% Table: Correctness vs Citation Approval Comparison (Enhanced)
\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Comparison of Correctness vs Citation-Based Judging on Physics CLDs}
\label{tab:physics-comparison-enhanced}
\begin{tabular}{lccccc}
\toprule
\textbf{CLD} & \textbf{Correctness} & \textbf{Citation (All)} & \textbf{Citations Retrieved} & \textbf{Citation (Retr.)} & \textbf{Gap} \\
\midrule"""]
    
    for r in results:
        name = r['name'].replace('-', '--')
        latex.append(f"{name} & {r['correctness_approval']:.1f}\\% & {r['citation_approval_all']:.1f}\\% & "
                    f"{r['retrieval_rate']:.1f}\\% ({r['retrieved']}/{r['total_edges']}) & "
                    f"{r['citation_approval_retrieved']:.1f}\\% & {r['gap']:.1f}\\% \\\\")
    
    latex.append(r"""\midrule
\textbf{Total} & """ + f"{total_corr:.1f}\\%" + r""" & """ + f"{total_cit_all:.1f}\\%" + r""" & """ + 
                f"{total_retrieval:.1f}\\% ({total_retrieved}/{total_edges})" + r""" & """ + 
                f"{total_cit_retrieved:.1f}\\%" + r""" & """ + f"{total_gap:.1f}\\%" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} For the Correctness Judge, \emph{approval} = edges with verdict \texttt{CORRECT} (score 1.0). For the Citation Judge, \emph{approval} = edges with verdict \texttt{FULLY\_SUPPORTED} or \texttt{PARTIALLY\_SUPPORTED} (score $\ge 0.5$); \texttt{NO\_CITATION} = retrieval failure. ``Citation (All)'' counts \texttt{NO\_CITATION} as not approved; ``Citations Retrieved'' = edges with $\ge$1 successfully retrieved citation; ``Citation (Retr.)'' = approval conditioned on retrieval. Gap = Correctness $-$ Citation (Retr.).
\end{tablenotes}
\end{threeparttable}
\end{table}
""")
    
    return '\n'.join(latex)


def print_detailed_breakdown(results: list):
    """Print detailed breakdown per CLD."""
    print("\n## DETAILED BREAKDOWN BY CLD\n")
    
    for r in results:
        print(f"### {r['name']}")
        print(f"- Total edges: {r['total_edges']}")
        print(f"- Correctness: {r['correctness_approval']:.1f}%")
        print(f"- Citation verdicts: Fully={r['fully']}, Partial={r['partial']}, "
              f"Not Supported={r['not_supported']}, No Citation={r['no_citation']}")
        print(f"- Retrieved: {r['retrieved']}/{r['total_edges']} ({r['retrieval_rate']:.1f}%)")
        print(f"- Approval (all): {r['citation_approval_all']:.1f}%")
        print(f"- Approval (retrieved only): {r['citation_approval_retrieved']:.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced comparison table")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--output", type=str, help="Output LaTeX file path")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    args = parser.parse_args()
    
    # Find results file
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    
    if args.results:
        results_path = Path(args.results)
    else:
        # Find the most recent results file
        json_files = list(results_dir.glob("*results*.json"))
        if not json_files:
            print("Error: No results files found in results/")
            return 1
        results_path = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Using: {results_path}")
    
    # Load and analyze
    data = load_results(str(results_path))
    results = analyze_cld_results(data)
    
    if not results:
        print("Error: No CLD results found in file")
        return 1
    
    # Print markdown table
    print_markdown_table(results)
    
    # Print detailed breakdown if requested
    if args.detailed:
        print_detailed_breakdown(results)
    
    # Generate and save LaTeX
    latex = generate_latex_table(results)
    
    output_dir = results_dir / "physics_cld_batch_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"comparison_table_{timestamp}.tex"
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

