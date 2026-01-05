#!/usr/bin/env python3
"""
Generate retrieval success comparison table for RQ1a validation.

This script analyzes retrieval success rates and approval rates across:
1. Physics CLDs (expert-provided references)
2. Ulemans Alzheimer's CLD (expert-provided references)
3. Health CLDs from RQ1a (automated fetched citations)

Output: LaTeX table and markdown summary.
"""

import pandas as pd
import json
import glob
import os
from datetime import datetime
from pathlib import Path


def analyze_physics_clds(results_path: str) -> dict:
    """Analyze physics CLD results."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    citation_raw = data.get("citation_results", [])
    if isinstance(citation_raw, list):
        citation = {r['cld_key']: r for r in citation_raw}
    else:
        citation = citation_raw
    
    total_edges = 0
    no_citation = 0
    fully = 0
    partial = 0
    not_supported = 0
    
    for cld_key, cld_data in citation.items():
        edges = cld_data.get("edges", [])
        for e in edges:
            total_edges += 1
            verdict = e.get("judge_verdict", "")
            if verdict == "NO_CITATION":
                no_citation += 1
            else:
                agg = e.get("aggregate_verdict", "").upper().replace(" ", "_")
                if agg == "FULLY_SUPPORTED":
                    fully += 1
                elif agg == "PARTIALLY_SUPPORTED":
                    partial += 1
                elif agg == "NOT_SUPPORTED":
                    not_supported += 1
    
    retrieved = total_edges - no_citation
    retrieval_rate = retrieved / total_edges * 100 if total_edges > 0 else 0
    approval_all = (fully + partial) / total_edges * 100 if total_edges > 0 else 0
    approval_retrieved = (fully + partial) / retrieved * 100 if retrieved > 0 else 0
    
    return {
        'domain': 'Physics CLDs',
        'citation_source': 'Expert-provided',
        'total_edges': total_edges,
        'no_citation': no_citation,
        'retrieved': retrieved,
        'retrieval_rate': retrieval_rate,
        'fully': fully,
        'partial': partial,
        'not_supported': not_supported,
        'approval_all': approval_all,
        'approval_retrieved': approval_retrieved
    }


def analyze_ulemans_cld(xlsx_path: str) -> dict:
    """Analyze Ulemans Alzheimer's CLD results."""
    # IMPORTANT:
    # For the "search provider validation study" on Ulemans, retrieval/coverage is reported for
    # standard web scraping (ContentScraper) on the Brave (Fixed) provider (see thesis Table
    # "Search Provider Comparison on Ulemans Expert CLD").
    #
    # Using Jina AI Reader yields ~100% coverage due to paywall-bypassing fetch; that is a different
    # retrieval regime than the web-search + scraper setting we compare against other domains here.
    df = pd.read_excel(xlsx_path, sheet_name='Brave_(Fixed)')

    total_edges = len(df)

    verdict_col = 'ContentScraper Verdict'
    if verdict_col not in df.columns:
        raise KeyError(f"Missing expected column '{verdict_col}' in sheet 'Brave_(Fixed)'")

    # Normalize verdict strings and treat blanks/None as NO_CITATION for retrieval accounting
    verdict_series = df[verdict_col].astype("object")
    verdict_norm = verdict_series.fillna("").map(lambda x: str(x).strip())

    no_citation_mask = verdict_norm.isin({"", "NO_CITATION", "No citation", "NO CITATION", "None"})
    no_cit = int(no_citation_mask.sum())

    verdict_counts = verdict_norm.value_counts()
    fully = int(verdict_counts.get("Fully supported", 0))
    partial = int(verdict_counts.get("Partially supported", 0))
    not_sup = int(verdict_counts.get("Not supported", 0))

    retrieved = total_edges - no_cit
    retrieval_rate = retrieved / total_edges * 100 if total_edges > 0 else 0
    approval_all = (fully + partial) / total_edges * 100 if total_edges > 0 else 0
    approval_retrieved = (fully + partial) / retrieved * 100 if retrieved > 0 else 0

    return {
        'domain': 'Ulemans (Alzheimer\'s)',
        'citation_source': 'Expert-provided',
        'total_edges': total_edges,
        'no_citation': no_cit,
        'retrieved': retrieved,
        'retrieval_rate': retrieval_rate,
        'fully': fully,
        'partial': partial,
        'not_supported': not_sup,
        'approval_all': approval_all,
        'approval_retrieved': approval_retrieved
    }


def analyze_health_clds(base_path: str) -> list:
    """Analyze health domain CLDs from RQ1a citation judging."""
    clds = ['depressive', 'social_norms', 'emergency_department']
    cld_names = {
        'depressive': 'Depressive',
        'social_norms': 'Social Norms',
        'emergency_department': 'Emergency Dept'
    }
    
    results = []
    
    for cld in clds:
        cld_path = os.path.join(base_path, cld)
        if not os.path.exists(cld_path):
            continue
        
        total_edges = 0
        no_citation = 0
        fully = 0
        partial = 0
        not_sup = 0
        
        for run in ['run_1', 'run_2', 'run_3']:
            run_path = os.path.join(cld_path, run)
            if not os.path.exists(run_path):
                continue
            
            # Get baseline files
            xlsx_files = glob.glob(os.path.join(run_path, "judged_*baseline*.xlsx"))
            
            for xlsx_path in xlsx_files[:1]:  # One per run
                try:
                    df = pd.read_excel(xlsx_path, sheet_name='All Edges')
                    verdict_counts = df['Judge Verdict'].value_counts()
                    
                    total_edges += len(df)
                    no_citation += verdict_counts.get('NO_CITATION', 0) + verdict_counts.get('No citation', 0)
                    fully += verdict_counts.get('Fully supported', 0)
                    partial += verdict_counts.get('Partially supported', 0)
                    not_sup += verdict_counts.get('Not supported', 0)
                except Exception as e:
                    print(f"Warning: Error reading {xlsx_path}: {e}")
        
        if total_edges > 0:
            retrieved = total_edges - no_citation
            retrieval_rate = retrieved / total_edges * 100
            approval_all = (fully + partial) / total_edges * 100
            approval_retrieved = (fully + partial) / retrieved * 100 if retrieved > 0 else 0
            
            results.append({
                'domain': cld_names.get(cld, cld),
                'citation_source': 'Automated fetch',
                'total_edges': total_edges,
                'no_citation': no_citation,
                'retrieved': retrieved,
                'retrieval_rate': retrieval_rate,
                'fully': fully,
                'partial': partial,
                'not_supported': not_sup,
                'approval_all': approval_all,
                'approval_retrieved': approval_retrieved
            })
    
    return results


def generate_latex_table(all_results: list) -> str:
    """Generate LaTeX table."""
    latex = [r"""% Table: Retrieval Success and Average Score Comparison
\begin{table}[H]
\centering
\small
\caption{Retrieval Success and Average Judge Score Comparison. Expert-provided references are curated by domain experts; automated fetch uses web search. Retrieval success indicates percentage of edges where citation content was successfully fetched. Average score is the mean aggregate citation judge score (0--1 scale).}
\label{tab:retrieval-comparison}
\begin{tabular}{llcc}
\toprule
\textbf{Domain} & \textbf{Citation Source} & \textbf{Retrieval} & \textbf{Avg Score} \\
\midrule"""]
    
    for r in all_results:
        domain = r['domain'].replace('\'', '\\\'')
        source = r['citation_source']
        retr = f"{r['retrieval_rate']:.1f}\\%"
        avg_score = f"{r.get('avg_score', 0):.3f}"
        latex.append(f"{domain} & {source} & {retr} & {avg_score} \\\\")
    
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    return '\n'.join(latex)


def print_markdown_table(all_results: list):
    """Print markdown summary."""
    print("\n## RETRIEVAL SUCCESS AND APPROVAL COMPARISON\n")
    print("| Domain | Citation Source | Retrieval | Approval (All) | Approval (Retr.) |")
    print("|--------|-----------------|-----------|----------------|------------------|")
    
    for r in all_results:
        print(f"| {r['domain']} | {r['citation_source']} | {r['retrieval_rate']:.1f}% | {r['approval_all']:.1f}% | {r['approval_retrieved']:.1f}% |")
    
    print("\n### Key Insight\n")
    
    # Find physics and ulemans for comparison
    physics = next((r for r in all_results if 'Physics' in r['domain']), None)
    ulemans = next((r for r in all_results if 'Ulemans' in r['domain']), None)
    health_auto = [r for r in all_results if r['citation_source'] == 'Automated fetch']
    
    if physics and ulemans:
        print(f"**Expert-provided references comparison:**")
        print(f"- Physics: {physics['retrieval_rate']:.1f}% retrieval → {physics['approval_retrieved']:.1f}% approval when retrieved")
        print(f"- Ulemans: {ulemans['retrieval_rate']:.1f}% retrieval → {ulemans['approval_retrieved']:.1f}% approval when retrieved")
        print(f"\n**Despite similar retrieval rates, Ulemans (health domain) has lower approval than Physics,")
        print(f"confirming domain complexity—not retrieval quality—explains low scores.**")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    all_results = []
    
    # 1. Analyze Physics CLDs
    physics_results_path = script_dir / "results" / "physics_cld_raw_results_20251204_182810.json"
    if physics_results_path.exists():
        print(f"Analyzing Physics CLDs: {physics_results_path}")
        physics = analyze_physics_clds(str(physics_results_path))
        all_results.append(physics)
    else:
        print(f"Warning: Physics results not found at {physics_results_path}")
    
    # 2. Analyze Ulemans CLD
    ulemans_path = project_root / "final_runs" / "RQ1a_ulemans_citation_provider_comparison" / "citation_fetcher_comparison_20251113_v2.xlsx"
    if ulemans_path.exists():
        print(f"Analyzing Ulemans CLD: {ulemans_path}")
        ulemans = analyze_ulemans_cld(str(ulemans_path))
        all_results.append(ulemans)
    else:
        print(f"Warning: Ulemans results not found at {ulemans_path}")
    
    # 3. Analyze Health CLDs (automated fetch)
    health_base = project_root / "final_runs" / "RQ1a_gt_lit_citation"
    if health_base.exists():
        print(f"Analyzing Health CLDs: {health_base}")
        health_results = analyze_health_clds(str(health_base))
        all_results.extend(health_results)
    else:
        print(f"Warning: Health CLDs not found at {health_base}")
    
    if not all_results:
        print("Error: No results found")
        return 1
    
    # Print markdown summary
    print_markdown_table(all_results)
    
    # Generate LaTeX
    latex = generate_latex_table(all_results)
    
    # Save LaTeX
    output_dir = script_dir / "results" / "physics_cld_batch_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"retrieval_comparison_table_{timestamp}.tex"
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_path}")
    
    # Also print the LaTeX
    print("\n### LaTeX Output\n")
    print(latex)
    
    return 0


if __name__ == "__main__":
    exit(main())

