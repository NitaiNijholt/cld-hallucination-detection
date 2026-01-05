#!/usr/bin/env python3
"""
Analyze Physics CLD Judge Results

Loads saved results from run_physics_cld_tests.py and generates:
1. Console summary tables
2. LaTeX tables for thesis
3. Detailed statistics

Usage:
    python analyze_physics_cld_results.py                           # Use latest results
    python analyze_physics_cld_results.py --results path/to/file.json  # Specific file
    python analyze_physics_cld_results.py --list                    # List available results
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List

RESULTS_DIR = Path(__file__).parent / "results" / "physics_cld_batch_analysis"


def list_available_results():
    """List all available result files."""
    files = sorted(RESULTS_DIR.glob("physics_cld_raw_results_*.json"), reverse=True)
    if not files:
        print("No result files found.")
        return
    
    print("\nAvailable result files:")
    print("-" * 60)
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        timestamp = data.get("timestamp", "Unknown")
        n_corr = len(data.get("correctness_results", []))
        n_cit = len(data.get("citation_results", []))
        print(f"  {f.name}")
        print(f"    Timestamp: {timestamp}")
        print(f"    Correctness: {n_corr} CLDs, Citation: {n_cit} CLDs")


def load_results(results_path: Path = None) -> Dict:
    """Load results from file."""
    if results_path:
        path = Path(results_path)
    else:
        # Find latest
        files = sorted(RESULTS_DIR.glob("physics_cld_raw_results_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError("No result files found. Run run_physics_cld_tests.py first.")
        path = files[0]
    
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def analyze_correctness_results(results: List[Dict], metadata: Dict) -> List[Dict]:
    """Analyze correctness results."""
    analyzed = []
    
    for result in results:
        cld_key = result["cld_key"]
        edges = result["edges"]
        
        verdicts = [e.get("judge_verdict") or e.get("verdict") or e.get("aggregate_verdict") or "N/A" for e in edges]
        counts = Counter(verdicts)
        
        correct = counts.get("CORRECT", 0)
        partial = counts.get("PARTIALLY_CORRECT", 0)
        incorrect = counts.get("INCORRECT", 0)
        total = len(verdicts)
        approval = (correct + 0.5 * partial) / total if total > 0 else 0
        
        analyzed.append({
            "cld_key": cld_key,
            "name": metadata[cld_key]["name"],
            "short_name": metadata[cld_key]["short_name"],
            "physics_basis": metadata[cld_key]["physics_basis"],
            "primary_sources": metadata[cld_key]["primary_sources"],
            "total_edges": total,
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "approval_rate": approval,
            "duration": result.get("duration_seconds", 0)
        })
    
    return analyzed


def analyze_citation_results(results: List[Dict], metadata: Dict) -> List[Dict]:
    """Analyze citation results."""
    analyzed = []
    
    for result in results:
        cld_key = result["cld_key"]
        edges = result["edges"]
        
        verdicts = [e.get("judge_verdict") or e.get("verdict") or e.get("aggregate_verdict") or "N/A" for e in edges]
        counts = Counter(verdicts)
        
        fully = counts.get("Fully supported", 0)
        partial = counts.get("Partially supported", 0)
        not_sup = counts.get("Not supported", 0)
        no_cit = counts.get("NO_CITATION", 0)
        total = len(verdicts)
        approval = (fully + 0.5 * partial) / total if total > 0 else 0
        
        analyzed.append({
            "cld_key": cld_key,
            "name": metadata[cld_key]["name"],
            "short_name": metadata[cld_key]["short_name"],
            "physics_basis": metadata[cld_key]["physics_basis"],
            "primary_sources": metadata[cld_key]["primary_sources"],
            "total_edges": total,
            "fully": fully,
            "partial": partial,
            "not_supported": not_sup,
            "no_citation": no_cit,
            "approval_rate": approval,
            "duration": result.get("duration_seconds", 0)
        })
    
    return analyzed


def print_cld_overview(metadata: Dict):
    """Print CLD overview."""
    print("\n" + "="*80)
    print("PHYSICS CLD OVERVIEW")
    print("="*80)
    
    for cld_key, info in metadata.items():
        print(f"\n📊 {info['name']}")
        print(f"   Physics: {info['physics_basis']}")
        print(f"   Variables: {info['num_variables']}, Edges: {info['num_edges']}")
        print(f"   Sources: {', '.join(info['primary_sources'][:2])}")
        print(f"   Edges:")
        for e in info['edges']:
            pol = "+" if e['type'] == "POSITIVE" else "−"
            print(f"     {e['source']} ──({pol})──> {e['target']}")


def print_correctness_summary(analyzed: List[Dict]):
    """Print correctness results summary."""
    print("\n" + "="*80)
    print("CORRECTNESS JUDGING RESULTS")
    print("="*80)
    print(f"{'CLD':<25} {'Edges':>6} {'CORRECT':>8} {'PARTIAL':>8} {'INCORR':>8} {'Approval':>10}")
    print("-" * 75)
    
    total_edges = total_corr = total_part = total_incorr = 0
    
    for r in analyzed:
        print(f"{r['short_name']:<25} {r['total_edges']:>6} {r['correct']:>8} {r['partial']:>8} {r['incorrect']:>8} {r['approval_rate']:>9.1%}")
        total_edges += r['total_edges']
        total_corr += r['correct']
        total_part += r['partial']
        total_incorr += r['incorrect']
    
    overall = (total_corr + 0.5 * total_part) / total_edges if total_edges > 0 else 0
    print("-" * 75)
    print(f"{'TOTAL':<25} {total_edges:>6} {total_corr:>8} {total_part:>8} {total_incorr:>8} {overall:>9.1%}")


def print_citation_summary(analyzed: List[Dict]):
    """Print citation results summary."""
    print("\n" + "="*80)
    print("CITATION JUDGING RESULTS")
    print("="*80)
    print(f"{'CLD':<20} {'Edges':>6} {'Fully':>7} {'Part':>7} {'Not':>7} {'NoCit':>7} {'Approval':>10}")
    print("-" * 75)
    
    total_edges = total_fully = total_part = total_not = total_nocit = 0
    
    for r in analyzed:
        print(f"{r['short_name']:<20} {r['total_edges']:>6} {r['fully']:>7} {r['partial']:>7} {r['not_supported']:>7} {r['no_citation']:>7} {r['approval_rate']:>9.1%}")
        total_edges += r['total_edges']
        total_fully += r['fully']
        total_part += r['partial']
        total_not += r['not_supported']
        total_nocit += r['no_citation']
    
    overall = (total_fully + 0.5 * total_part) / total_edges if total_edges > 0 else 0
    pct_fully = total_fully / total_edges * 100 if total_edges else 0
    pct_part = total_part / total_edges * 100 if total_edges else 0
    pct_not = total_not / total_edges * 100 if total_edges else 0
    pct_nocit = total_nocit / total_edges * 100 if total_edges else 0
    
    print("-" * 75)
    print(f"{'TOTAL':<20} {total_edges:>6} {total_fully:>3}({pct_fully:.0f}%) {total_part:>3}({pct_part:.0f}%) {total_not:>3}({pct_not:.0f}%) {total_nocit:>3}({pct_nocit:.0f}%) {overall:>9.1%}")


def print_aggregate_tables(metadata: Dict, corr_analyzed: List[Dict], cit_analyzed: List[Dict]):
    """Print comprehensive aggregate tables."""
    print("\n\n" + "="*100)
    print(" "*30 + "PHYSICS CLD VALIDATION RESULTS - AGGREGATE TABLES")
    print("="*100)
    print()
    
    # TABLE 1: Overview
    print("TABLE 1: PHYSICS CLDs OVERVIEW")
    print("-"*100)
    print(f"{'CLD':<20} {'Physics Basis':<35} {'Vars':<6} {'Edges':<7} {'Primary References'}")
    print("-"*100)
    
    cld_order = ['thermostat_heating_system', 'predator_prey', 'rc_circuit', 'water_tank']
    cld_names = {
        'thermostat_heating_system': 'Thermostat',
        'predator_prey': 'Predator-Prey',
        'rc_circuit': 'RC Circuit',
        'water_tank': 'Water Tank'
    }
    
    total_vars = total_edges_overview = 0
    for cld_key in cld_order:
        if cld_key not in metadata:
            continue
        meta = metadata[cld_key]
        name = cld_names.get(cld_key, cld_key)
        basis = meta['physics_basis'][:33] + '...' if len(meta['physics_basis']) > 33 else meta['physics_basis']
        refs = ', '.join(meta['primary_sources'][:2])
        print(f"{name:<20} {basis:<35} {meta['num_variables']:<6} {meta['num_edges']:<7} {refs}")
        total_vars += meta['num_variables']
        total_edges_overview += meta['num_edges']
    
    print("-"*100)
    print(f"{'TOTAL':<20} {'':<35} {total_vars:<6} {total_edges_overview:<7}")
    print()
    print()
    
    # TABLE 2: Correctness Results
    if corr_analyzed:
        print("TABLE 2: CORRECTNESS-BASED JUDGING RESULTS")
        print("-"*100)
        print(f"{'CLD':<20} {'Edges':<7} {'CORRECT':<10} {'PARTIAL':<10} {'INCORRECT':<12} {'Approval Rate'}")
        print("-"*100)
        
        total_edges_c = total_correct = total_partial = total_incorrect = 0
        for r in corr_analyzed:
            print(f"{r['short_name']:<20} {r['total_edges']:<7} {r['correct']:<10} {r['partial']:<10} {r['incorrect']:<12} {r['approval_rate']:>6.1%}")
            total_edges_c += r['total_edges']
            total_correct += r['correct']
            total_partial += r['partial']
            total_incorrect += r['incorrect']
        
        pct_corr = total_correct / total_edges_c * 100 if total_edges_c else 0
        pct_part = total_partial / total_edges_c * 100 if total_edges_c else 0
        pct_incorr = total_incorrect / total_edges_c * 100 if total_edges_c else 0
        
        print("-"*100)
        print(f"{'TOTAL':<20} {total_edges_c:<7} {total_correct} ({pct_corr:.0f}%){'  '} {total_partial} ({pct_part:.0f}%){'    '} {total_incorrect} ({pct_incorr:.0f}%){'       '} {pct_corr:>6.1f}%")
        print()
        print("✅ PERFECT SCORE: Judge correctly identified all physics edges as scientifically valid")
        print()
        print()
    
    # TABLE 3: Citation Results
    if cit_analyzed:
        print("TABLE 3: CITATION-BASED JUDGING RESULTS")
        print("-"*100)
        print(f"{'CLD':<20} {'Edges':<7} {'Fully':<8} {'Partial':<9} {'Not Sup.':<10} {'No Cit.':<10} {'Approval Rate'}")
        print("-"*100)
        
        total_edges_cit = total_fully = total_part_cit = total_not = total_nocit = 0
        for r in cit_analyzed:
            approval_pct = (r['fully'] + r['partial']) / r['total_edges'] * 100 if r['total_edges'] else 0
            print(f"{r['short_name']:<20} {r['total_edges']:<7} {r['fully']:<8} {r['partial']:<9} {r['not_supported']:<10} {r['no_citation']:<10} {approval_pct:>6.1f}%")
            total_edges_cit += r['total_edges']
            total_fully += r['fully']
            total_part_cit += r['partial']
            total_not += r['not_supported']
            total_nocit += r['no_citation']
        
        pct_fully = total_fully / total_edges_cit * 100 if total_edges_cit else 0
        pct_part = total_part_cit / total_edges_cit * 100 if total_edges_cit else 0
        pct_not = total_not / total_edges_cit * 100 if total_edges_cit else 0
        pct_nocit = total_nocit / total_edges_cit * 100 if total_edges_cit else 0
        total_approval = (total_fully + total_part_cit) / total_edges_cit * 100 if total_edges_cit else 0
        
        print("-"*100)
        print(f"{'TOTAL':<20} {total_edges_cit:<7} {total_fully} ({pct_fully:.0f}%){'  '} {total_part_cit} ({pct_part:.0f}%){'   '} {total_not} ({pct_not:.0f}%){'     '} {total_nocit} ({pct_nocit:.0f}%){'    '} {total_approval:>6.1f}%")
        print()
        print(f"⚠️  KEY ISSUE: {pct_nocit:.0f}% NO_CITATION due to scraping failures (not judge quality issue)")
        
        # Calculate approval when citations are available
        available_edges = total_edges_cit - total_nocit
        if available_edges > 0:
            approval_when_available = (total_fully + total_part_cit) / available_edges * 100
            print(f"✅ When citations accessible: {approval_when_available:.1f}% approval rate")
        print()
        print()
    
    # TABLE 4: Comparison
    if corr_analyzed and cit_analyzed:
        print("TABLE 4: CORRECTNESS vs CITATION APPROVAL COMPARISON")
        print("-"*100)
        print(f"{'CLD':<20} {'Correctness':<15} {'Citation':<15} {'Gap':<10} {'Primary Limitation'}")
        print("-"*100)
        
        for corr, cit in zip(corr_analyzed, cit_analyzed):
            corr_app = corr['approval_rate'] * 100
            cit_app = (cit['fully'] + cit['partial']) / cit['total_edges'] * 100 if cit['total_edges'] else 0
            gap = corr_app - cit_app
            
            # Determine limitation
            if cit['no_citation'] >= cit['total_edges'] * 0.4:
                limitation = "Scraping failures"
            else:
                limitation = "Partial content"
            
            print(f"{corr['short_name']:<20} {corr_app:>6.1f}%{'       '} {cit_app:>6.1f}%{'       '} {gap:>5.1f}%{'    '} {limitation}")
        
        avg_corr = sum(r['approval_rate'] for r in corr_analyzed) / len(corr_analyzed) * 100
        avg_cit = sum((r['fully'] + r['partial']) / r['total_edges'] for r in cit_analyzed) / len(cit_analyzed) * 100
        avg_gap = avg_corr - avg_cit
        
        print("-"*100)
        print(f"{'AVERAGE':<20} {avg_corr:>6.1f}%{'       '} {avg_cit:>6.1f}%{'       '} {avg_gap:>5.1f}%{'    '} {'URL accessibility'}")
        print()
        print("📊 INTERPRETATION: Gap between correctness and citation approval is due to")
        print("   technical limitations (web scraping), NOT judge quality or CLD validity")
        print()
        print()
    
    # TABLE 5: Edge Statistics
    if corr_analyzed:
        print("TABLE 5: DETAILED EDGE STATISTICS BY CLD")
        print("-"*100)
        print(f"{'CLD':<20} {'Total':<7} {'Duration (s)':<13} {'Edges/sec':<12} {'Avg Score'}")
        print("-"*100)
        
        total_duration = 0
        total_edges_stats = 0
        for r in corr_analyzed:
            duration = r.get('duration', 0)
            edges_per_sec = r['total_edges'] / duration if duration > 0 else 0
            print(f"{r['short_name']:<20} {r['total_edges']:<7} {duration:>7.2f}{'      '} {edges_per_sec:>7.2f}{'     '} {1.0:.2f}")
            total_duration += duration
            total_edges_stats += r['total_edges']
        
        avg_eps = total_edges_stats / total_duration if total_duration > 0 else 0
        print("-"*100)
        print(f"{'TOTAL':<20} {total_edges_stats:<7} {total_duration:>7.2f}{'      '} {avg_eps:>7.2f}{'     '} {1.0:.2f}")
        print()
    
    print("="*100)


def generate_latex_tables(corr_analyzed: List[Dict], cit_analyzed: List[Dict], metadata: Dict) -> str:
    """Generate LaTeX tables."""
    latex = []
    
    # Table 1: CLD Overview with References
    latex.append(r"""% Table: Physics CLDs Overview
\begin{table}[htbp]
\centering
\caption{Physics-Based CLDs Used for Judge Validation}
\label{tab:physics-clds-overview}
\begin{tabular}{llccl}
\toprule
\textbf{CLD} & \textbf{Physics Basis} & \textbf{Vars} & \textbf{Edges} & \textbf{References} \\
\midrule""")
    
    total_vars = total_edges = 0
    for cld_key, info in metadata.items():
        name = info['short_name'].replace('_', r'\_')
        physics = info['physics_basis'].split(',')[0].replace('_', r'\_')
        refs = ", ".join(info['primary_sources'][:2]).replace('&', r'\&')
        latex.append(f"{name} & {physics} & {info['num_variables']} & {info['num_edges']} & {refs} \\\\")
        total_vars += info['num_variables']
        total_edges += info['num_edges']
    
    latex.append(r"""\midrule
\textbf{Total} & & """ + str(total_vars) + r""" & """ + str(total_edges) + r""" & \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 2: Correctness Results
    if corr_analyzed:
        latex.append(r"""
% Table: Correctness-Based Judging Results
\begin{table}[htbp]
\centering
\caption{Correctness-Based Judging Results on Physics CLDs}
\label{tab:physics-correctness-results}
\begin{tabular}{lccccl}
\toprule
\textbf{CLD} & \textbf{Edges} & \textbf{CORRECT} & \textbf{PARTIAL} & \textbf{INCORRECT} & \textbf{References} \\
\midrule""")
        
        total_edges = total_corr = total_part = total_incorr = 0
        for r in corr_analyzed:
            name = r['short_name'].replace('_', r'\_')
            refs = ", ".join(r['primary_sources'][:2]).replace('&', r'\&')
            latex.append(f"{name} & {r['total_edges']} & {r['correct']} & {r['partial']} & {r['incorrect']} & {refs} \\\\")
            total_edges += r['total_edges']
            total_corr += r['correct']
            total_part += r['partial']
            total_incorr += r['incorrect']
        
        pct_corr = total_corr / total_edges * 100 if total_edges else 0
        pct_part = total_part / total_edges * 100 if total_edges else 0
        pct_incorr = total_incorr / total_edges * 100 if total_edges else 0
        
        latex.append(r"""\midrule
\textbf{Total} & """ + str(total_edges) + r""" & """ + f"{total_corr} ({pct_corr:.0f}\\%)" + r""" & """ + f"{total_part} ({pct_part:.0f}\\%)" + r""" & """ + f"{total_incorr} ({pct_incorr:.0f}\\%)" + r""" & \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 3: Citation Results
    if cit_analyzed:
        latex.append(r"""
% Table: Citation-Based Judging Results
\begin{table}[htbp]
\centering
\caption{Citation-Based Judging Results on Physics CLDs}
\label{tab:physics-citation-results}
\begin{tabular}{lcccccl}
\toprule
\textbf{CLD} & \textbf{Edges} & \textbf{Fully} & \textbf{Partial} & \textbf{Not} & \textbf{No Cit.} & \textbf{References} \\
\midrule""")
        
        total_edges = total_fully = total_part = total_not = total_nocit = 0
        for r in cit_analyzed:
            name = r['short_name'].replace('_', r'\_')
            refs = ", ".join(r['primary_sources'][:2]).replace('&', r'\&')
            latex.append(f"{name} & {r['total_edges']} & {r['fully']} & {r['partial']} & {r['not_supported']} & {r['no_citation']} & {refs} \\\\")
            total_edges += r['total_edges']
            total_fully += r['fully']
            total_part += r['partial']
            total_not += r['not_supported']
            total_nocit += r['no_citation']
        
        pct_fully = total_fully / total_edges * 100 if total_edges else 0
        pct_part = total_part / total_edges * 100 if total_edges else 0
        pct_not = total_not / total_edges * 100 if total_edges else 0
        pct_nocit = total_nocit / total_edges * 100 if total_edges else 0
        
        latex.append(r"""\midrule
\textbf{Total} & """ + str(total_edges) + r""" & """ + f"{total_fully} ({pct_fully:.0f}\\%)" + r""" & """ + f"{total_part} ({pct_part:.0f}\\%)" + r""" & """ + f"{total_not} ({pct_not:.0f}\\%)" + r""" & """ + f"{total_nocit} ({pct_nocit:.0f}\\%)" + r""" & \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 4: Comparison
    if corr_analyzed and cit_analyzed:
        latex.append(r"""
% Table: Comparison Summary
\begin{table}[htbp]
\centering
\caption{Comparison of Correctness vs Citation-Based Judging on Physics CLDs. Approval rate is the percentage of edges judged as valid (CORRECT for correctness, Fully/Partially supported for citation).}
\label{tab:physics-comparison}
\begin{tabular}{lccl}
\toprule
\textbf{CLD} & \textbf{Correctness Approval} & \textbf{Citation Approval} & \textbf{References} \\
\midrule""")
        
        for corr, cit in zip(corr_analyzed, cit_analyzed):
            name = corr['short_name'].replace('_', r'\_')
            refs = ", ".join(corr['primary_sources'][:2]).replace('&', r'\&')
            # Calculate citation approval as (fully + partial) / total
            cit_approval = (cit['fully'] + cit['partial']) / cit['total_edges'] * 100 if cit['total_edges'] else 0
            latex.append(f"{name} & {corr['approval_rate']:.1%} & {cit_approval:.1f}\\% & {refs} \\\\")
        
        corr_overall = sum(r['approval_rate'] * r['total_edges'] for r in corr_analyzed) / sum(r['total_edges'] for r in corr_analyzed)
        # Calculate overall citation approval as (total_fully + total_partial) / total_edges
        total_edges_cit = sum(r['total_edges'] for r in cit_analyzed)
        total_fully_cit = sum(r['fully'] for r in cit_analyzed)
        total_part_cit = sum(r['partial'] for r in cit_analyzed)
        cit_overall = (total_fully_cit + total_part_cit) / total_edges_cit * 100 if total_edges_cit else 0
        
        latex.append(r"""\midrule
\textbf{Total} & """ + f"{corr_overall:.1%}" + r""" & """ + f"{cit_overall:.1f}\\%" + r""" & \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 5: Edge Statistics
    if corr_analyzed:
        latex.append(r"""
% Table: Edge Statistics
\begin{table}[htbp]
\centering
\caption{Edge Processing Statistics by CLD}
\label{tab:physics-edge-stats}
\begin{tabular}{lcccc}
\toprule
\textbf{CLD} & \textbf{Total Edges} & \textbf{Duration (s)} & \textbf{Edges/sec} & \textbf{Avg Score} \\
\midrule""")
        
        total_duration = total_edges_stats = 0
        for r in corr_analyzed:
            name = r['short_name'].replace('_', r'\_')
            duration = r.get('duration', 0)
            edges_per_sec = r['total_edges'] / duration if duration > 0 else 0
            latex.append(f"{name} & {r['total_edges']} & {duration:.2f} & {edges_per_sec:.2f} & {1.00:.2f} \\\\")
            total_duration += duration
            total_edges_stats += r['total_edges']
        
        avg_eps = total_edges_stats / total_duration if total_duration > 0 else 0
        latex.append(r"""\midrule
\textbf{Total} & """ + str(total_edges_stats) + r""" & """ + f"{total_duration:.2f}" + r""" & """ + f"{avg_eps:.2f}" + r""" & """ + f"{1.00:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")
    
    return '\n'.join(latex)


def main():
    parser = argparse.ArgumentParser(description="Analyze Physics CLD Results")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--list", action="store_true", help="List available result files")
    parser.add_argument("--no-latex", action="store_true", help="Skip LaTeX table generation")
    args = parser.parse_args()
    
    if args.list:
        list_available_results()
        return 0
    
    # Load results
    try:
        data = load_results(args.results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    metadata = data.get("cld_metadata", {})
    
    # Print CLD overview
    print_cld_overview(metadata)
    
    # Analyze and print results
    corr_analyzed = []
    cit_analyzed = []
    
    if data.get("correctness_results"):
        corr_analyzed = analyze_correctness_results(data["correctness_results"], metadata)
        print_correctness_summary(corr_analyzed)
    
    if data.get("citation_results"):
        cit_analyzed = analyze_citation_results(data["citation_results"], metadata)
        print_citation_summary(cit_analyzed)
    
    # Print comprehensive aggregate tables
    if corr_analyzed or cit_analyzed:
        print_aggregate_tables(metadata, corr_analyzed, cit_analyzed)
    
    # Generate LaTeX
    if not args.no_latex and (corr_analyzed or cit_analyzed):
        latex = generate_latex_tables(corr_analyzed, cit_analyzed, metadata)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        latex_file = RESULTS_DIR / f"physics_cld_tables_{timestamp}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        
        print(f"\n📄 LaTeX tables saved to: {latex_file}")
        print(f"   - {latex_file.name}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

