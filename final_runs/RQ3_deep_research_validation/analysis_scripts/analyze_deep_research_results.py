#!/usr/bin/env python3
"""
Analyze Deep Research Results: Compare direct causal evidence detection across TP, FN, and FP edges.

Updated 2025-12-18: Now uses full 285-edge dataset from all 3 CLDs (Depressive, Social, Older Persons)
instead of the previous 229-edge subset that only included FP edges from Older Persons CLD.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()


def compute_cohens_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's h effect size for proportion difference.
    
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    
    Cohen's h measures the difference between two proportions
    on an arcsine-transformed scale, which stabilizes variance.
    
    Interpretation (Cohen, 1988):
    - |h| < 0.20: negligible
    - 0.20 <= |h| < 0.50: small
    - 0.50 <= |h| < 0.80: medium
    - |h| >= 0.80: large
    
    Args:
        p1: First proportion (e.g., TP detection rate as decimal)
        p2: Second proportion (e.g., FP detection rate as decimal)
    
    Returns:
        Cohen's h value (positive means p1 > p2)
    """
    # Convert percentages to proportions if needed
    if p1 > 1:
        p1 = p1 / 100
    if p2 > 1:
        p2 = p2 / 100
    
    # Clamp to valid range [0, 1]
    p1 = max(0, min(1, p1))
    p2 = max(0, min(1, p2))
    
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    
    return phi1 - phi2


def interpret_cohens_h(h: float) -> str:
    """
    Interpret Cohen's h effect size (Cohen, 1988).
    
    Args:
        h: Cohen's h value
    
    Returns:
        Interpretation string
    """
    if np.isnan(h):
        return "—"
    h_abs = abs(h)
    if h_abs < 0.20:
        return "negligible"
    elif h_abs < 0.50:
        return "small"
    elif h_abs < 0.80:
        return "medium"
    else:
        return "large"


def test_proportion_difference(n1_success: int, n1_total: int, n2_success: int, n2_total: int) -> Dict:
    """
    Test difference between two proportions using z-test and Fisher's exact test.
    
    Uses two-proportion z-test (large sample) and Fisher's exact test (robust for small samples).
    Both tests are reported to demonstrate robustness.
    
    Args:
        n1_success: Number of successes in group 1 (e.g., TP detected)
        n1_total: Total in group 1 (e.g., total TP)
        n2_success: Number of successes in group 2 (e.g., FP detected)
        n2_total: Total in group 2 (e.g., total FP)
    
    Returns:
        Dict with z_stat, z_p, fisher_p, and sample size adequacy check
    """
    from scipy.stats import fisher_exact
    
    # Calculate proportions
    p1 = n1_success / n1_total if n1_total > 0 else 0
    p2 = n2_success / n2_total if n2_total > 0 else 0
    
    # Two-proportion z-test
    # Pooled proportion
    p_pooled = (n1_success + n2_success) / (n1_total + n2_total) if (n1_total + n2_total) > 0 else 0
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1_total + 1/n2_total)) if (n1_total > 0 and n2_total > 0 and p_pooled > 0 and p_pooled < 1) else np.nan
    
    # z statistic
    z_stat = (p1 - p2) / se if se > 0 else np.nan
    
    # p-value (two-tailed)
    from scipy.stats import norm
    z_p = 2 * (1 - norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan
    
    # Fisher's exact test (2x2 contingency table)
    # [[success1, failure1], [success2, failure2]]
    table = [[n1_success, n1_total - n1_success],
             [n2_success, n2_total - n2_success]]
    _, fisher_p = fisher_exact(table)
    
    # Sample size adequacy: expected cell counts >= 5 for chi-square/z-test
    expected_cells = [
        n1_total * (n1_success + n2_success) / (n1_total + n2_total),
        n1_total * (n1_total + n2_total - n1_success - n2_success) / (n1_total + n2_total),
        n2_total * (n1_success + n2_success) / (n1_total + n2_total),
        n2_total * (n1_total + n2_total - n1_success - n2_success) / (n1_total + n2_total)
    ]
    adequate_sample = all(e >= 5 for e in expected_cells)
    
    return {
        'p1': p1,
        'p2': p2,
        'z_stat': z_stat,
        'z_p': z_p,
        'fisher_p': fisher_p,
        'adequate_sample': adequate_sample,
        'min_expected_cell': min(expected_cells),
        'cohens_h': compute_cohens_h(p1 * 100, p2 * 100)
    }

# Input/output roots (support rerouting outputs under a single run folder)
# Default INPUT_DIR to the RQ3_deep_research folder (parent of analysis_scripts)
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = INPUT_DIR / "Data"

# Full 285-edge dataset: all 3 CLDs with complete TP/FP/FN coverage
DATA_FILES = [
    ("deep_research_results_depressive_symptoms_20251013_032002_84edges.json", "Depressive"),
    ("deep_research_results_social_norms_20251012_213641_17edges.json", "Social"),
    ("deep_research_results_older_persons_ALL_EDGES_184edges.json", "Older"),
]


def load_results(json_file: str) -> List[Dict]:
    """Load results from a JSON file, handling both formats."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Handle both formats: {'results': [...]} and direct list
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        return data.get('results', [])


def analyze_direct_causal_by_classification(results: List[Dict]) -> Dict:
    """Analyze direct causal evidence detection by classification."""
    stats = {}
    
    for result in results:
        cls = result['classification']
        
        if cls not in stats:
            stats[cls] = {
                'total': 0,
                'direct_causal_found': 0,
                'direct_causal_not_found': 0,
                'confidences': [],
                'tokens': []
            }
        
        stats[cls]['total'] += 1
        
        if result.get('direct_causal_found', False):
            stats[cls]['direct_causal_found'] += 1
            if result.get('strongest_causal_evidence_confidence'):
                stats[cls]['confidences'].append(result['strongest_causal_evidence_confidence'])
        else:
            stats[cls]['direct_causal_not_found'] += 1
        
        if result.get('total_tokens'):
            stats[cls]['tokens'].append(result['total_tokens'])
    
    # Calculate percentages and averages
    for cls in stats:
        s = stats[cls]
        s['detection_rate'] = (s['direct_causal_found'] / s['total']) * 100 if s['total'] > 0 else 0
        s['avg_confidence'] = sum(s['confidences']) / len(s['confidences']) if s['confidences'] else 0
        s['avg_tokens'] = sum(s['tokens']) / len(s['tokens']) if s['tokens'] else 0
    
    return stats


def print_classification_table(stats: Dict):
    """Print a table showing direct causal evidence detection by classification."""
    console.print("\n[bold cyan]═" * 90 + "[/bold cyan]")
    console.print("[bold cyan]DIRECT CAUSAL EVIDENCE DETECTION BY CLASSIFICATION[/bold cyan]")
    console.print("[bold cyan]═" * 90 + "[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class", style="cyan", width=8)
    table.add_column("Total\nEdges", justify="right", width=8)
    table.add_column("Direct Causal\nFOUND", justify="right", width=12)
    table.add_column("Direct Causal\nNOT Found", justify="right", width=12)
    table.add_column("Detection\nRate", justify="right", width=10)
    table.add_column("Avg Conf.\n(when found)", justify="right", width=12)
    table.add_column("Interpretation", width=30)
    
    # Sort by TP, FN, FP
    for cls in ['TP', 'FN', 'FP']:
        if cls in stats:
            s = stats[cls]
            
            if cls == 'TP':
                interp = "[green]✓ SHOULD find evidence[/green]"
            elif cls == 'FN':
                interp = "[yellow]Rejected by experts[/yellow]"
            else:  # FP
                interp = "[red]Spurious (not in GT)[/red]"
            
            table.add_row(
                cls,
                str(s['total']),
                f"[bold]{s['direct_causal_found']}[/bold]",
                str(s['direct_causal_not_found']),
                f"{s['detection_rate']:.1f}%",
                f"{s['avg_confidence']:.1f}/10" if s['avg_confidence'] > 0 else "N/A",
                interp
            )
    
    console.print(table)


def print_discrimination_analysis(stats: Dict):
    """Print discrimination analysis comparing classifications with Cohen's h effect sizes."""
    console.print("\n[bold yellow]═" * 90 + "[/bold yellow]")
    console.print("[bold yellow]DISCRIMINATION ANALYSIS (with Cohen's h effect sizes)[/bold yellow]")
    console.print("[bold yellow]═" * 90 + "[/bold yellow]\n")
    
    # Get detection rates
    tp_rate = stats.get('TP', {}).get('detection_rate', 0)
    fn_rate = stats.get('FN', {}).get('detection_rate', 0)
    fp_rate = stats.get('FP', {}).get('detection_rate', 0)
    
    # TP vs FP (most important for validation)
    if 'TP' in stats and 'FP' in stats:
        gap = tp_rate - fp_rate
        h = compute_cohens_h(tp_rate, fp_rate)
        h_interp = interpret_cohens_h(h)
        console.print(f"[bold]1. TP vs FP Discrimination:[/bold] {gap:.1f} pp, Cohen's h = {h:.3f} ({h_interp})")
        if gap > 20:
            console.print("   → [green]EXCELLENT[/green]: System strongly distinguishes true from false positives")
        elif gap > 10:
            console.print("   → [green]GOOD[/green]: System can distinguish true from false positives")
        elif gap > 5:
            console.print("   → [yellow]MODERATE[/yellow]: Some discrimination ability")
        else:
            console.print("   → [red]POOR[/red]: System cannot distinguish true from false positives!")
        console.print()
    
    # TP vs FN
    if 'TP' in stats and 'FN' in stats:
        gap = tp_rate - fn_rate
        h = compute_cohens_h(tp_rate, fn_rate)
        h_interp = interpret_cohens_h(h)
        console.print(f"[bold]2. TP vs FN Discrimination:[/bold] {gap:.1f} pp, Cohen's h = {h:.3f} ({h_interp})")
        if gap > 20:
            console.print("   → [green]GOOD[/green]: System distinguishes accepted vs rejected edges")
        elif gap > 10:
            console.print("   → [yellow]MODERATE[/yellow]: Some discrimination between accepted/rejected")
        else:
            console.print("   → [yellow]LIMITED[/yellow]: Weak discrimination")
        console.print()
    
    # FP vs FN (unexpected comparison)
    if 'FP' in stats and 'FN' in stats:
        gap = fp_rate - fn_rate
        h = compute_cohens_h(fp_rate, fn_rate)
        h_interp = interpret_cohens_h(h)
        console.print(f"[bold]3. FP vs FN Discrimination:[/bold] {gap:.1f} pp, Cohen's h = {h:.3f} ({h_interp})")
        if gap > 20:
            console.print("   → [red]SURPRISING[/red]: FP edges show MORE evidence than FN edges!")
        elif gap > 10:
            console.print("   → [yellow]UNEXPECTED[/yellow]: FP edges have higher detection than FN")
        elif gap > -10:
            console.print("   → [yellow]SIMILAR[/yellow]: FP and FN show similar detection rates")
        else:
            console.print("   → [green]EXPECTED[/green]: FN edges show more evidence than FP")
        console.print()
    
    # Print Cohen's h interpretation guide
    console.print("[dim]Cohen's h interpretation (Cohen, 1988): |h| < 0.20 = negligible, 0.20-0.50 = small, 0.50-0.80 = medium, ≥ 0.80 = large[/dim]\n")


def print_implications(stats: Dict):
    """Print implications of the findings."""
    console.print("\n[bold magenta]═" * 90 + "[/bold magenta]")
    console.print("[bold magenta]KEY FINDINGS & IMPLICATIONS[/bold magenta]")
    console.print("[bold magenta]═" * 90 + "[/bold magenta]\n")
    
    tp_rate = stats.get('TP', {}).get('detection_rate', 0)
    fn_rate = stats.get('FN', {}).get('detection_rate', 0)
    fp_rate = stats.get('FP', {}).get('detection_rate', 0)
    
    console.print("[bold]Detection Rates:[/bold]")
    if 'TP' in stats:
        console.print(f"  • TP (True Positives):  {tp_rate:.1f}% - [green]Ground truth edges experts accepted[/green]")
    if 'FN' in stats:
        console.print(f"  • FN (False Negatives): {fn_rate:.1f}% - [yellow]Ground truth edges experts rejected[/yellow]")
    if 'FP' in stats:
        console.print(f"  • FP (False Positives): {fp_rate:.1f}% - [red]LLM-generated, not in ground truth[/red]")
    console.print()
    
    # Key insights
    console.print("[bold]Insights:[/bold]")
    
    if fp_rate > 70 and abs(tp_rate - fp_rate) < 5:
        console.print("  • [red]⚠ Critical Issue:[/red] System finds evidence for FP edges at nearly the same")
        console.print("    rate as TP edges, suggesting it cannot reliably validate causal claims.")
    
    if fp_rate > fn_rate:
        console.print(f"  • [yellow]⚠ Unexpected Pattern:[/yellow] FP edges ({fp_rate:.1f}%) show MORE direct causal")
        console.print(f"    evidence than FN edges ({fn_rate:.1f}%). This suggests:")
        console.print("    - FP edges may represent plausible-but-excluded relationships")
        console.print("    - FN edges were rejected for non-evidential reasons (domain knowledge, scope)")
        console.print("    - The system may be detecting correlational rather than causal evidence")
    
    if tp_rate > fn_rate and tp_rate > fp_rate:
        console.print(f"  • [green]✓ Positive Finding:[/green] TP edges show highest detection rate")
    
    console.print()


def print_token_summary(stats: Dict):
    """Print token usage summary."""
    console.print("\n[bold blue]═" * 90 + "[/bold blue]")
    console.print("[bold blue]TOKEN USAGE SUMMARY[/bold blue]")
    console.print("[bold blue]═" * 90 + "[/bold blue]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Class", style="cyan", width=8)
    table.add_column("Edges", justify="right", width=10)
    table.add_column("Avg Tokens/Edge", justify="right", width=18)
    table.add_column("Total Tokens", justify="right", width=18)
    
    total_edges = 0
    total_tokens = 0
    
    for cls in ['TP', 'FN', 'FP']:
        if cls in stats and stats[cls]['tokens']:
            s = stats[cls]
            edges = len(s['tokens'])
            total = sum(s['tokens'])
            avg = s['avg_tokens']
            
            table.add_row(
                cls,
                str(edges),
                f"{avg:,.0f}",
                f"{total:,.0f}"
            )
            
            total_edges += edges
            total_tokens += total
    
    console.print(table)
    console.print(f"\n[bold]Overall Average:[/bold] {total_tokens/total_edges:,.0f} tokens/edge")
    console.print(f"[bold]Total Tokens:[/bold] {total_tokens:,.0f}")
    console.print(f"[bold]Estimated Cost:[/bold] ${total_tokens/1_000_000:.2f} (at $1/M tokens)")


def print_cld_breakdown(all_results: List[Dict], cld_names: Dict[str, str]):
    """Print detection rates broken down by CLD."""
    console.print("\n[bold green]═" * 90 + "[/bold green]")
    console.print("[bold green]DETECTION RATES BY CLD[/bold green]")
    console.print("[bold green]═" * 90 + "[/bold green]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("CLD", style="cyan", width=15)
    table.add_column("Class", width=6)
    table.add_column("Total", justify="right", width=8)
    table.add_column("Found", justify="right", width=8)
    table.add_column("Rate", justify="right", width=10)
    
    # Group by CLD
    cld_stats = {}
    for r in all_results:
        cld = r.get('CLD', 'Unknown')
        # Normalize CLD name
        cld_short = 'Older' if 'older' in cld.lower() or 'emergency' in cld.lower() else \
                    'Social' if 'social' in cld.lower() else \
                    'Depressive' if 'depressive' in cld.lower() else cld
        cls = r['classification']
        key = (cld_short, cls)
        
        if key not in cld_stats:
            cld_stats[key] = {'total': 0, 'found': 0}
        cld_stats[key]['total'] += 1
        if r.get('direct_causal_found'):
            cld_stats[key]['found'] += 1
    
    # Print by CLD
    for cld in ['Depressive', 'Social', 'Older']:
        for cls in ['TP', 'FP', 'FN']:
            key = (cld, cls)
            if key in cld_stats:
                s = cld_stats[key]
                rate = 100 * s['found'] / s['total'] if s['total'] > 0 else 0
                table.add_row(cld, cls, str(s['total']), str(s['found']), f"{rate:.1f}%")
        table.add_row("", "", "", "", "")  # Separator
    
    console.print(table)


def print_proportion_tests(stats: Dict):
    """Print statistical tests for proportion differences."""
    console.print("\n[bold cyan]═" * 90 + "[/bold cyan]")
    console.print("[bold cyan]STATISTICAL ASSUMPTION TESTS: PROPORTION COMPARISONS[/bold cyan]")
    console.print("[bold cyan]═" * 90 + "[/bold cyan]\n")
    
    if 'TP' not in stats or 'FP' not in stats:
        console.print("[red]Missing TP or FP data for proportion test[/red]")
        return {}
    
    # TP vs FP comparison
    tp_stats = stats['TP']
    fp_stats = stats['FP']
    
    tp_total = tp_stats['total']
    tp_found = tp_stats['direct_causal_found']
    fp_total = fp_stats['total']
    fp_found = fp_stats['direct_causal_found']
    
    result = test_proportion_difference(tp_found, tp_total, fp_found, fp_total)
    
    console.print("[bold]TP vs FP Detection Rate Comparison:[/bold]")
    console.print(f"  TP: {tp_found}/{tp_total} = {result['p1']*100:.1f}%")
    console.print(f"  FP: {fp_found}/{fp_total} = {result['p2']*100:.1f}%")
    console.print(f"  Gap: {(result['p1'] - result['p2'])*100:.1f} percentage points")
    console.print()
    console.print("[bold]Effect Size:[/bold]")
    console.print(f"  Cohen's h = {result['cohens_h']:.3f} ({interpret_cohens_h(result['cohens_h'])})")
    console.print()
    console.print("[bold]Significance Tests:[/bold]")
    console.print(f"  Two-proportion z-test: z = {result['z_stat']:.3f}, p = {result['z_p']:.4f}")
    console.print(f"  Fisher's exact test: p = {result['fisher_p']:.4f}")
    console.print()
    console.print("[bold]Assumption Check:[/bold]")
    console.print(f"  Sample size adequacy (all expected cells ≥ 5): {'Yes' if result['adequate_sample'] else 'No'}")
    console.print(f"  Minimum expected cell count: {result['min_expected_cell']:.1f}")
    console.print()
    
    # Interpretation
    sig = result['z_p'] < 0.05
    console.print("[bold]Conclusion:[/bold]")
    if sig:
        console.print(f"  [green]Significant difference (p < 0.05): TP detection rate differs from FP[/green]")
    else:
        console.print(f"  [yellow]No significant difference (p = {result['z_p']:.3f}): Cannot conclude TP ≠ FP[/yellow]")
    
    # Save results
    return {
        'comparison': 'TP_vs_FP',
        'tp_total': tp_total,
        'tp_found': tp_found,
        'fp_total': fp_total,
        'fp_found': fp_found,
        'z_stat': result['z_stat'],
        'z_p': result['z_p'],
        'fisher_p': result['fisher_p'],
        'cohens_h': result['cohens_h'],
        'adequate_sample': result['adequate_sample']
    }


def main():
    """Main analysis function."""
    console.print("\n[bold green]🔬 DEEP RESEARCH RESULTS ANALYSIS (FULL 285-EDGE DATASET)[/bold green]")
    console.print("[bold green]═" * 90 + "[/bold green]\n")
    
    console.print("[cyan]Using complete dataset from all 3 CLDs (no FP sample bias)[/cyan]\n")
    
    # Load all results from the three CLD files
    all_results = []
    cld_names = {}
    
    console.print("[cyan]Loading results...[/cyan]")
    for filename, cld_name in DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            console.print(f"[red]Error: File not found: {filepath}[/red]")
            return
        
        results = load_results(str(filepath))
        
        # Add CLD name to each result for breakdown analysis
        for r in results:
            if 'CLD' not in r:
                r['CLD'] = cld_name
        
        # Count classifications
        class_counts = Counter([r['classification'] for r in results])
        console.print(f"  • {cld_name}: {len(results)} edges (TP={class_counts.get('TP', 0)}, FP={class_counts.get('FP', 0)}, FN={class_counts.get('FN', 0)})")
        
        all_results.extend(results)
        cld_names[filename] = cld_name
    
    # Summary
    total_class = Counter([r['classification'] for r in all_results])
    console.print(f"\n  [bold]Total: {len(all_results)} edges[/bold]")
    console.print(f"  [bold]  TP: {total_class['TP']}, FP: {total_class['FP']}, FN: {total_class['FN']}[/bold]")
    
    # Analyze
    stats = analyze_direct_causal_by_classification(all_results)
    
    # Print tables and analysis
    print_classification_table(stats)
    print_discrimination_analysis(stats)
    print_cld_breakdown(all_results, cld_names)
    print_implications(stats)
    print_token_summary(stats)
    
    # Print proportion tests (statistical assumption testing)
    proportion_test_results = print_proportion_tests(stats)
    
    # Save proportion test results to JSON
    if proportion_test_results:
        import json
        output_file = OUTPUT_DIR / 'rq3_proportion_tests.json'
        with open(output_file, 'w') as f:
            json.dump(proportion_test_results, f, indent=2, default=str)
        console.print(f"\n[green]✓ Proportion test results saved to: {output_file}[/green]")
    
    # Export summary for thesis
    console.print("\n[bold cyan]═" * 90 + "[/bold cyan]")
    console.print("[bold cyan]THESIS VALUES (copy to generate_figures.py)[/bold cyan]")
    console.print("[bold cyan]═" * 90 + "[/bold cyan]\n")
    
    for cls in ['TP', 'FP', 'FN']:
        if cls in stats:
            s = stats[cls]
            console.print(f"'{cls}': {{'total': {s['total']}, 'found': {s['direct_causal_found']}, 'rate': {s['detection_rate']:.1f}, 'conf': {s['avg_confidence']:.2f}}},")
    
    console.print("\n[bold green]═" * 90 + "[/bold green]")
    console.print("[bold green]Analysis complete![/bold green]\n")


if __name__ == "__main__":
    main()
