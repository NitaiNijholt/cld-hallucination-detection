#!/usr/bin/env python3
"""
Test Deep Research on ACTUAL ground truth edges from RQ1 experiments.
This tests the system's ability to discriminate between TP, FP, TN, FN edges.
"""

import asyncio
import json
import pandas as pd
from pydantic_ai_deepresearch_orchestration import DeepResearchManager
from rich.console import Console
from rich.table import Table
import time
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# Reduce verbosity of httpx
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Ensure pydantic AI logger is visible
logging.getLogger('pydantic_ai_deepresearch_orchestration').setLevel(logging.INFO)

console = Console()

# Global telemetry tracker
class Telemetry:
    def __init__(self):
        self.total_api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.verdict_counts = {}
        self.per_edge_stats = []
        self.start_time = time.time()
        
    def add_edge_result(self, result, input_tokens, output_tokens):
        self.total_api_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        
        verdict = result.get('verdict', 'ERROR')
        self.verdict_counts[verdict] = self.verdict_counts.get(verdict, 0) + 1
        
        self.per_edge_stats.append({
            'edge': f"{result['source']} → {result['target']}",
            'classification': result['classification'],
            'CLD': result['CLD'],
            'verdict': verdict,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'duration': result['duration']
        })
    
    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            'total_edges_processed': len(self.per_edge_stats),
            'total_api_calls': self.total_api_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens,
            'verdict_counts': self.verdict_counts,
            'elapsed_time_seconds': round(elapsed, 1),
            'elapsed_time_minutes': round(elapsed / 60, 1),
            'per_edge_statistics': self.per_edge_stats
        }

TELEMETRY = Telemetry()

# Load ground truth edges from all 3 CLDs (CSV format)
edges_df = pd.read_csv('deep_research_all_validation_edges.csv')
console.print(f"[bold cyan]Loaded {len(edges_df)} edges from CSV[/bold cyan]")

# Convert to dict format grouped by classification
GROUND_TRUTH_EDGES = {}
for _, row in edges_df.iterrows():
    cls = row['Classification']
    if cls not in GROUND_TRUTH_EDGES:
        GROUND_TRUTH_EDGES[cls] = []
    GROUND_TRUTH_EDGES[cls].append({
        'CLD': row['CLD'],
        'Source': row['Source'],
        'Target': row['Target'],
        'Motivation': row['Motivation'],
        'Expected': row['Expected']
    })

async def test_edge(classification: str, edge: dict, manager: DeepResearchManager):
    """Test a single edge with Deep Research."""
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]Testing {classification}: {edge['Source']} → {edge['Target']}[/bold cyan]")
    console.print(f"[yellow]Expected:[/yellow] {edge['Expected']}")
    console.print(f"{'='*80}\n")
    
    logger.info("="*80)
    logger.info(f"TESTING EDGE: {classification}")
    logger.info(f"Source: {edge['Source']}")
    logger.info(f"Target: {edge['Target']}")
    logger.info(f"Classification: {classification}")
    logger.info(f"Expected: {edge['Expected']}")
    logger.info(f"Motivation: {edge['Motivation']}")
    logger.info("="*80)
    
    # Format as causal claim
    claim = f"{edge['Source']} causes {edge['Target']}. {edge['Motivation']}"
    
    logger.info(f"Formatted claim: {claim[:200]}...")
    
    start_time = time.time()
    
    try:
        # Run Deep Research
        verdict = await manager.run(claim)
        
        duration = time.time() - start_time
        
        # Extract results
        result = {
            "CLD": edge.get('CLD', 'Unknown'),
            "classification": classification,
            "source": edge['Source'],
            "target": edge['Target'],
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "direct_causal_found": verdict.direct_causal_evidence_found,
            "strongest_causal_evidence_confidence": verdict.strongest_causal_evidence_confidence,
            "strongest_causal_evidence_url": verdict.strongest_causal_evidence_url if verdict.strongest_causal_evidence_url else None,
            "strongest_causal_evidence_passage": verdict.strongest_causal_evidence_passage if verdict.strongest_causal_evidence_passage else None,
            "strongest_causal_evidence_study_type": verdict.strongest_causal_evidence_study_type if verdict.strongest_causal_evidence_study_type else None,
            "strongest_causal_evidence_summary": verdict.strongest_causal_evidence_summary if verdict.strongest_causal_evidence_summary else None,
            "evidence_count": len(verdict.evidence_urls),
            "evidence_urls": verdict.evidence_urls,
            "reasoning": verdict.judge_reasoning,  # Full reasoning
            "duration": f"{duration:.1f}s",
            "expected": edge['Expected']
        }
        
        # Print summary
        console.print(f"[bold green]✓ Verdict:[/bold green] {verdict.verdict.upper()} (confidence: {verdict.confidence}/10)")
        console.print(f"[bold]Direct Causal Evidence:[/bold] {verdict.direct_causal_evidence_found}")
        if verdict.direct_causal_evidence_found:
            console.print(f"[bold]Causal Confidence:[/bold] {verdict.strongest_causal_evidence_confidence}/10")
        console.print(f"[bold]Evidence Sources:[/bold] {len(verdict.evidence_urls)}")
        console.print(f"[bold]Duration:[/bold] {duration:.1f}s")
        
        logger.info(f"RESULT: verdict={verdict.verdict}, confidence={verdict.confidence}/10, direct_causal={verdict.direct_causal_evidence_found}")
        logger.info(f"  Duration: {duration:.1f}s")
        
        return result
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        logger.error(f"Error testing edge: {e}", exc_info=True)
        return {
            "classification": classification,
            "source": edge['Source'],
            "target": edge['Target'],
            "verdict": "ERROR",
            "confidence": 0,
            "direct_causal_found": False,
            "error": str(e),
            "duration": f"{time.time() - start_time:.1f}s"
        }


async def run_all_tests():
    """Run tests on all ground truth edges."""
    console.print("[bold blue]🔬 DEEP RESEARCH GROUND TRUTH VALIDATION TEST[/bold blue]")
    console.print("Testing discrimination ability on ACTUAL ground truth edges from RQ1\\n")
    
    logger.info("="*80)
    logger.info("DEEP RESEARCH GROUND TRUTH VALIDATION - SETUP")
    logger.info("="*80)
    logger.info("Sample edges to test:")
    for classification, edges in GROUND_TRUTH_EDGES.items():
        logger.info(f"  {classification}: {len(edges)} edges")
        for i, edge in enumerate(edges, 1):
            logger.info(f"    {i}. {edge['Source']} → {edge['Target']}")
    total_edges = sum(len(edges) for edges in GROUND_TRUTH_EDGES.values())
    logger.info(f"Total edges: {total_edges}")
    logger.info("="*80)
    
    # Create Deep Research manager
    manager = DeepResearchManager(
        max_iterations=1,
        minimum_evidence_count=2,
        max_sources_per_query=5
    )
    
    logger.info("Deep Research Manager configured:")
    logger.info(f"  max_iterations: 1")
    logger.info(f"  minimum_evidence_count: 2")
    logger.info(f"  max_sources_per_query: 5")
    logger.info("="*80)
    
    # Collect all edges
    all_tasks = []
    for classification, edges in GROUND_TRUTH_EDGES.items():
        for edge in edges:
            all_tasks.append((classification, edge))
    
    console.print(f"\\n[bold magenta]{'='*80}[/bold magenta]")
    console.print(f"[bold magenta]RUNNING ALL {len(all_tasks)} EDGES IN PARALLEL[/bold magenta]")
    console.print(f"[bold magenta]{'='*80}[/bold magenta]")
    
    # Run ALL edges in parallel
    console.print(f"\\n[cyan]Processing {len(all_tasks)} edges in parallel...[/cyan]")
    logger.info(f"Running {len(all_tasks)} edges in parallel using asyncio.gather()")
    
    tasks = [test_edge(classification, edge, manager) for classification, edge in all_tasks]
    all_results = await asyncio.gather(*tasks)
    
    logger.info(f"All {len(all_results)} edges completed")
    
    return all_results


def analyze_results(results):
    """Analyze results for discrimination ability."""
    console.print("\\n\\n[bold blue]📊 DISCRIMINATION ANALYSIS[/bold blue]")
    console.print("="*80)
    
    # Create summary table
    table = Table(title="Deep Research Results by Classification")
    table.add_column("Class", style="cyan")
    table.add_column("Source → Target", style="white")
    table.add_column("Verdict", style="green")
    table.add_column("Conf.", justify="right")
    table.add_column("Direct Causal", justify="center")
    table.add_column("Evidence", justify="right")
    
    for r in results:
        if r.get('verdict') == 'ERROR':
            table.add_row(
                r['classification'],
                f"{r['source'][:25]}... → {r['target'][:25]}...",
                "[red]ERROR[/red]",
                "-",
                "-",
                "-"
            )
        else:
            verdict_color = "green" if r['verdict'] in ['supported', 'partially_supported'] else "red"
            table.add_row(
                r['classification'],
                f"{r['source'][:25]}... → {r['target'][:25]}...",
                f"[{verdict_color}]{r['verdict'].upper()}[/{verdict_color}]",
                str(r['confidence']),
                "✓" if r['direct_causal_found'] else "✗",
                str(r['evidence_count'])
            )
    
    console.print(table)
    
    # Calculate discrimination metrics
    console.print("\\n[bold]Discrimination Metrics:[/bold]")
    
    # Group by classification
    by_class = {}
    for r in results:
        if r.get('verdict') != 'ERROR':
            cls = r['classification']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(r)
    
    # Calculate support scores (1=supported, 0.5=partial, 0=unsupported)
    def verdict_score(verdict):
        if verdict == 'supported':
            return 1.0
        elif verdict == 'partially_supported':
            return 0.5
        else:
            return 0.0
    
    console.print("\\n[bold cyan]Support Scores by Classification:[/bold cyan]")
    console.print("  (1.0 = supported, 0.5 = partially_supported, 0.0 = unsupported)")
    
    for cls in ['TP', 'FP', 'TN', 'FN']:
        if cls in by_class:
            scores = [verdict_score(r['verdict']) for r in by_class[cls]]
            mean_score = sum(scores) / len(scores) if scores else 0
            confidences = [r['confidence'] for r in by_class[cls]]
            mean_conf = sum(confidences) / len(confidences) if confidences else 0
            console.print(f"  {cls}: score={mean_score:.2f}, confidence={mean_conf:.1f}/10 (n={len(by_class[cls])})")
    
    # Key discrimination test: TP vs FP
    console.print("\\n[bold yellow]🎯 Key Discrimination Test: TP vs FP[/bold yellow]")
    if 'TP' in by_class and 'FP' in by_class:
        tp_scores = [verdict_score(r['verdict']) for r in by_class['TP']]
        fp_scores = [verdict_score(r['verdict']) for r in by_class['FP']]
        
        tp_mean = sum(tp_scores) / len(tp_scores)
        fp_mean = sum(fp_scores) / len(fp_scores)
        diff = tp_mean - fp_mean
        
        console.print(f"  TP mean support score: {tp_mean:.2f}")
        console.print(f"  FP mean support score: {fp_mean:.2f}")
        console.print(f"  Difference: {diff:.2f}")
        
        if diff >= 0.4:
            console.print(f"  [bold green]✅ STRONG discrimination (Δ ≥ 0.4)[/bold green]")
        elif diff >= 0.2:
            console.print(f"  [yellow]⚠️  MODERATE discrimination (0.2 ≤ Δ < 0.4)[/yellow]")
        elif diff > 0:
            console.print(f"  [bold red]❌ WEAK discrimination (0 < Δ < 0.2)[/bold red]")
        else:
            console.print(f"  [bold red]❌ NO discrimination (Δ ≤ 0)[/bold red]")
    
    # Save results
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"deep_research_results_{timestamp}_all_3_CLDs_107edges.json"
    output_path = os.path.abspath(output_file)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\\n[bold]Results saved to:[/bold] {output_path}")
    
    return by_class


async def main():
    """Main entry point."""
    start_time = time.time()
    
    # Run all tests
    results = await run_all_tests()
    
    # Analyze results
    analyze_results(results)
    
    total_time = time.time() - start_time
    console.print(f"\\n[bold]Total test duration:[/bold] {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    asyncio.run(main())

