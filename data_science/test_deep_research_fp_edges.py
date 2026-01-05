#!/usr/bin/env python3
"""
Test Deep Research on ACTUAL ground truth edges from RQ1 experiments with full telemetry.
This tests the system's ability to discriminate between TP, FP, TN, FN edges.
"""

import asyncio
import json
import pandas as pd
from pydantic_ai_deepresearch_orchestration import DeepResearchManager
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import time
import logging
import sys
from datetime import datetime

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
    
    def print_status(self):
        """Print current status"""
        console.print(f"\n[bold cyan]📊 TELEMETRY STATUS[/bold cyan]")
        console.print(f"  Edges processed: {len(self.per_edge_stats)}")
        console.print(f"  Total tokens: {self.total_tokens:,} (In: {self.total_input_tokens:,}, Out: {self.total_output_tokens:,})")
        console.print(f"  Verdicts: {self.verdict_counts}")

TELEMETRY = Telemetry()

# Load FALSE POSITIVE edges from Older Persons CLD only (CSV format)
edges_df = pd.read_csv('deep_research_fp_edges_older_persons_only.csv')
console.print(f"[bold cyan]Loaded {len(edges_df)} FP edges from Older Persons CLD[/bold cyan]")

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

async def test_edge(classification: str, edge: dict, edge_num: int, total_edges: int):
    """Test a single edge with Deep Research."""
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]Edge {edge_num}/{total_edges} - {classification}: {edge['Source']} → {edge['Target']}[/bold cyan]")
    console.print(f"[yellow]Expected:[/yellow] {edge['Expected']}")
    console.print(f"{'='*80}\n")
    
    logger.info("="*80)
    logger.info(f"TESTING EDGE {edge_num}/{total_edges}: {classification}")
    logger.info(f"CLD: {edge.get('CLD', 'Unknown')}")
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
        # CRITICAL: Create a fresh manager instance for each edge to avoid state contamination!
        # When edges run in parallel, sharing the same manager causes massive token bloat
        # because search_results and evidence_scores accumulate across all parallel edges.
        manager = DeepResearchManager(
            max_iterations=1,
            minimum_evidence_count=2,
            max_sources_per_query=5
        )
        
        # Track tokens before edge run
        tokens_before = manager.total_tokens
        agent_calls_before = manager.agent_call_count
        
        # Run Deep Research
        verdict = await manager.run(claim)
        
        duration = time.time() - start_time
        
        # Get ACTUAL token usage from manager (not estimates!)
        total_tokens = manager.total_tokens - tokens_before
        agent_calls = manager.agent_call_count - agent_calls_before
        
        # Note: We don't have input/output breakdown per edge, only total
        # Approximate split based on typical 98/2 ratio
        input_tokens = int(total_tokens * 0.98)
        output_tokens = int(total_tokens * 0.02)
        
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
            "reasoning": verdict.judge_reasoning,
            "duration": f"{duration:.1f}s",
            "expected": edge['Expected'],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        # Add to telemetry
        TELEMETRY.add_edge_result(result, input_tokens, output_tokens)
        
        # Print summary
        console.print(f"[bold green]✓ Verdict:[/bold green] {verdict.verdict.upper()} (confidence: {verdict.confidence}/10)")
        console.print(f"[bold]Direct Causal Evidence:[/bold] {verdict.direct_causal_evidence_found}")
        if verdict.direct_causal_evidence_found:
            console.print(f"[bold]Causal Confidence:[/bold] {verdict.strongest_causal_evidence_confidence}/10")
        console.print(f"[bold]Evidence Sources:[/bold] {len(verdict.evidence_urls)}")
        console.print(f"[bold]Duration:[/bold] {duration:.1f}s")
        console.print(f"[bold]Tokens (ACTUAL):[/bold] {total_tokens:,} (API calls: {agent_calls})")
        console.print(f"[dim]  Breakdown (approx): In:{input_tokens:,}, Out:{output_tokens:,}[/dim]")
        
        logger.info(f"RESULT: verdict={verdict.verdict}, confidence={verdict.confidence}/10, direct_causal={verdict.direct_causal_evidence_found}")
        logger.info(f"  Duration: {duration:.1f}s, Tokens: {total_tokens:,} (actual), API calls: {agent_calls}")
        
        return result
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        logger.error(f"Error testing edge: {e}", exc_info=True)
        result = {
            "CLD": edge.get('CLD', 'Unknown'),
            "classification": classification,
            "source": edge['Source'],
            "target": edge['Target'],
            "verdict": "ERROR",
            "confidence": 0,
            "direct_causal_found": False,
            "error": str(e),
            "duration": f"{time.time() - start_time:.1f}s",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        TELEMETRY.add_edge_result(result, 0, 0)
        return result


async def run_all_tests():
    """Run tests on FALSE POSITIVE (FP) edges from Older Persons CLD in batches of 5."""
    console.print("[bold blue]🔬 DEEP RESEARCH FALSE POSITIVE (FP) EDGE TEST - OLDER PERSONS CLD[/bold blue]")
    console.print("Testing FALSE POSITIVE edges from Older Persons Emergency Department CLD\n")
    
    logger.info("="*80)
    logger.info("DEEP RESEARCH GROUND TRUTH VALIDATION - SETUP")
    logger.info("="*80)
    logger.info("Sample edges to test:")
    for classification, edges in GROUND_TRUTH_EDGES.items():
        logger.info(f"  {classification}: {len(edges)} edges")
    total_edges = sum(len(edges) for edges in GROUND_TRUTH_EDGES.values())
    logger.info(f"Total edges: {total_edges}")
    logger.info("="*80)
    
    logger.info("Deep Research configuration (each edge gets fresh manager):")
    logger.info(f"  max_iterations: 1")
    logger.info(f"  minimum_evidence_count: 2")
    logger.info(f"  max_sources_per_query: 5")
    logger.info(f"  batch_size: 5")
    logger.info(f"  search_limit_per_query: 5")
    logger.info("="*80)
    
    # Collect all edges
    all_tasks = []
    for classification, edges in GROUND_TRUTH_EDGES.items():
        for edge in edges:
            all_tasks.append((classification, edge))
    
    estimated_tokens = len(all_tasks) * 242000  # ~242K per edge based on TP/FN run
    estimated_cost = estimated_tokens / 1_000_000  # $1 per million tokens
    estimated_hours = (len(all_tasks) * 5.3 / 60) / 5  # 5.3 min/edge with batch of 5 in parallel
    
    console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
    console.print(f"[bold magenta]RUNNING ALL {len(all_tasks)} FP EDGES (~242K tokens/edge, ~{estimated_tokens/1_000_000:.1f}M total, ~{estimated_hours:.1f} hrs)[/bold magenta]")
    console.print(f"[bold magenta]Configuration: Search limit=5, Max articles=3, Batch size=5[/bold magenta]")
    console.print(f"[bold magenta]Expected cost: ~${estimated_cost:.2f}[/bold magenta]")
    console.print(f"[bold magenta]{'='*80}[/bold magenta]\n")
    
    # Process edges in batches of 5
    BATCH_SIZE = 5
    all_results = []
    
    for batch_start in range(0, len(all_tasks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_tasks))
        batch = all_tasks[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(all_tasks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        console.print(f"\n[bold yellow]📦 BATCH {batch_num}/{total_batches} - Processing edges {batch_start+1} to {batch_end}[/bold yellow]")
        logger.info(f"Starting batch {batch_num}/{total_batches}")
        
        # Run batch in parallel (each edge gets its own fresh manager instance)
        tasks = [
            test_edge(classification, edge, batch_start + i + 1, len(all_tasks))
            for i, (classification, edge) in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        logger.info(f"Batch {batch_num}/{total_batches} completed")
        TELEMETRY.print_status()
        
        # Small delay between batches to avoid rate limiting
        if batch_end < len(all_tasks):
            console.print(f"\n[dim]Waiting 5 seconds before next batch...[/dim]")
            await asyncio.sleep(5)
    
    logger.info(f"All {len(all_results)} edges completed")
    
    return all_results


def analyze_results(results):
    """Analyze results for discrimination ability."""
    console.print("\n\n[bold blue]📊 DISCRIMINATION ANALYSIS[/bold blue]")
    console.print("="*80)
    
    # Create summary table (first 20 results to avoid clutter)
    table = Table(title="Deep Research Results (First 20)")
    table.add_column("Class", style="cyan")
    table.add_column("Source → Target", style="white")
    table.add_column("Verdict", style="green")
    table.add_column("Conf.", justify="right")
    table.add_column("Tokens", justify="right")
    
    for i, r in enumerate(results[:20]):
        if r.get('verdict') == 'ERROR':
            table.add_row(
                r['classification'],
                f"{r['source'][:20]}... → {r['target'][:20]}...",
                "[red]ERROR[/red]",
                "-",
                "0"
            )
        else:
            verdict_color = "green" if r['verdict'] in ['supported', 'partially_supported'] else "red"
            table.add_row(
                r['classification'],
                f"{r['source'][:20]}... → {r['target'][:20]}...",
                f"[{verdict_color}]{r['verdict'].upper()}[/{verdict_color}]",
                str(r['confidence']),
                f"{r['total_tokens']:,}"
            )
    
    console.print(table)
    
    # Calculate discrimination metrics
    console.print("\n[bold]Discrimination Metrics:[/bold]")
    
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
    
    console.print("\n[bold cyan]Support Scores by Classification:[/bold cyan]")
    console.print("  (1.0 = supported, 0.5 = partially_supported, 0.0 = unsupported)")
    
    for cls in ['TP', 'FP', 'TN', 'FN']:
        if cls in by_class:
            scores = [verdict_score(r['verdict']) for r in by_class[cls]]
            mean_score = sum(scores) / len(scores) if scores else 0
            confidences = [r['confidence'] for r in by_class[cls]]
            mean_conf = sum(confidences) / len(confidences) if confidences else 0
            console.print(f"  {cls}: score={mean_score:.2f}, confidence={mean_conf:.1f}/10 (n={len(by_class[cls])})")
    
    # Get telemetry summary
    telemetry_summary = TELEMETRY.get_summary()
    
    # Print telemetry
    console.print("\n[bold magenta]💰 TOKEN USAGE SUMMARY:[/bold magenta]")
    console.print(f"  Total edges: {telemetry_summary['total_edges_processed']}")
    console.print(f"  Total tokens: {telemetry_summary['total_tokens']:,}")
    console.print(f"    Input tokens: {telemetry_summary['total_input_tokens']:,}")
    console.print(f"    Output tokens: {telemetry_summary['total_output_tokens']:,}")
    console.print(f"  Total duration: {telemetry_summary['elapsed_time_minutes']:.1f} minutes")
    console.print(f"  Avg tokens per edge: {telemetry_summary['total_tokens'] // max(1, telemetry_summary['total_edges_processed']):,}")
    
    # Save results with telemetry
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"deep_research_results_FP_{timestamp}_older_persons_CLD_{len(results)}edges.json"
    output_path = os.path.abspath(output_file)
    
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "total_edges": len(results),
            "test_configuration": {
                "max_iterations": 1,
                "minimum_evidence_count": 2,
                "max_sources_per_query": 5,
                "batch_size": 10
            }
        },
        "telemetry": telemetry_summary,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    console.print(f"\n[bold]Results saved to:[/bold] {output_path}")
    
    return by_class


async def main():
    """Main entry point."""
    start_time = time.time()
    
    # Run all tests
    results = await run_all_tests()
    
    # Analyze results
    analyze_results(results)
    
    total_time = time.time() - start_time
    console.print(f"\n[bold]Total test duration:[/bold] {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    asyncio.run(main())
