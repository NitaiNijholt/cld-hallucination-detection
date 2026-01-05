#!/usr/bin/env python3
"""
Test Deep Research on Social Norms and Obesity Prevalence CLD edges.
Tests on TP, FP, and FN edges (excludes TN).
"""

import asyncio
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from pydantic_ai_deepresearch_orchestration import DeepResearchManager
from dataclasses import dataclass, field
import logging

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Telemetry:
    """Track telemetry across all edges."""
    edges_processed: int = 0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    verdicts: dict = field(default_factory=dict)
    
    def add_result(self, verdict: str, tokens: int, input_tokens: int = 0, output_tokens: int = 0):
        """Add a result to telemetry."""
        self.edges_processed += 1
        self.total_tokens += tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.verdicts[verdict] = self.verdicts.get(verdict, 0) + 1
    
    def print_status(self):
        """Print current telemetry status."""
        console.print("\n[bold cyan]📊 TELEMETRY STATUS[/bold cyan]")
        console.print(f"Edges processed: {self.edges_processed}")
        console.print(f"Total tokens: {self.total_tokens:,} (In: {self.total_input_tokens:,}, Out: {self.total_output_tokens:,})")
        console.print(f"Verdicts: {self.verdicts}")


async def test_edge(edge: dict, telemetry: Telemetry) -> dict:
    """Test a single edge with deep research."""
    try:
        # Create fresh manager for this edge
        manager = DeepResearchManager()
        
        # Construct claim
        claim = f"{edge['Source']} causes {edge['Target']}"
        
        console.print(f"\n[bold blue]{'='*80}[/bold blue]")
        console.print(f"[bold]Testing Edge:[/bold] {claim}")
        console.print(f"[bold]Classification:[/bold] {edge['Classification']}")
        console.print(f"[bold blue]{'='*80}[/bold blue]\n")
        
        # Run deep research
        verdict = await manager.run(claim)
        
        # Get actual token usage from manager
        actual_tokens = manager.total_tokens
        actual_api_calls = manager.agent_call_count
        
        # Estimate input/output split (rough approximation)
        estimated_input = int(actual_tokens * 0.98)  # ~98% input
        estimated_output = int(actual_tokens * 0.02)  # ~2% output
        
        # Add to telemetry
        telemetry.add_result(
            verdict.verdict,
            actual_tokens,
            estimated_input,
            estimated_output
        )
        
        # Create result dict
        result = {
            "CLD": edge['CLD'],
            "classification": edge['Classification'],
            "source": edge['Source'],
            "target": edge['Target'],
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            
            # Direct causal evidence
            "direct_causal_found": verdict.direct_causal_evidence_found or False,
            "strongest_causal_evidence_confidence": verdict.strongest_causal_evidence_confidence,
            "strongest_causal_evidence_url": verdict.strongest_causal_evidence_url,
            "strongest_causal_evidence_passage": verdict.strongest_causal_evidence_passage,
            "strongest_causal_evidence_study_type": verdict.strongest_causal_evidence_study_type,
            "strongest_causal_evidence_summary": verdict.strongest_causal_evidence_summary,
            
            # Full verdict details
            "judge_reasoning": verdict.judge_reasoning,
            "modified_claim": verdict.modified_claim,
            "evidence_urls": verdict.evidence_urls,
            "key_evidence_summaries": verdict.key_evidence_summaries,
            "limitations": verdict.limitations,
            "future_research": verdict.future_research,
            
            # Iteration history
            "iteration_history": [
                {
                    "iteration": hist.iteration_number,
                    "claim": hist.claim_tested,
                    "evidence_count": hist.evidence_count,
                    "high_quality_count": hist.high_quality_count,
                    "judgment": hist.judgment,
                    "judgment_confidence": hist.judgment_confidence
                }
                for hist in (verdict.iteration_history or [])
            ],
            
            # Actual token usage
            "total_tokens": actual_tokens,
            "api_calls": actual_api_calls,
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimated_output
        }
        
        console.print(f"\n[green]✓ Edge complete:[/green] {verdict.verdict} (confidence: {verdict.confidence}/10)")
        console.print(f"[green]  Tokens (ACTUAL): {actual_tokens:,} (API calls: {actual_api_calls})[/green]")
        console.print(f"[green]  Breakdown (approx): In:{estimated_input:,}, Out:{estimated_output:,}[/green]")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing edge {edge['Source']} → {edge['Target']}: {str(e)}")
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        
        return {
            "CLD": edge['CLD'],
            "classification": edge['Classification'],
            "source": edge['Source'],
            "target": edge['Target'],
            "verdict": "ERROR",
            "error": str(e)
        }


async def run_all_tests():
    """Run tests on all Social Norms CLD edges."""
    
    console.print("[bold green]🔬 DEEP RESEARCH: SOCIAL NORMS AND OBESITY PREVALENCE CLD[/bold green]")
    console.print("[bold green]" + "="*80 + "[/bold green]\n")
    
    # Load edges from CSV
    edges_df = pd.read_csv('deep_research_social_norms_edges.csv')
    edges = edges_df.to_dict('records')
    
    console.print(f"[cyan]Loaded {len(edges)} edges from Social Norms CLD:[/cyan]")
    console.print(f"  TP: {sum(1 for e in edges if e['Classification'] == 'TP')}")
    console.print(f"  FP: {sum(1 for e in edges if e['Classification'] == 'FP')}")
    console.print(f"  FN: {sum(1 for e in edges if e['Classification'] == 'FN')}")
    console.print(f"  TN: 0 (excluded)\n")
    
    # Estimate time and cost
    avg_tokens_per_edge = 250000  # Based on previous runs
    total_estimated_tokens = len(edges) * avg_tokens_per_edge
    estimated_cost = total_estimated_tokens / 1_000_000  # $1 per million tokens (rough estimate)
    estimated_minutes = len(edges) * 5  # ~5 minutes per edge
    
    console.print(f"[yellow]📊 Estimates:[/yellow]")
    console.print(f"  Tokens: ~{total_estimated_tokens:,} ({avg_tokens_per_edge:,} per edge)")
    console.print(f"  Cost: ~${estimated_cost:.2f}")
    console.print(f"  Duration: ~{estimated_minutes} minutes ({estimated_minutes/60:.1f} hours)\n")
    
    # Initialize telemetry
    telemetry = Telemetry()
    
    # Process edges in batches of 5 with delays
    results = []
    batch_size = 5
    
    for batch_start in range(0, len(edges), batch_size):
        batch_end = min(batch_start + batch_size, len(edges))
        batch = edges[batch_start:batch_end]
        
        console.print(f"\n[bold yellow]Processing batch {batch_start//batch_size + 1}/{(len(edges)-1)//batch_size + 1}[/bold yellow]")
        console.print(f"[yellow]Edges {batch_start+1}-{batch_end} of {len(edges)}[/yellow]\n")
        
        # Process batch in parallel
        batch_tasks = [test_edge(edge, telemetry) for edge in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        console.print(f"\n[green]Batch {batch_start//batch_size + 1} completed[/green]")
        telemetry.print_status()
        
        # Delay between batches (except after last batch)
        if batch_end < len(edges):
            console.print(f"\n[yellow]⏸️  Waiting 5 seconds before next batch...[/yellow]")
            await asyncio.sleep(5)
    
    console.print(f"\n[bold green]All {len(edges)} edges completed[/bold green]\n")
    
    # Final analysis
    analyze_results(results, telemetry)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"deep_research_results_social_norms_{timestamp}_{len(edges)}edges.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "cld": "Social_norms_and_obesity_prevalence",
            "timestamp": timestamp,
            "total_edges": len(edges),
            "results": results,
            "telemetry": {
                "total_tokens": telemetry.total_tokens,
                "total_input_tokens": telemetry.total_input_tokens,
                "total_output_tokens": telemetry.total_output_tokens,
                "verdicts": telemetry.verdicts
            }
        }, f, indent=2)
    
    console.print(f"[bold green]Results saved to: {output_file}[/bold green]\n")
    
    return results


def analyze_results(results: list, telemetry: Telemetry):
    """Analyze and display results."""
    
    console.print("\n[bold cyan]📊 DISCRIMINATION ANALYSIS[/bold cyan]")
    console.print("="*80 + "\n")
    
    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class", style="cyan", width=8)
    table.add_column("Source → Target", width=35)
    table.add_column("Verdict", width=15)
    table.add_column("Conf.", justify="right", width=7)
    table.add_column("Tokens", justify="right", width=10)
    
    for result in results[:20]:  # Show first 20
        table.add_row(
            result['classification'],
            f"{result['source'][:17]}... → {result['target'][:17]}..." if len(result['source']) > 17 else f"{result['source']} → {result['target']}",
            result['verdict'].upper() if result['verdict'] != 'ERROR' else 'ERROR',
            str(result.get('confidence', '-')),
            f"{result.get('total_tokens', 0):,}" if result.get('total_tokens') else '-'
        )
    
    console.print(table)
    
    # Discrimination metrics
    console.print("\n[bold]Discrimination Metrics:[/bold]\n")
    
    # Support scores by classification
    console.print("Support Scores by Classification:")
    console.print("  (1.0 = supported, 0.5 = partially_supported, 0.0 = unsupported)")
    
    for cls in ['TP', 'FN', 'FP']:
        cls_results = [r for r in results if r['classification'] == cls and r['verdict'] != 'ERROR']
        if cls_results:
            scores = []
            for r in cls_results:
                if r['verdict'] == 'supported':
                    scores.append(1.0)
                elif r['verdict'] == 'partially_supported':
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            
            avg_score = sum(scores) / len(scores) if scores else 0
            avg_conf = sum(r.get('confidence', 0) for r in cls_results) / len(cls_results) if cls_results else 0
            
            console.print(f"  {cls}: score={avg_score:.2f}, confidence={avg_conf:.1f}/10 (n={len(cls_results)})")
    
    # Token usage summary
    console.print(f"\n[bold yellow]💰 TOKEN USAGE SUMMARY:[/bold yellow]")
    console.print(f"  Total edges: {len(results)}")
    console.print(f"  Total tokens: {telemetry.total_tokens:,}")
    console.print(f"    Input tokens: {telemetry.total_input_tokens:,}")
    console.print(f"    Output tokens: {telemetry.total_output_tokens:,}")
    console.print(f"  Avg tokens per edge: {telemetry.total_tokens // len(results):,}" if results else "  N/A")


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    results = asyncio.run(run_all_tests())
    
    duration = time.time() - start_time
    console.print(f"\n[bold]Total test duration: {duration:.1f}s ({duration/60:.1f} minutes)[/bold]")

