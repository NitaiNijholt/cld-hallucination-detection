#!/usr/bin/env python3
"""
Run ALL 285 edges (3 CLDs) using parallel batch processing with multidimensional deep research.
Based on test_5_edges_batch_multidimensional.py
"""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from pydantic_ai_deepresearch_multidimensional import DeepResearchManager
from rich.console import Console

console = Console()

# Load edges from all 3 CLDs
def load_all_edges():
    """Load edges from the 3 CLD files."""
    edge_files = [
        "deep_research_social_norms_edges.json",
        "deep_research_depressive_symptoms_edges.json",
        "deep_research_older_persons_edges.json"
    ]
    
    all_edges = []
    
    for file_path in edge_files:
        console.print(f"[cyan]Loading {file_path}...[/cyan]")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        cld_name = data['cld_name']
        edges = data['edges']
        
        console.print(f"  ✓ {cld_name}: {len(edges)} edges (TP:{data['TP']}, FP:{data['FP']}, FN:{data['FN']})")
        
        all_edges.extend(edges)
    
    console.print(f"\n[bold green]Total edges loaded: {len(all_edges)}[/bold green]\n")
    return all_edges


async def test_edge(edge: dict, edge_num: int, total_edges: int):
    """Test a single edge with Deep Research."""
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]Edge {edge_num}/{total_edges} - {edge['Classification']}: {edge['Source']} → {edge['Target']}[/bold cyan]")
    console.print(f"[yellow]CLD:[/yellow] {edge['CLD']}")
    console.print(f"[yellow]Expected:[/yellow] {edge['Expected']}")
    console.print(f"{'='*80}\n")
    
    # Format as causal claim
    claim = f"{edge['Source']} causes {edge['Target']}. {edge['Motivation']}"
    
    start_time = time.time()
    
    try:
        # CRITICAL: Fresh manager instance for each edge to avoid state contamination
        manager = DeepResearchManager(
            max_iterations=2,
            minimum_evidence_count=3,
            max_sources_per_query=5
        )
        
        verdict = await manager.run(claim)
        
        elapsed_time = time.time() - start_time
        
        # Extract all fields including telemetry
        result = {
            "edge_num": edge_num,
            "classification": edge['Classification'],
            "cld": edge['CLD'],
            "source": edge['Source'],
            "target": edge['Target'],
            "expected": edge['Expected'],
            "elapsed_seconds": round(elapsed_time, 2),
            
            # Telemetry
            "total_tokens": verdict.total_tokens,
            "total_api_calls": verdict.total_api_calls,
            "research_elapsed_time_seconds": verdict.elapsed_time_seconds,
            "iterations_completed": verdict.iterations_completed,
            
            # Core verdict
            "original_claim": verdict.original_claim,
            "modified_claim": verdict.modified_claim,
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "judge_reasoning": verdict.judge_reasoning,
            
            # Binary judgment
            "is_direct_causal": verdict.is_direct_causal,
            "mechanistic_specificity": verdict.mechanistic_specificity,
            "direct_causal_reasoning": verdict.direct_causal_reasoning,
            "direct_causal_confidence": verdict.direct_causal_confidence,
            
            # Snippet consensus
            "snippet_consensus_support": verdict.snippet_consensus_support,
            "snippet_consensus_quality": verdict.snippet_consensus_quality,
            "counter_evidence_found": verdict.counter_evidence_found,
            "counter_evidence_summary": verdict.counter_evidence_summary,
            "alternative_mechanisms_found": verdict.alternative_mechanisms_found,
            "alternative_mechanisms_summary": verdict.alternative_mechanisms_summary,
            
            # Article consensus
            "consensus_mechanism": verdict.consensus_mechanism,
            "consensus_effect": verdict.consensus_effect,
            "consensus_quality": verdict.consensus_quality,
            
            # Support scores
            "average_snippet_support_score": verdict.average_snippet_support_score,
            "average_article_support_score": verdict.average_article_support_score,
            "average_combined_support_score": verdict.average_combined_support_score,
            
            # Strongest evidence
            "strongest_causal_quote": verdict.strongest_causal_quote,
            "strongest_evidence_url": verdict.strongest_evidence_url,
            
            # Multi-dimensional scores
            "evidence_quality": verdict.evidence_quality,
            "mechanistic_specificity": verdict.mechanistic_specificity,
            "construct_validity": verdict.construct_validity,
            "scope_relevance": verdict.scope_relevance,
            "theoretical_coherence": verdict.theoretical_coherence,
            "overall_quality_score": verdict.overall_quality_score,
            
            "success": True,
            "error": None
        }
        
        console.print(f"\n[bold green]✓ COMPLETED ({elapsed_time:.1f}s):[/bold green] {edge['Source']} → {edge['Target']}")
        console.print(f"  Verdict: {verdict.verdict} (Conf: {verdict.confidence}/10)")
        console.print(f"  Direct Causal: {verdict.is_direct_causal} (Mech Spec: {verdict.mechanistic_specificity}/10)")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"[red]✗ ERROR ({elapsed_time:.1f}s):[/red] {edge['Source']} → {edge['Target']}: {str(e)}")
        
        return {
            "edge_num": edge_num,
            "classification": edge['Classification'],
            "cld": edge['CLD'],
            "source": edge['Source'],
            "target": edge['Target'],
            "expected": edge['Expected'],
            "elapsed_seconds": round(elapsed_time, 2),
            "success": False,
            "error": str(e)
        }


async def run_parallel_tests():
    """Run all edges in parallel."""
    all_edges = load_all_edges()
    
    console.print(f"\n{'='*80}")
    console.print(f"[bold magenta]RUNNING ALL {len(all_edges)} EDGES IN PARALLEL[/bold magenta]")
    console.print(f"[bold magenta]Using multidimensional deep research with full telemetry[/bold magenta]")
    console.print(f"{'='*80}\n")
    
    # Run all edges in parallel
    tasks = [
        test_edge(edge, i + 1, len(all_edges))
        for i, edge in enumerate(all_edges)
    ]
    
    results = await asyncio.gather(*tasks)
    
    return results


def analyze_results(results):
    """Analyze and display results."""
    console.print(f"\n\n{'='*80}")
    console.print("[bold blue]📊 RESULTS ANALYSIS[/bold blue]")
    console.print(f"{'='*80}\n")
    
    # Success/failure
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    console.print(f"✓ Successful: {len(successful)}/{len(results)}")
    console.print(f"✗ Failed: {len(failed)}/{len(results)}")
    
    # By CLD
    console.print(f"\n[bold cyan]By CLD:[/bold cyan]")
    for cld in ["Social_norms_and_obesity_prevalence", 
                "Depressive_symptoms_in_response_to_a_stressor_in_healthy_adults",
                "older_persons_emergency_department_visits_and_interactionstitled_spreadsheet"]:
        cld_results = [r for r in successful if r['cld'] == cld]
        if cld_results:
            cld_name = cld.replace("_", " ").title()
            console.print(f"  {cld_name}: {len(cld_results)} edges")
    
    # By classification
    console.print(f"\n[bold cyan]By Classification:[/bold cyan]")
    for cls in ["TP", "FP", "FN"]:
        cls_results = [r for r in successful if r['classification'] == cls]
        console.print(f"  {cls}: {len(cls_results)} edges")
    
    # Telemetry summary
    console.print(f"\n[bold cyan]Telemetry Summary:[/bold cyan]")
    if successful:
        total_tokens = sum(r.get('total_tokens', 0) for r in successful if r.get('total_tokens'))
        total_api_calls = sum(r.get('total_api_calls', 0) for r in successful if r.get('total_api_calls'))
        total_research_time = sum(r.get('research_elapsed_time_seconds', 0) for r in successful if r.get('research_elapsed_time_seconds'))
        
        console.print(f"  Total Tokens: {total_tokens:,}")
        console.print(f"  Total API Calls: {total_api_calls}")
        console.print(f"  Total Research Time: {total_research_time/60:.1f} min ({total_research_time:.1f}s)")
        console.print(f"  Avg per Edge: {total_tokens/len(successful):,.0f} tokens, {total_api_calls/len(successful):.1f} calls, {total_research_time/len(successful):.1f}s")
    
    console.print(f"\n{'='*80}\n")


async def main():
    """Main entry point."""
    console.print("\n[bold]RUNNING ALL 285 EDGES WITH MULTIDIMENSIONAL DEEP RESEARCH[/bold]\n")
    
    start_time = time.time()
    
    # Run tests
    results = await run_parallel_tests()
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path("/home/nitai/code/causalix.ai/data_science/test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"all_285_edges_multidim_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze
    analyze_results(results)
    
    console.print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    console.print(f"💾 Results saved to: {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())






