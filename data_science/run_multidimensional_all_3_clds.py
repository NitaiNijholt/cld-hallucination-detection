#!/usr/bin/env python3
"""
Run multidimensional deep research on edges from a single CLD or all 3 CLDs.

Usage:
  python run_multidimensional_all_3_clds.py deep_research_social_norms_edges.json
  python run_multidimensional_all_3_clds.py deep_research_depressive_symptoms_edges.json
  python run_multidimensional_all_3_clds.py deep_research_older_persons_edges.json
  python run_multidimensional_all_3_clds.py --all  # Run all 3 CLDs (285 edges)

This will process ALL edges (TP, FP, FN) in batches of 5.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from pydantic_ai_deepresearch_multidimensional import DeepResearchManager
from rich.console import Console

def load_edges_from_file(file_path):
    """Load edges from a single CLD JSON file."""
    console = Console()
    all_edges = []
    
    try:
        with open(file_path) as f:
            data = json.load(f)
            for edge in data.get('edges', []):
                all_edges.append({
                    'cld': data['cld_name'],
                    'classification': edge['Classification'],
                    'source': edge['Source'],
                    'target': edge['Target'],
                    'motivation': edge.get('Motivation', ''),
                    'claim': f"{edge['Source']} causes {edge['Target']}. {edge.get('Motivation', '')}"
                })
        console.print(f"[green]✓ Loaded {data['cld_name']}: {len(all_edges)} edges (TP:{data['TP']}, FP:{data['FP']}, FN:{data['FN']})[/green]")
        return all_edges, data['cld_name']
    except Exception as e:
        console.print(f"[red]✗ Error loading {file_path}: {e}[/red]")
        return [], None


def load_edges_from_all_3_clds():
    """Load ALL edges (TP, FP, FN) from all 3 CLDs."""
    console = Console()
    all_edges = []
    cld_files = [
        'deep_research_social_norms_edges.json',
        'deep_research_depressive_symptoms_edges.json',
        'deep_research_older_persons_edges.json'
    ]
    
    for file_path in cld_files:
        edges, _ = load_edges_from_file(file_path)
        all_edges.extend(edges)
    
    return all_edges, "ALL_3_CLDs"


async def main():
    console = Console()
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        console.print("[red]Error: No input file specified![/red]")
        console.print("\n[bold]Usage:[/bold]")
        console.print("  python run_multidimensional_all_3_clds.py deep_research_social_norms_edges.json")
        console.print("  python run_multidimensional_all_3_clds.py deep_research_depressive_symptoms_edges.json")
        console.print("  python run_multidimensional_all_3_clds.py deep_research_older_persons_edges.json")
        console.print("  python run_multidimensional_all_3_clds.py --all")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    
    # Load edges based on argument
    if input_arg == "--all":
        console.print("[bold]Multi-Dimensional Deep Research on ALL 285 edges from 3 CLDs[/bold]\n")
        all_edges, cld_name = load_edges_from_all_3_clds()
    else:
        console.print(f"[bold]Multi-Dimensional Deep Research on {Path(input_arg).stem}[/bold]\n")
        all_edges, cld_name = load_edges_from_file(input_arg)
    
    if not all_edges:
        console.print("[red]No edges loaded! Exiting.[/red]")
        return
    
    # Count by CLD and classification
    console.print("\n[bold cyan]Edge Summary:[/bold cyan]")
    cld_names = list(set(e['cld'] for e in all_edges))
    for cld in cld_names:
        cld_edges = [e for e in all_edges if e['cld'] == cld]
        tp_count = len([e for e in cld_edges if e['classification'] == 'TP'])
        fp_count = len([e for e in cld_edges if e['classification'] == 'FP'])
        fn_count = len([e for e in cld_edges if e['classification'] == 'FN'])
        console.print(f"  {cld}: {len(cld_edges)} total (TP: {tp_count}, FP: {fp_count}, FN: {fn_count})")
    
    total_tp = len([e for e in all_edges if e['classification'] == 'TP'])
    total_fp = len([e for e in all_edges if e['classification'] == 'FP'])
    total_fn = len([e for e in all_edges if e['classification'] == 'FN'])
    console.print(f"\n[bold]Total: {len(all_edges)} edges (TP: {total_tp}, FP: {total_fp}, FN: {total_fn})[/bold]\n")
    
    # Process edges in batches of 5 (parallel within batch, sequential between batches)
    BATCH_SIZE = 5
    all_results = []
    
    for batch_start in range(0, len(all_edges), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_edges))
        batch = all_edges[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(all_edges) + BATCH_SIZE - 1) // BATCH_SIZE
        
        console.print(f"\n[bold yellow]📦 BATCH {batch_num}/{total_batches} - Processing edges {batch_start+1} to {batch_end}[/bold yellow]")
        
        # Run batch in parallel (each edge gets its own fresh manager instance)
        async def process_edge(edge_info, edge_num):
            console.print(f"\n{'='*80}")
            console.print(f"[bold cyan]EDGE {edge_num}/{len(all_edges)}: {edge_info['cld']} - {edge_info['classification']}[/bold cyan]")
            console.print(f"[bold]{edge_info['claim']}[/bold]")
            console.print(f"{'='*80}\n")
            
            try:
                # Fresh manager for each edge
                manager = DeepResearchManager(max_iterations=2, minimum_evidence_count=3, max_sources_per_query=5)
                verdict = await manager.run(edge_info['claim'])
                
                # Convert to dict and add metadata
                result_dict = verdict.model_dump()
                result_dict['cld'] = edge_info['cld']
                result_dict['ground_truth_classification'] = edge_info['classification']
                result_dict['source_variable'] = edge_info['source']
                result_dict['target_variable'] = edge_info['target']
                
                # Display multi-dimensional scores
                if verdict.overall_quality_score:
                    console.print("\n[bold green]═══ MULTI-DIMENSIONAL QUALITY SCORES ═══[/bold green]")
                    console.print(f"  Causal Strength:           {verdict.causal_strength:.2f}/10")
                    console.print(f"  Mechanistic Specificity:   {verdict.mechanistic_specificity:.2f}/10")
                    console.print(f"  Construct Validity:        {verdict.construct_validity:.2f}/10")
                    console.print(f"  Scope Relevance:           {verdict.scope_relevance:.2f}/10")
                    console.print(f"  Theoretical Coherence:     {verdict.theoretical_coherence:.2f}/10")
                    console.print(f"  [bold]OVERALL QUALITY SCORE:     {verdict.overall_quality_score:.2f}/10[/bold]")
                    console.print(f"\n  Ground Truth: {edge_info['classification']}")
                
                return result_dict
                
            except Exception as e:
                console.print(f"[red]Error processing edge: {e}[/red]")
                import traceback
                traceback.print_exc()
                return {
                    "cld": edge_info['cld'],
                    "ground_truth_classification": edge_info['classification'],
                    "source_variable": edge_info['source'],
                    "target_variable": edge_info['target'],
                    "original_claim": edge_info['claim'],
                    "error": str(e)
                }
        
        # Create tasks for this batch
        tasks = [
            process_edge(edge_info, batch_start + i + 1)
            for i, edge_info in enumerate(batch)
        ]
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        console.print(f"\n[bold green]✓ Batch {batch_num}/{total_batches} completed[/bold green]")
        
        # Small delay between batches to avoid rate limiting
        if batch_end < len(all_edges):
            console.print(f"[dim]Waiting 5 seconds before next batch...[/dim]")
            await asyncio.sleep(5)
    
    results = all_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"multidimensional_{cld_name}_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[bold green]✓ Results saved to: {output_file}[/bold green]")
    
    # Print summary statistics
    console.print("\n[bold cyan]═══ SUMMARY STATISTICS ═══[/bold cyan]")
    successful = [r for r in results if 'error' not in r and r.get('overall_quality_score') is not None]
    console.print(f"  Successfully processed: {len(successful)}/{len(all_edges)} edges")
    
    if successful:
        tp_scores = [r['overall_quality_score'] for r in successful if r['ground_truth_classification'] == 'TP']
        fp_scores = [r['overall_quality_score'] for r in successful if r['ground_truth_classification'] == 'FP']
        
        if tp_scores:
            console.print(f"\n  [green]TP edges (n={len(tp_scores)}):[/green]")
            console.print(f"    Average Overall Score: {sum(tp_scores)/len(tp_scores):.2f}/10")
            console.print(f"    Range: {min(tp_scores):.2f} - {max(tp_scores):.2f}")
        
        if fp_scores:
            console.print(f"\n  [yellow]FP edges (n={len(fp_scores)}):[/yellow]")
            console.print(f"    Average Overall Score: {sum(fp_scores)/len(fp_scores):.2f}/10")
            console.print(f"    Range: {min(fp_scores):.2f} - {max(fp_scores):.2f}")
        
        if tp_scores and fp_scores:
            gap = sum(tp_scores)/len(tp_scores) - sum(fp_scores)/len(fp_scores)
            console.print(f"\n  [bold magenta]TP vs FP discrimination gap: {gap:+.2f} points[/bold magenta]")
            if gap > 0:
                console.print(f"  [green]✓ TP scores higher than FP (good discrimination!)[/green]")
            else:
                console.print(f"  [yellow]⚠ FP scores higher than or equal to TP[/yellow]")

if __name__ == "__main__":
    asyncio.run(main())
