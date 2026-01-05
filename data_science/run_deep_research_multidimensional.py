#!/usr/bin/env python3
"""
Run deep research with integrated multi-dimensional evidence assessment.

This script uses the enhanced PydanticAI deep research manager that automatically
scores edges on 5 dimensions after completing the research:
  1. Causal Strength (study design quality)
  2. Mechanistic Specificity (direct vs mediated)
  3. Construct Validity (right variables measured)
  4. Scope Relevance (population/context match)
  5. Theoretical Coherence (fits theory)
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from pydantic_ai_deepresearch_multidimensional import DeepResearchManager
from rich.console import Console

async def main():
    console = Console()
    console.print("[bold]Deep Research with Multi-Dimensional Assessment[/bold]\n")
    
    # Example edges to test
    test_edges = [
        "Group-level BMI → Norm BMI",
        "Physical Activity Level → Total Daily Energy Intake (TDEI)",
        "Body Mass Index (BMI) → Basal Metabolic Rate (BMR)"
    ]
    
    console.print(f"Testing {len(test_edges)} edges with multi-dimensional assessment\n")
    
    # Run deep research on each edge
    results = []
    manager = DeepResearchManager(max_iterations=2, minimum_evidence_count=3, max_sources_per_query=3)
    
    for i, edge in enumerate(test_edges, 1):
        console.print(f"\n{'='*80}")
        console.print(f"[bold cyan]EDGE {i}/{len(test_edges)}: {edge}[/bold cyan]")
        console.print(f"{'='*80}\n")
        
        try:
            verdict = await manager.run(edge)
            
            # Convert to dict for JSON serialization
            result_dict = verdict.model_dump()
            results.append(result_dict)
            
            # Display multi-dimensional scores
            if verdict.overall_quality_score:
                console.print("\n[bold green]═══ MULTI-DIMENSIONAL QUALITY SCORES ═══[/bold green]")
                console.print(f"  Causal Strength:           {verdict.causal_strength:.2f}/10")
                console.print(f"  Mechanistic Specificity:   {verdict.mechanistic_specificity:.2f}/10")
                console.print(f"  Construct Validity:        {verdict.construct_validity:.2f}/10")
                console.print(f"  Scope Relevance:           {verdict.scope_relevance:.2f}/10")
                console.print(f"  Theoretical Coherence:     {verdict.theoretical_coherence:.2f}/10")
                console.print(f"  [bold]OVERALL QUALITY SCORE:     {verdict.overall_quality_score:.2f}/10[/bold]")
                console.print(f"\n  Reasoning: {verdict.multidimensional_reasoning}")
            
        except Exception as e:
            console.print(f"[red]Error processing edge: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append({"edge": edge, "error": str(e)})
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"deep_research_multidimensional_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[bold green]✓ Results saved to: {output_file}[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
