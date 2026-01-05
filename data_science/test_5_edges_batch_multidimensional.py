#!/usr/bin/env python3
"""
Test 5 edges using parallel batch processing with the NEW multidimensional deep research.
Based on test_deep_research_ground_truth.py framework.
"""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from pydantic_ai_deepresearch_multidimensional import DeepResearchManager
from rich.console import Console

console = Console()

# 5 test edges: 2 TP, 2 FP, 1 FN
GROUND_TRUTH_EDGES = {
    "TP": [
        {
            "CLD": "Social_Norms",
            "Source": "Group-level BMI",
            "Target": "Norm BMI",
            "Motivation": "Group-level BMI causes Norm BMI to change in a direct causal relationship. As the average BMI within a group increases, the perceived normative or acceptable BMI (Norm BMI) tends to rise, as group norms adjust to what is commonly observed. Conversely, if Group-level BMI decreases, the Norm BMI typically shifts downward as well.",
            "Expected": "Supported - direct effect"
        },
        {
            "CLD": "Energy_Balance",
            "Source": "Physical Activity Level",
            "Target": "Total Daily Energy Expenditure (TDEE)",
            "Motivation": "Physical Activity Level causes Total Daily Energy Expenditure (TDEE) to change in a direct causal relationship. Increasing Physical Activity Level directly raises TDEE because more energy is required for movement and exercise. Conversely, decreasing Physical Activity Level directly lowers TDEE. This effect is not mediated by other variables in the list.",
            "Expected": "Supported - direct mechanism"
        }
    ],
    "FP": [
        {
            "CLD": "Social_Norms",
            "Source": "Body Mass Index (BMI)",
            "Target": "Group-level BMI",
            "Motivation": "Body Mass Index (BMI) causes Group-level BMI to change because group-level BMI is typically calculated as the average of individual BMIs within a group. An increase in an individual's BMI directly increases the group's average BMI, while a decrease in an individual's BMI lowers the group-level BMI. This relationship is direct and positive, as changes in individual BMI causally affect the aggregate group measure.",
            "Expected": "Unsupported - compositional fallacy"
        },
        {
            "CLD": "Energy_Balance",
            "Source": "Physical Activity Level",
            "Target": "Basal Metabolic Rate",
            "Motivation": "Physical Activity Level causes Basal Metabolic Rate to change in a direct causal relationship. Increasing physical activity over time can lead to increases in muscle mass and metabolic adaptation, which in turn raises basal metabolic rate. Conversely, reduced physical activity can lower muscle mass and metabolic demand, decreasing basal metabolic rate. This relationship is supported by physiological mechanisms where physical activity induces metabolic changes that affect baseline energy expenditure.",
            "Expected": "Partially supported - mediated by muscle mass"
        }
    ],
    "FN": [
        {
            "CLD": "Energy_Balance",
            "Source": "Total Daily Energy Intake (TDEI)",
            "Target": "Body Mass Index (BMI)",
            "Motivation": "Total Daily Energy Intake (TDEI) does not directly cause Body Mass Index (BMI) in the presence of the mediating variable Total Daily Energy Expenditure (TDEE) in the CLD. The causal pathway is that TDEI and TDEE together determine energy balance, which then affects BMI. Therefore, the direct causal relationship between TDEI and BMI should not be included in the CLD.",
            "Expected": "Supported but mediated - should not be in CLD"
        }
    ]
}


async def test_edge(classification: str, edge: dict, edge_num: int, total_edges: int):
    """Test a single edge with Deep Research."""
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]Edge {edge_num}/{total_edges} - {classification}: {edge['Source']} → {edge['Target']}[/bold cyan]")
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
        
        # Extract all new fields including telemetry
        result = {
            "edge_num": edge_num,
            "classification": classification,
            "cld": edge['CLD'],
            "source": edge['Source'],
            "target": edge['Target'],
            "expected": edge['Expected'],
            "elapsed_seconds": round(elapsed_time, 2),
            
            # Telemetry (NEW)
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
            
            # Snippet consensus (NEW)
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
            
            # Support scores (NEW)
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
        console.print(f"  Snippet Consensus: {verdict.snippet_consensus_support if verdict.snippet_consensus_support else 'N/A'}")
        console.print(f"  Avg Support: Snippets={verdict.average_snippet_support_score}, Articles={verdict.average_article_support_score}, Combined={verdict.average_combined_support_score}")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"[red]✗ ERROR ({elapsed_time:.1f}s):[/red] {edge['Source']} → {edge['Target']}: {str(e)}")
        
        return {
            "edge_num": edge_num,
            "classification": classification,
            "cld": edge['CLD'],
            "source": edge['Source'],
            "target": edge['Target'],
            "expected": edge['Expected'],
            "elapsed_seconds": round(elapsed_time, 2),
            "success": False,
            "error": str(e)
        }


async def run_parallel_tests():
    """Run all edges in batches of 3."""
    # Flatten edges into single list
    all_tasks = []
    for classification, edges in GROUND_TRUTH_EDGES.items():
        for edge in edges:
            all_tasks.append((classification, edge))
    
    BATCH_SIZE = 3
    total_batches = (len(all_tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    console.print(f"\n{'='*80}")
    console.print(f"[bold magenta]RUNNING ALL {len(all_tasks)} EDGES IN BATCHES OF {BATCH_SIZE}[/bold magenta]")
    console.print(f"[bold magenta]Total batches: {total_batches}[/bold magenta]")
    console.print(f"[bold magenta]Using NEW multidimensional deep research with snippet consensus tracking[/bold magenta]")
    console.print(f"{'='*80}\n")
    
    all_results = []
    
    # Run in batches
    for batch_start in range(0, len(all_tasks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_tasks))
        batch = all_tasks[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        
        console.print(f"\n[bold yellow]📦 BATCH {batch_num}/{total_batches} - Processing edges {batch_start+1} to {batch_end}[/bold yellow]\n")
        
        # Run batch in parallel
        tasks = [
            test_edge(classification, edge, batch_start + i + 1, len(all_tasks))
            for i, (classification, edge) in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        console.print(f"\n[bold green]✓ Batch {batch_num}/{total_batches} completed[/bold green]\n")
        
        # Small delay between batches
        if batch_end < len(all_tasks):
            console.print(f"[dim]Waiting 5 seconds before next batch...[/dim]\n")
            await asyncio.sleep(5)
    
    return all_results


def analyze_results(results):
    """Analyze and display results."""
    console.print(f"\n\n{'='*80}")
    console.print("[bold blue]📊 TEST RESULTS ANALYSIS[/bold blue]")
    console.print(f"{'='*80}\n")
    
    # Success/failure
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    console.print(f"✓ Successful: {len(successful)}/{len(results)}")
    console.print(f"✗ Failed: {len(failed)}/{len(results)}")
    
    # By classification
    console.print(f"\n[bold cyan]Results by Classification:[/bold cyan]")
    for cls in ["TP", "FP", "FN"]:
        cls_results = [r for r in successful if r['classification'] == cls]
        if cls_results:
            console.print(f"\n  [{cls}] - {len(cls_results)} edges:")
            for r in cls_results:
                console.print(f"    {r['source']} → {r['target']}")
                console.print(f"      Verdict: {r['verdict']} (Conf: {r['confidence']}/10)")
                console.print(f"      Direct Causal: {r['is_direct_causal']} (Mech Spec: {r['mechanistic_specificity']}/10)")
                console.print(f"      Support Scores: Snip={r['average_snippet_support_score']}, Art={r['average_article_support_score']}, Comb={r['average_combined_support_score']}")
                console.print(f"      Expected: {r['expected']}")
    
    # New feature verification
    console.print(f"\n[bold cyan]New Features Verification:[/bold cyan]")
    snippet_consensus_count = sum(1 for r in successful if r.get('snippet_consensus_support'))
    counter_evidence_count = sum(1 for r in successful if r.get('counter_evidence_found'))
    alt_mech_count = sum(1 for r in successful if r.get('alternative_mechanisms_found'))
    
    console.print(f"  Snippet consensus populated: {snippet_consensus_count}/{len(successful)}")
    console.print(f"  Counter-evidence found: {counter_evidence_count}/{len(successful)}")
    console.print(f"  Alternative mechanisms found: {alt_mech_count}/{len(successful)}")
    
    # Telemetry summary
    console.print(f"\n[bold cyan]Telemetry Summary:[/bold cyan]")
    if successful:
        total_tokens = sum(r.get('total_tokens', 0) for r in successful if r.get('total_tokens'))
        total_api_calls = sum(r.get('total_api_calls', 0) for r in successful if r.get('total_api_calls'))
        avg_research_time = sum(r.get('research_elapsed_time_seconds', 0) for r in successful if r.get('research_elapsed_time_seconds')) / len(successful)
        
        console.print(f"  Total Tokens (all edges): {total_tokens:,}")
        console.print(f"  Total API Calls (all edges): {total_api_calls}")
        console.print(f"  Avg Research Time per Edge: {avg_research_time:.1f}s ({avg_research_time/60:.1f} min)")
        console.print(f"  Avg Tokens per Edge: {total_tokens/len(successful):,.0f}")
        console.print(f"  Avg API Calls per Edge: {total_api_calls/len(successful):.1f}")
    
    console.print(f"\n{'='*80}\n")


async def main():
    """Main entry point."""
    console.print("\n[bold]5-EDGE BATCH TEST WITH NEW MULTIDIMENSIONAL DEEP RESEARCH[/bold]\n")
    
    start_time = time.time()
    
    # Run tests
    results = await run_parallel_tests()
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path("/home/nitai/code/causalix.ai/data_science/test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_5_edges_multidim_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze
    analyze_results(results)
    
    console.print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    console.print(f"💾 Results saved to: {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
