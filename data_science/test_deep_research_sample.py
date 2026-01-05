#!/usr/bin/env python3
"""
Test Deep Research on sample edges from different classifications (TP, FP, TN, FN)
to assess if it can discriminate between correct and incorrect causal edges.
"""

import asyncio
import json
from pydantic_ai_deepresearch_orchestration import DeepResearchManager
from rich.console import Console
from rich.table import Table
import time
import logging
import sys

# Set up logging - reduce httpx verbosity, show our detailed logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Reduce verbosity of httpx to only show warnings
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Ensure the pydantic AI deep research logger is visible
logging.getLogger('pydantic_ai_deepresearch_orchestration').setLevel(logging.INFO)

console = Console()

# Sample edges from each classification - ALL 11 EDGES
SAMPLE_EDGES = {
    "TP": [
        {
            "Source": "Basal Metabolic Rate",
            "Target": "Total Daily Energy Expenditure (TDEE)",
            "Motivation": "Basal Metabolic Rate causes Total Daily Energy Expenditure (TDEE) to change in a direct causal relationship. BMR is a major component of TDEE, so increasing BMR directly increases TDEE, while decreasing BMR directly decreases TDEE.",
            "Classification": "TP",
            "Expected": "Should find STRONG causal evidence (RCTs, experiments)",
        },
        {
            "Source": "Total Daily Energy Expenditure (TDEE)",
            "Target": "Body Mass Index (BMI)",
            "Motivation": "Total Daily Energy Expenditure (TDEE) causes Body Mass Index (BMI) to change in an inverse relationship. Increasing TDEE leads to greater energy expenditure, resulting in weight loss and thus decreasing BMI, while decreasing TDEE results in less energy expenditure, leading to weight gain and increasing BMI.",
            "Classification": "TP",
            "Expected": "Should find STRONG causal evidence (metabolic studies, RCTs)",
        },
        {
            "Source": "Total Daily Energy Intake (TDEI)",
            "Target": "Body Mass Index (BMI)",
            "Motivation": "Total Daily Energy Intake (TDEI) causes Body Mass Index (BMI) to change through energy balance. Increased energy intake, when exceeding expenditure, leads to positive energy balance and weight gain, thus increasing BMI.",
            "Classification": "TP",
            "Expected": "Should find STRONG causal evidence (nutrition RCTs, energy balance studies)",
        },
        {
            "Source": "Body Mass Index (BMI)",
            "Target": "Weight Norm Misperception",
            "Motivation": "Body Mass Index (BMI) causes Weight Norm Misperception through social comparison and cognitive biases. As an individual's BMI increases, they may develop misperceptions about normal weight ranges, either underestimating their own weight or believing higher weights are more common than they actually are.",
            "Classification": "TP",
            "Expected": "Should find evidence (psychology studies on weight perception)",
        }
    ],
    "FP": [
        {
            "Source": "Socio-cultural Ideal BMI",
            "Target": "Individual Ideal BMI",
            "Motivation": "Socio-cultural Ideal BMI causes Individual Ideal BMI to change in a direct causal relationship. As the socio-cultural ideal shifts higher, individuals tend to adjust their own ideal BMI upwards in response to societal norms and pressures.",
            "Classification": "FP",
            "Expected": "Should find WEAK/NO direct causal evidence (correlational studies only)",
        },
        {
            "Source": "Physical Activity Level",
            "Target": "Basal Metabolic Rate",
            "Motivation": "Physical Activity Level causes Basal Metabolic Rate to change in a direct causal relationship. Increased physical activity stimulates muscle growth and metabolic adaptation, leading to higher BMR, while decreased activity leads to muscle loss and lower BMR.",
            "Classification": "FP",
            "Expected": "Should find WEAK/NO direct causal evidence (BMR is resting rate, not affected by PA directly)",
        },
        {
            "Source": "Actual BMI",
            "Target": "Weight Norm Misperception",
            "Motivation": "Actual BMI causes Weight Norm Misperception through cognitive biases. Higher actual BMI may lead to normalized perception of higher weights, creating misperceptions about population weight norms.",
            "Classification": "FP",
            "Expected": "Should find WEAK/NO direct causal evidence (reverse causality possible)",
        },
        {
            "Source": "Perceived Weight Status",
            "Target": "Weight Control Efforts",
            "Motivation": "Perceived Weight Status causes Weight Control Efforts to change directly. When individuals perceive themselves as overweight, they are more likely to engage in weight control behaviors.",
            "Classification": "FP",
            "Expected": "Should find WEAK/NO direct causal evidence (confounding by actual weight)",
        }
    ],
    "TN": [
        {
            "Source": "Individual Ideal BMI",
            "Target": "Basal Metabolic Rate",
            "Motivation": "Individual Ideal BMI causes Basal Metabolic Rate to change. As someone's ideal BMI increases, their metabolic rate adjusts accordingly.",
            "Classification": "TN",
            "Expected": "Should find NO causal evidence (ideal is mental construct, BMR is physiological)",
        },
        {
            "Source": "Weight Norm Misperception",
            "Target": "Physical Activity Level",
            "Motivation": "Weight Norm Misperception causes Physical Activity Level to change. Misperceiving weight norms leads to changes in exercise behavior.",
            "Classification": "TN",
            "Expected": "Should find NO causal evidence (misperception doesn't directly cause PA)",
        }
    ],
    "FN": [
        {
            "Source": "Body Mass Index (BMI)",
            "Target": "BMI Heritability",
            "Motivation": "Body Mass Index (BMI) causes BMI Heritability estimates to change. As population BMI changes, the estimated heritability of BMI is affected.",
            "Classification": "FN",
            "Expected": "Should find NO direct evidence (heritability is population statistic, not individual)",
        }
    ]
}


async def test_edge(classification: str, edge: dict, manager: DeepResearchManager):
    """Test a single edge with Deep Research."""
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]Testing {classification}: {edge['Source']} → {edge['Target']}[/bold cyan]")
    console.print(f"[yellow]Expected:[/yellow] {edge['Expected']}")
    console.print(f"{'='*80}\n")
    
    # Log edge details
    logger.info("="*80)
    logger.info(f"TESTING EDGE: {classification}")
    logger.info(f"Source: {edge['Source']}")
    logger.info(f"Target: {edge['Target']}")
    logger.info(f"Classification: {classification}")
    logger.info(f"Expected result: {edge['Expected']}")
    logger.info(f"Motivation: {edge['Motivation']}")
    logger.info("="*80)
    
    # Format as causal claim
    claim = f"{edge['Source']} causes {edge['Target']}. {edge['Motivation']}"
    
    logger.info(f"Formatted claim for Deep Research: {claim[:200]}...")
    
    start_time = time.time()
    
    try:
        # Run Deep Research (returns only verdict now)
        verdict = await manager.run(claim)
        
        duration = time.time() - start_time
        
        # Extract key results
        result = {
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
            "evidence_urls": verdict.evidence_urls,  # Include all evidence URLs
            "reasoning": verdict.judge_reasoning,  # Include FULL reasoning, not truncated
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
        
        # Log detailed results
        logger.info(f"RESULT: verdict={verdict.verdict}, confidence={verdict.confidence}/10, direct_causal={verdict.direct_causal_evidence_found}")
        logger.info(f"  Evidence count: {len(verdict.evidence_urls)}")
        logger.info(f"  Evidence URLs: {verdict.evidence_urls[:3]}")  # First 3 URLs
        logger.info(f"  Reasoning: {verdict.judge_reasoning[:300]}...")
        if verdict.direct_causal_evidence_found:
            logger.info(f"  Strongest causal evidence URL: {verdict.strongest_causal_evidence_url}")
            logger.info(f"  Strongest causal evidence study type: {verdict.strongest_causal_evidence_study_type}")
            logger.info(f"  Strongest causal evidence passage: {verdict.strongest_causal_evidence_passage[:200] if verdict.strongest_causal_evidence_passage else 'None'}...")
            logger.info(f"  Strongest causal evidence summary: {verdict.strongest_causal_evidence_summary[:200] if verdict.strongest_causal_evidence_summary else 'None'}...")
        logger.info(f"  Duration: {duration:.1f}s")
        
        return result
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
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
    """Run tests on all sample edges."""
    console.print("[bold blue]🔬 DEEP RESEARCH DISCRIMINATION TEST[/bold blue]")
    console.print("Testing if Deep Research can discriminate between TP/FP/TN/FN edges\n")
    
    # Log test setup
    logger.info("="*80)
    logger.info("DEEP RESEARCH DISCRIMINATION TEST - SETUP")
    logger.info("="*80)
    logger.info("Sample edges to test:")
    for classification, edges in SAMPLE_EDGES.items():
        logger.info(f"  {classification}: {len(edges)} edges")
        for i, edge in enumerate(edges, 1):
            logger.info(f"    {i}. {edge['Source']} → {edge['Target']}")
    logger.info(f"Total edges: {sum(len(edges) for edges in SAMPLE_EDGES.values())}")
    logger.info("="*80)
    
    # Create Deep Research manager with reduced iterations for faster testing
    manager = DeepResearchManager(
        max_iterations=1,  # Single iteration for speed
        minimum_evidence_count=2,
        max_sources_per_query=5  # 5 sources per query for better coverage
    )
    
    logger.info("Deep Research Manager configured (reduced depth for speed):")
    logger.info(f"  max_iterations: 1 (single iteration)")
    logger.info(f"  minimum_evidence_count: 2")
    logger.info(f"  max_sources_per_query: 5 (better evidence coverage)")
    logger.info("="*80)
    
    all_results = []
    
    # Collect all edges across all classifications
    all_tasks = []
    for classification, edges in SAMPLE_EDGES.items():
        for edge in edges:
            all_tasks.append((classification, edge))
    
    console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
    console.print(f"[bold magenta]RUNNING ALL {len(all_tasks)} EDGES IN PARALLEL[/bold magenta]")
    console.print(f"[bold magenta]{'='*80}[/bold magenta]")
    
    # Run ALL edges in parallel
    console.print(f"\n[cyan]Processing {len(all_tasks)} edges in parallel...[/cyan]")
    logger.info(f"Running {len(all_tasks)} edges in parallel using asyncio.gather()")
    
    tasks = [test_edge(classification, edge, manager) for classification, edge in all_tasks]
    all_results = await asyncio.gather(*tasks)
    
    logger.info(f"All {len(all_results)} edges completed")
    
    return all_results


def analyze_results(results):
    """Analyze results to see if Deep Research can discriminate."""
    console.print("\n\n[bold blue]📊 DISCRIMINATION ANALYSIS[/bold blue]")
    console.print("="*80)
    
    # Create summary table
    table = Table(title="Deep Research Results by Classification")
    table.add_column("Class", style="cyan")
    table.add_column("Source → Target", style="white")
    table.add_column("Verdict", style="green")
    table.add_column("Conf.", justify="right")
    table.add_column("Direct Causal", justify="center")
    table.add_column("Causal Conf.", justify="right")
    table.add_column("Evidence", justify="right")
    
    for r in results:
        if r.get('verdict') == 'ERROR':
            table.add_row(
                r['classification'],
                f"{r['source'][:20]}... → {r['target'][:20]}...",
                "[red]ERROR[/red]",
                "-",
                "-",
                "-",
                "-"
            )
        else:
            verdict_color = "green" if r['verdict'] in ['supported', 'partially_supported'] else "red"
            table.add_row(
                r['classification'],
                f"{r['source'][:20]}... → {r['target'][:20]}...",
                f"[{verdict_color}]{r['verdict'].upper()}[/{verdict_color}]",
                str(r['confidence']),
                "✓" if r['direct_causal_found'] else "✗",
                str(r.get('strongest_causal_evidence_confidence', '-')),
                str(r['evidence_count'])
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
    
    # Calculate mean confidence for each classification
    console.print("\n[bold cyan]Mean Confidence by Classification:[/bold cyan]")
    for cls in ['TP', 'FP', 'TN', 'FN']:
        if cls in by_class:
            confidences = [r['confidence'] for r in by_class[cls]]
            mean_conf = sum(confidences) / len(confidences)
            console.print(f"  {cls}: {mean_conf:.2f}/10 (n={len(confidences)})")
    
    # Check direct causal evidence detection
    console.print("\n[bold cyan]Direct Causal Evidence Detection:[/bold cyan]")
    for cls in ['TP', 'FP', 'TN', 'FN']:
        if cls in by_class:
            detected = sum(1 for r in by_class[cls] if r['direct_causal_found'])
            total = len(by_class[cls])
            console.print(f"  {cls}: {detected}/{total} ({detected/total*100:.0f}%)")
    
    # Key question: Can it discriminate TP from FP?
    console.print("\n[bold yellow]🎯 Key Discrimination Test:[/bold yellow]")
    if 'TP' in by_class and 'FP' in by_class:
        tp_conf = sum(r['confidence'] for r in by_class['TP']) / len(by_class['TP'])
        fp_conf = sum(r['confidence'] for r in by_class['FP']) / len(by_class['FP'])
        
        console.print(f"  TP mean confidence: {tp_conf:.2f}/10")
        console.print(f"  FP mean confidence: {fp_conf:.2f}/10")
        console.print(f"  Difference: {tp_conf - fp_conf:.2f}")
        
        if tp_conf > fp_conf + 1.0:
            console.print("  [bold green]✓ Good discrimination! TP > FP by >1 point[/bold green]")
        elif tp_conf > fp_conf:
            console.print("  [yellow]⚠ Weak discrimination. TP > FP but difference <1 point[/yellow]")
        else:
            console.print("  [bold red]✗ No discrimination. FP ≥ TP[/bold red]")
    
    # Save results
    output_file = "deep_research_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold]Results saved to:[/bold] {output_file}")
    
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
