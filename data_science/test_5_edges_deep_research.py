"""
Test the deep research system on 5 sample edges to verify claim handling.
"""
import asyncio
import json
from pydantic_ai_deepresearch_multidimensional import DeepResearchManager
from pathlib import Path
from datetime import datetime

# 5 actual edges from saved CLDs: 2 TP, 2 FP, 1 FN for diverse testing
TEST_EDGES = [
    {
        "source": "Group-level BMI",
        "target": "Norm BMI",
        "motivation": "Group-level BMI causes Norm BMI to change in a direct causal relationship. As the average BMI within a group increases, the perceived normative or acceptable BMI (Norm BMI) tends to rise, as group norms adjust to what is commonly observed. Conversely, if Group-level BMI decreases, the Norm BMI typically shifts downward as well.",
        "classification": "TP"
    },
    {
        "source": "Physical Activity Level",
        "target": "Total Daily Energy Expenditure (TDEE)",
        "motivation": "Physical Activity Level causes Total Daily Energy Expenditure (TDEE) to change in a direct causal relationship. Increasing Physical Activity Level directly raises TDEE because more energy is required for movement and exercise. Conversely, decreasing Physical Activity Level directly lowers TDEE. This effect is not mediated by other variables in the list.",
        "classification": "TP"
    },
    {
        "source": "Body Mass Index (BMI)",
        "target": "Group-level BMI",
        "motivation": "Body Mass Index (BMI) causes Group-level BMI to change because group-level BMI is typically calculated as the average of individual BMIs within a group. An increase in an individual's BMI directly increases the group's average BMI, while a decrease in an individual's BMI lowers the group-level BMI. This relationship is direct and positive, as changes in individual BMI causally affect the aggregate group measure.",
        "classification": "FP"
    },
    {
        "source": "Physical Activity Level",
        "target": "Basal Metabolic Rate",
        "motivation": "Physical Activity Level causes Basal Metabolic Rate to change in a direct causal relationship. Increasing physical activity over time can lead to increases in muscle mass and metabolic adaptation, which in turn raises basal metabolic rate. Conversely, reduced physical activity can lower muscle mass and metabolic demand, decreasing basal metabolic rate. This relationship is supported by physiological mechanisms where physical activity induces metabolic changes that affect baseline energy expenditure.",
        "classification": "FP"
    },
    {
        "source": "Total Daily Energy Intake (TDEI)",
        "target": "Body Mass Index (BMI)",
        "motivation": "Total Daily Energy Intake (TDEI) does not directly cause Body Mass Index (BMI) in the presence of the mediating variable Total Daily Energy Expenditure (TDEE) in the CLD. The causal pathway is that TDEI and TDEE together determine energy balance, which then affects BMI. Therefore, the direct causal relationship between TDEI and BMI should not be included in the CLD.",
        "classification": "FN"
    }
]

async def test_edge(manager: DeepResearchManager, edge_data: dict, edge_num: int):
    """Test a single edge and return results."""
    source = edge_data["source"]
    target = edge_data["target"]
    claim = edge_data["motivation"]
    classification = edge_data["classification"]
    
    print(f"\n{'='*80}")
    print(f"TESTING EDGE {edge_num} ({classification}): {source} -> {target}")
    print(f"{'='*80}\n")
    
    try:
        verdict = await manager.run(claim)
        
        # Extract key results
        result = {
            "edge_num": edge_num,
            "source": source,
            "target": target,
            "ground_truth_classification": classification,
            "test_claim": claim,
            "original_claim": verdict.original_claim,
            "modified_claim": verdict.modified_claim,
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "is_direct_causal": verdict.is_direct_causal,
            "mechanistic_specificity_score": verdict.mechanistic_specificity_score,
            "direct_causal_confidence": verdict.direct_causal_confidence,
            "consensus_mechanism": verdict.consensus_mechanism,
            "strongest_causal_quote": verdict.strongest_causal_quote,
            "strongest_evidence_url": verdict.strongest_evidence_url,
            "success": True,
            "error": None
        }
        
        print(f"\n{'='*80}")
        print(f"EDGE {edge_num} RESULTS ({classification}):")
        print(f"  Source -> Target: {source} -> {target}")
        print(f"  Original Claim: {result['original_claim'][:100]}...")
        print(f"  Modified Claim: {result['modified_claim'][:100] if result['modified_claim'] else '(None)'}...")
        print(f"  Verdict: {result['verdict']} (Confidence: {result['confidence']}/10)")
        print(f"  Is Direct Causal: {result['is_direct_causal']}")
        print(f"  Mechanistic Specificity: {result['mechanistic_specificity_score']}/10")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR ON EDGE {edge_num}: {str(e)}")
        print(f"{'='*80}\n")
        
        return {
            "edge_num": edge_num,
            "source": source,
            "target": target,
            "ground_truth_classification": classification,
            "test_claim": claim,
            "success": False,
            "error": str(e)
        }

async def main():
    """Run tests on all 5 edges IN PARALLEL."""
    print(f"\n{'='*80}")
    print("DEEP RESEARCH SYSTEM TEST - 5 EDGES (PARALLEL)")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path("/home/nitai/code/causalix.ai/data_science/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run all tests IN PARALLEL using asyncio.gather
    print("🚀 Running all 5 edges in parallel...\n")
    
    tasks = []
    for i, edge_data in enumerate(TEST_EDGES, 1):
        # Create a fresh manager for each edge to avoid state contamination
        manager = DeepResearchManager(
            max_iterations=2,  # Reduced for faster testing
            minimum_evidence_count=3,
            max_sources_per_query=5
        )
        tasks.append(test_edge(manager, edge_data, i))
    
    # Run all edges in parallel
    results = await asyncio.gather(*tasks)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_5_edges_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_file}")
    
    # Check claim handling and show comparison with ground truth
    print(f"\n{'='*80}")
    print("CLAIM HANDLING & GROUND TRUTH COMPARISON")
    print(f"{'='*80}")
    
    for r in results:
        if r['success']:
            edge_num = r['edge_num']
            source = r['source']
            target = r['target']
            ground_truth = r['ground_truth_classification']
            test_claim = r['test_claim']
            original = r['original_claim']
            modified = r['modified_claim']
            is_direct = r['is_direct_causal']
            mech_spec = r['mechanistic_specificity_score']
            
            # Verify original claim matches test claim
            if original == test_claim:
                status = "✓ CORRECT"
            else:
                status = "✗ MISMATCH"
            
            print(f"\nEdge {edge_num}: {source} -> {target} (Ground Truth: {ground_truth})")
            print(f"  Claim Match: {status}")
            print(f"  Is Direct Causal: {is_direct} (Mech Spec: {mech_spec}/10)")
            print(f"  Test Claim:     {test_claim[:80]}...")
            print(f"  Original Claim: {original[:80]}...")
            print(f"  Modified Claim: {modified[:80] if modified else '(None - no modification)'}...")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
