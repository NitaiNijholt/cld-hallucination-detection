#!/usr/bin/env python3
"""
Compute Enriched Ground Truth Metrics via Deep Research Validation.

This script computes what the LLM generation F1/Precision/Recall would be if we 
used Deep Research (DR) to enrich the expert ground truth by promoting FP edges 
that have direct causal literature evidence to TP status.

Methodology:
- Original GT: Expert-constructed CLDs (TP + FN edges)
- Enriched GT: Original GT + FP edges where DR found direct causal evidence
- This tests the hypothesis that some "hallucinations" are actually valid 
  relationships that experts omitted due to scope/time constraints

Caveats (documented in thesis):
1. Domain variation: DR detection rates vary (28% Depressive vs 76% Older Persons FPs)
2. Correlational confound: DR may detect correlational rather than causal evidence
3. Circularity risk: LLM-based DR validating LLM-generated edges
4. Single-run: No DR replication for uncertainty quantification
5. Requires expert validation: Results are preliminary pending domain expert review

Output: Saves results to JSON and prints thesis-ready values.

Author: Automated analysis for MSc thesis
Date: 2024-12-18
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os

# Input/output roots (support rerouting outputs under a single run folder)
# Default INPUT_DIR to the RQ3_deep_research folder (parent of analysis_scripts)
INPUT_DIR = Path(os.environ.get("RQ3_INPUT_DIR", str(Path(__file__).parent.parent))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("RQ3_OUTPUT_DIR", str(INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files - same as analyze_deep_research_results.py (kept under INPUT_DIR/data)
DATA_DIR = INPUT_DIR / "Data"
DATA_FILES = [
    ("deep_research_results_depressive_symptoms_20251013_032002_84edges.json", "Depressive"),
    ("deep_research_results_social_norms_20251012_213641_17edges.json", "Social"),
    ("deep_research_results_older_persons_ALL_EDGES_184edges.json", "Older"),
]


def load_results(json_file: Path) -> List[Dict]:
    """Load results from a JSON file, handling both formats."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        return data.get('results', [])


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute Precision, Recall, and F1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def analyze_enrichment(results: List[Dict]) -> Dict:
    """
    Analyze the effect of enriching ground truth with DR-validated FP edges.
    
    Returns detailed breakdown including:
    - Original metrics (expert GT only)
    - Enriched metrics (expert GT + DR-validated FPs)
    - Per-CLD breakdown
    """
    # Count by classification and DR status
    counts = {
        'TP': {'total': 0, 'dr_found': 0, 'dr_not_found': 0},
        'FP': {'total': 0, 'dr_found': 0, 'dr_not_found': 0},
        'FN': {'total': 0, 'dr_found': 0, 'dr_not_found': 0},
    }
    
    # Per-CLD breakdown
    cld_counts = {}
    
    for result in results:
        cls = result['classification']
        cld = result.get('CLD', 'Unknown')
        dr_found = result.get('direct_causal_found', False)
        
        if cls not in counts:
            continue
            
        counts[cls]['total'] += 1
        if dr_found:
            counts[cls]['dr_found'] += 1
        else:
            counts[cls]['dr_not_found'] += 1
        
        # Per-CLD tracking
        if cld not in cld_counts:
            cld_counts[cld] = {
                'TP': {'total': 0, 'dr_found': 0},
                'FP': {'total': 0, 'dr_found': 0},
                'FN': {'total': 0, 'dr_found': 0},
            }
        cld_counts[cld][cls]['total'] += 1
        if dr_found:
            cld_counts[cld][cls]['dr_found'] += 1
    
    # Original metrics (expert GT)
    original_tp = counts['TP']['total']
    original_fp = counts['FP']['total']
    original_fn = counts['FN']['total']
    original_metrics = compute_metrics(original_tp, original_fp, original_fn)
    
    # Enriched metrics: promote DR-validated FPs to TPs
    fp_promoted = counts['FP']['dr_found']  # FPs with DR evidence → become TPs
    enriched_tp = original_tp + fp_promoted
    enriched_fp = original_fp - fp_promoted  # Remaining FPs without evidence
    enriched_fn = original_fn  # FNs unchanged (edges LLM didn't generate)
    enriched_metrics = compute_metrics(enriched_tp, enriched_fp, enriched_fn)
    
    # Calculate improvement for enriched
    f1_improvement = enriched_metrics['f1'] - original_metrics['f1']
    precision_improvement = enriched_metrics['precision'] - original_metrics['precision']
    recall_improvement = enriched_metrics['recall'] - original_metrics['recall']
    
    # === LLM+DR SYSTEM SCENARIO ===
    # In a complete LLM+DR system, DR would also discover FN edges with evidence
    # (edges the LLM missed but DR could find via comprehensive search)
    fn_discovered = counts['FN']['dr_found']  # FNs with DR evidence → system would find
    
    # LLM+DR system output: LLM edges + DR-discovered FNs
    # Compared against enriched GT (which includes promoted FPs)
    system_tp = enriched_tp + fn_discovered  # All validated edges
    system_fp = enriched_fp  # FPs without evidence (LLM hallucinations)
    system_fn = original_fn - fn_discovered  # Only FNs without DR evidence remain
    system_metrics = compute_metrics(system_tp, system_fp, system_fn)
    
    # Enriched GT size for this scenario
    enriched_gt_size = original_tp + original_fn + fp_promoted  # 44 + 63 + 111 = 218
    
    # System improvement over original
    system_f1_improvement = system_metrics['f1'] - original_metrics['f1']
    
    # Per-CLD enriched metrics
    cld_enriched = {}
    for cld, cld_data in cld_counts.items():
        cld_orig_tp = cld_data['TP']['total']
        cld_orig_fp = cld_data['FP']['total']
        cld_orig_fn = cld_data['FN']['total']
        cld_fp_promoted = cld_data['FP']['dr_found']
        cld_fn_discovered = cld_data['FN']['dr_found']
        
        # Enriched (FP promotion only)
        cld_enriched_tp = cld_orig_tp + cld_fp_promoted
        cld_enriched_fp = cld_orig_fp - cld_fp_promoted
        cld_enriched_fn = cld_orig_fn
        
        # LLM+DR System (FP promotion + FN discovery)
        cld_system_tp = cld_enriched_tp + cld_fn_discovered
        cld_system_fp = cld_enriched_fp
        cld_system_fn = cld_orig_fn - cld_fn_discovered
        
        cld_enriched[cld] = {
            'original': compute_metrics(cld_orig_tp, cld_orig_fp, cld_orig_fn),
            'enriched': compute_metrics(cld_enriched_tp, cld_enriched_fp, cld_enriched_fn),
            'system': compute_metrics(cld_system_tp, cld_system_fp, cld_system_fn),
            'fp_promoted': cld_fp_promoted,
            'fp_total': cld_orig_fp,
            'fn_discovered': cld_fn_discovered,
            'fn_total': cld_orig_fn,
            'promotion_rate': cld_fp_promoted / cld_orig_fp * 100 if cld_orig_fp > 0 else 0,
            'fn_discovery_rate': cld_fn_discovered / cld_orig_fn * 100 if cld_orig_fn > 0 else 0,
        }
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_edges': sum(c['total'] for c in counts.values()),
        'counts': counts,
        'original_metrics': original_metrics,
        'enriched_metrics': enriched_metrics,
        'system_metrics': system_metrics,
        'improvement': {
            'f1': f1_improvement,
            'precision': precision_improvement,
            'recall': recall_improvement,
            'f1_relative_pct': f1_improvement / original_metrics['f1'] * 100 if original_metrics['f1'] > 0 else 0,
        },
        'system_improvement': {
            'f1': system_f1_improvement,
            'f1_relative_pct': system_f1_improvement / original_metrics['f1'] * 100 if original_metrics['f1'] > 0 else 0,
        },
        'fp_promotion': {
            'promoted': fp_promoted,
            'not_promoted': counts['FP']['dr_not_found'],
            'promotion_rate_pct': fp_promoted / counts['FP']['total'] * 100 if counts['FP']['total'] > 0 else 0,
        },
        'fn_discovery': {
            'discovered': fn_discovered,
            'not_discovered': counts['FN']['dr_not_found'],
            'discovery_rate_pct': fn_discovered / counts['FN']['total'] * 100 if counts['FN']['total'] > 0 else 0,
        },
        'per_cld': cld_enriched,
    }


def print_results(analysis: Dict):
    """Print thesis-ready results."""
    print("\n" + "=" * 80)
    print("ENRICHED GROUND TRUTH ANALYSIS")
    print("=" * 80)
    print(f"\nTimestamp: {analysis['timestamp']}")
    print(f"Total edges analyzed: {analysis['total_edges']}")
    
    # Original metrics
    orig = analysis['original_metrics']
    print("\n--- ORIGINAL METRICS (Expert GT) ---")
    print(f"  TP: {orig['tp']}, FP: {orig['fp']}, FN: {orig['fn']}")
    print(f"  Precision: {orig['precision']:.3f} ({orig['precision']*100:.1f}%)")
    print(f"  Recall:    {orig['recall']:.3f} ({orig['recall']*100:.1f}%)")
    print(f"  F1:        {orig['f1']:.3f} ({orig['f1']*100:.1f}%)")
    
    # Enriched metrics (Scenario 1: FP promotion only)
    enr = analysis['enriched_metrics']
    print("\n--- SCENARIO 1: ENRICHED GT (Expert GT + DR-validated FPs) ---")
    print(f"  TP: {enr['tp']}, FP: {enr['fp']}, FN: {enr['fn']}")
    print(f"  Precision: {enr['precision']:.3f} ({enr['precision']*100:.1f}%)")
    print(f"  Recall:    {enr['recall']:.3f} ({enr['recall']*100:.1f}%)")
    print(f"  F1:        {enr['f1']:.3f} ({enr['f1']*100:.1f}%)")
    
    # LLM+DR System metrics (Scenario 2: FP promotion + FN discovery)
    sys = analysis['system_metrics']
    print("\n--- SCENARIO 2: LLM+DR SYSTEM (+ DR-discovered FNs) ---")
    print(f"  TP: {sys['tp']}, FP: {sys['fp']}, FN: {sys['fn']}")
    print(f"  Precision: {sys['precision']:.3f} ({sys['precision']*100:.1f}%)")
    print(f"  Recall:    {sys['recall']:.3f} ({sys['recall']*100:.1f}%)")
    print(f"  F1:        {sys['f1']:.3f} ({sys['f1']*100:.1f}%)")
    
    # Improvements
    imp = analysis['improvement']
    sys_imp = analysis['system_improvement']
    print("\n--- IMPROVEMENTS OVER ORIGINAL ---")
    print(f"  Scenario 1 (Enriched GT):  F1 +{imp['f1']:.3f} (+{imp['f1_relative_pct']:.0f}% relative)")
    print(f"  Scenario 2 (LLM+DR Sys):   F1 +{sys_imp['f1']:.3f} (+{sys_imp['f1_relative_pct']:.0f}% relative)")
    
    # FP promotion details
    fp = analysis['fp_promotion']
    print("\n--- FP PROMOTION ---")
    print(f"  FPs with DR evidence (promoted to TP): {fp['promoted']}")
    print(f"  FPs without DR evidence (remain FP):   {fp['not_promoted']}")
    print(f"  Promotion rate: {fp['promotion_rate_pct']:.1f}%")
    
    # FN discovery details
    fn = analysis['fn_discovery']
    print("\n--- FN DISCOVERY (LLM+DR System) ---")
    print(f"  FNs with DR evidence (system discovers): {fn['discovered']}")
    print(f"  FNs without DR evidence (tacit knowledge): {fn['not_discovered']}")
    print(f"  Discovery rate: {fn['discovery_rate_pct']:.1f}%")
    
    # Per-CLD breakdown
    print("\n--- PER-CLD BREAKDOWN ---")
    for cld, data in sorted(analysis['per_cld'].items()):
        print(f"\n  {cld}:")
        print(f"    Original F1:  {data['original']['f1']:.3f}")
        print(f"    Enriched F1:  {data['enriched']['f1']:.3f}")
        print(f"    LLM+DR F1:    {data['system']['f1']:.3f}")
        print(f"    FPs promoted: {data['fp_promoted']}/{data['fp_total']} ({data['promotion_rate']:.1f}%)")
        print(f"    FNs discovered: {data['fn_discovered']}/{data['fn_total']} ({data['fn_discovery_rate']:.1f}%)")
    
    # Thesis-ready values
    print("\n" + "=" * 80)
    print("THESIS-READY VALUES (copy to LaTeX)")
    print("=" * 80)
    print(f"""
% Original metrics
\\newcommand{{\\origPrecision}}{{{orig['precision']:.3f}}}
\\newcommand{{\\origRecall}}{{{orig['recall']:.3f}}}
\\newcommand{{\\origFone}}{{{orig['f1']:.3f}}}

% Scenario 1: Enriched GT (FP promotion only)
\\newcommand{{\\enrichPrecision}}{{{enr['precision']:.3f}}}
\\newcommand{{\\enrichRecall}}{{{enr['recall']:.3f}}}
\\newcommand{{\\enrichFone}}{{{enr['f1']:.3f}}}

% Scenario 2: LLM+DR System (FP promotion + FN discovery)
\\newcommand{{\\systemPrecision}}{{{sys['precision']:.3f}}}
\\newcommand{{\\systemRecall}}{{{sys['recall']:.3f}}}
\\newcommand{{\\systemFone}}{{{sys['f1']:.3f}}}

% FP Promotion
\\newcommand{{\\fpPromoted}}{{{fp['promoted']}}}
\\newcommand{{\\fpPromotionRate}}{{{fp['promotion_rate_pct']:.1f}\\%}}

% FN Discovery
\\newcommand{{\\fnDiscovered}}{{{fn['discovered']}}}
\\newcommand{{\\fnDiscoveryRate}}{{{fn['discovery_rate_pct']:.1f}\\%}}

% Improvements over original
\\newcommand{{\\enrichFoneImprovement}}{{+{imp['f1']:.3f}}}
\\newcommand{{\\enrichFoneImprovementPct}}{{{imp['f1_relative_pct']:.0f}\\%}}
\\newcommand{{\\systemFoneImprovement}}{{+{sys_imp['f1']:.3f}}}
\\newcommand{{\\systemFoneImprovementPct}}{{{sys_imp['f1_relative_pct']:.0f}\\%}}
""")


def main():
    """Load data, compute enriched GT metrics, and save results."""
    print("\nLoading Deep Research results...")
    
    all_results = []
    for filename, cld_name in DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found, skipping")
            continue
        results = load_results(filepath)
        # Add CLD name if not present
        for r in results:
            if 'CLD' not in r:
                r['CLD'] = cld_name
        all_results.extend(results)
        print(f"  Loaded {len(results)} edges from {cld_name}")
    
    print(f"\nTotal edges: {len(all_results)}")
    
    # Compute enrichment analysis
    analysis = analyze_enrichment(all_results)
    
    # Print results
    print_results(analysis)
    
    # Save to JSON
    output_dir = OUTPUT_DIR
    output_file = output_dir / f"enriched_gt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Also save a "latest" symlink-style file
    latest_file = output_dir / "enriched_gt_analysis_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Latest results: {latest_file}")
    
    return analysis


if __name__ == "__main__":
    main()







