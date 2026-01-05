#!/usr/bin/env python3
"""
Calculate Correction Metrics - Unified Script for All RQ1b Scenarios

This script calculates F1 improvement for corrector effectiveness across different ground truth types:
- Scenario 1: vs synthetic base session (corruption detection)
- Scenario 2/3: vs Excel validation edges (ground truth correctness/citation)

Usage:
  # Scenario 1: Compare to synthetic base session
  python calculate_correction_metrics.py \
    --corrected-session 35df0df2... \
    --gt-session 278d89e0... \
    --judged-session 846d2a58...

  # Scenario 2/3: Compare to Excel validation
  python calculate_correction_metrics.py \
    --corrected-session <id> \
    --gt-excel Social_norms_and_obesity_prevalence.xlsx \
    --judged-session <id>
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase

def get_edges_from_neo4j(session_id: str, neo4j_password: str = None) -> set:
    """Fetch edges from Neo4j session."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            query = '''
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND type(r) <> 'NONE'
            RETURN s.name as source, t.name as target, type(r) as type
            '''
            result = session.run(query, {'sid': session_id})
            edges = set([f"{r['source']}|{r['target']}|{r['type']}" for r in result])
        return edges
    finally:
        driver.close()

def load_edges_from_excel(excel_path: str) -> set:
    """Load ground truth edges from Excel validation file."""
    df = pd.read_excel(excel_path, sheet_name="Variable_links")
    edges = set()
    
    for _, row in df.iterrows():
        source = row['Source']
        target = row['Target']
        polarity = row['Polarity'].lower() if pd.notna(row['Polarity']) else 'none'
        
        # Map polarity to relationship type
        if polarity == 'positive':
            rel_type = 'POSITIVE'
        elif polarity == 'negative':
            rel_type = 'NEGATIVE'
        else:
            rel_type = 'NONE'
        
        # Only include causal edges
        if rel_type != 'NONE':
            edges.add(f"{source}|{target}|{rel_type}")
    
    return edges

def calculate_metrics(predicted_edges: set, ground_truth_edges: set) -> dict:
    """Calculate Precision, Recall, F1."""
    tp = len(predicted_edges.intersection(ground_truth_edges))
    fp = len(predicted_edges - ground_truth_edges)
    fn = len(ground_truth_edges - predicted_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_predicted': len(predicted_edges),
        'total_ground_truth': len(ground_truth_edges)
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate correction effectiveness metrics")
    parser.add_argument("--corrected-session", required=True, help="Corrected session ID")
    parser.add_argument("--judged-session", required=True, help="Judged session ID (before correction)")
    parser.add_argument("--gt-session", help="Ground truth session ID (for Scenario 1)")
    parser.add_argument("--gt-excel", help="Ground truth Excel file path (for Scenarios 2/3)")
    parser.add_argument("--output", default="correction_metrics.csv", help="Output file path (CSV)")
    parser.add_argument("--text", action="store_true", help="Output as text instead of CSV")
    parser.add_argument("--cld-name", default="unknown", help="CLD name for CSV")
    parser.add_argument("--run-number", type=int, default=0, help="Run number for CSV")
    parser.add_argument("--corrector-model", default="gpt-4.1", help="Corrector model used")
    parser.add_argument("--corrector-variant", default="original", help="Corrector prompt variant")
    parser.add_argument("--judge-model", default="gpt-4.1", help="Judge model used")
    parser.add_argument("--judge-type", default="correctness", help="Judge type (correctness/citation)")
    parser.add_argument("--judge-variant", default="baseline", help="Judge prompt variant (baseline/cot/mechanistic)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.gt_session and not args.gt_excel:
        print("Error: Must provide either --gt-session or --gt-excel")
        return 1
    
    if args.gt_session and args.gt_excel:
        print("Error: Provide only one of --gt-session or --gt-excel")
        return 1
    
    print("="*80)
    print("RQ1b CORRECTION METRICS CALCULATION")
    print("="*80)
    
    # Load ground truth
    if args.gt_session:
        print(f"Ground Truth Type: Neo4j Session (Synthetic Base)")
        print(f"GT Session: {args.gt_session}")
        gt_edges = get_edges_from_neo4j(args.gt_session)
    else:
        print(f"Ground Truth Type: Excel Validation File")
        print(f"GT Excel: {args.gt_excel}")
        gt_edges = load_edges_from_excel(args.gt_excel)
    
    print(f"Judged Session: {args.judged_session}")
    print(f"Corrected Session: {args.corrected_session}")
    print()
    
    # Load sessions
    judged_edges = get_edges_from_neo4j(args.judged_session)
    corrected_edges = get_edges_from_neo4j(args.corrected_session)
    
    print(f"Ground Truth: {len(gt_edges)} edges")
    print(f"Judged: {len(judged_edges)} edges")
    print(f"Corrected: {len(corrected_edges)} edges")
    print()
    
    # Calculate metrics
    metrics_judged = calculate_metrics(judged_edges, gt_edges)
    metrics_corrected = calculate_metrics(corrected_edges, gt_edges)
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Metric':<15} {'Judged':<15} {'Corrected':<15} {'Improvement':<15}")
    print("-"*65)
    print(f"{'Precision':<15} {metrics_judged['precision']:<15.3f} {metrics_corrected['precision']:<15.3f} {metrics_corrected['precision'] - metrics_judged['precision']:+.3f}")
    print(f"{'Recall':<15} {metrics_judged['recall']:<15.3f} {metrics_corrected['recall']:<15.3f} {metrics_corrected['recall'] - metrics_judged['recall']:+.3f}")
    print(f"{'F1 Score':<15} {metrics_judged['f1']:<15.3f} {metrics_corrected['f1']:<15.3f} {metrics_corrected['f1'] - metrics_judged['f1']:+.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  Judged:    TP={metrics_judged['tp']}, FP={metrics_judged['fp']}, FN={metrics_judged['fn']}")
    print(f"  Corrected: TP={metrics_corrected['tp']}, FP={metrics_corrected['fp']}, FN={metrics_corrected['fn']}")
    print("="*80)
    
    # Save to file (Excel by default, text if --text flag)
    if not args.text:
        # Excel output (single row for aggregation)
        import pandas as pd
        from datetime import datetime
        
        # Determine ground truth description
        if args.gt_session:
            gt_type = 'Synthetic_Base'
            gt_desc = f"Session_{args.gt_session[:8]}"
        else:
            gt_type = 'Excel_Validation'
            gt_desc = Path(args.gt_excel).name
        
        row = {
            # Experiment Metadata
            'CLD': args.cld_name,
            'Run': args.run_number,
            'Timestamp': datetime.now().isoformat(),
            # Session IDs
            'Judged_Session': args.judged_session,
            'Corrected_Session': args.corrected_session,
            # Ground Truth
            'GT_Type': gt_type,
            'GT_Description': gt_desc,
            'GT_Edges': len(gt_edges),
            # Model Configuration
            'Corrector_Model': args.corrector_model,
            'Corrector_Variant': args.corrector_variant,
            'Corrector_Temperature': 0.7,  # Default from code
            'Judge_Model': args.judge_model,
            'Judge_Type': args.judge_type,
            'Judge_Variant': args.judge_variant,
            'Judge_Temperature': 0.0,  # Default from code
            # Edge Counts
            'Judged_Total_Edges': len(judged_edges),
            'Corrected_Total_Edges': len(corrected_edges),
            # Confusion Matrix - Before
            'TP_Before': metrics_judged['tp'],
            'FP_Before': metrics_judged['fp'],
            'FN_Before': metrics_judged['fn'],
            # Confusion Matrix - After
            'TP_After': metrics_corrected['tp'],
            'FP_After': metrics_corrected['fp'],
            'FN_After': metrics_corrected['fn'],
            # Metrics - Before
            'Precision_Before': round(metrics_judged['precision'], 4),
            'Recall_Before': round(metrics_judged['recall'], 4),
            'F1_Before': round(metrics_judged['f1'], 4),
            # Metrics - After
            'Precision_After': round(metrics_corrected['precision'], 4),
            'Recall_After': round(metrics_corrected['recall'], 4),
            'F1_After': round(metrics_corrected['f1'], 4),
            # Deltas
            'Precision_Delta': round(metrics_corrected['precision'] - metrics_judged['precision'], 4),
            'Recall_Delta': round(metrics_corrected['recall'] - metrics_judged['recall'], 4),
            'F1_Delta': round(metrics_corrected['f1'] - metrics_judged['f1'], 4)
        }
        
        # Write to Excel
        output_path = Path(args.output)
        if output_path.suffix == '.csv':
            # Change extension to .xlsx
            output_path = output_path.with_suffix('.xlsx')
        
        # Create DataFrame from row
        new_df = pd.DataFrame([row])
        
        # Append to existing file or create new
        if output_path.exists():
            existing_df = pd.read_excel(output_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(output_path, index=False, engine='openpyxl')
        else:
            new_df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"\nExcel row appended to {output_path}")
    else:
        # Text output (human-readable)
        with open(args.output, 'w') as f:
            f.write(f"RQ1b Correction Metrics\n")
            f.write(f"{'='*80}\n")
            f.write(f"Ground Truth: {args.gt_session if args.gt_session else args.gt_excel}\n")
            f.write(f"Judged Session: {args.judged_session}\n")
            f.write(f"Corrected Session: {args.corrected_session}\n\n")
            f.write(f"Edges: GT={len(gt_edges)}, Judged={len(judged_edges)}, Corrected={len(corrected_edges)}\n\n")
            f.write(f"Metric          Judged          Corrected       Improvement\n")
            f.write(f"{'-'*65}\n")
            f.write(f"Precision       {metrics_judged['precision']:<15.3f} {metrics_corrected['precision']:<15.3f} {metrics_corrected['precision'] - metrics_judged['precision']:+.3f}\n")
            f.write(f"Recall          {metrics_judged['recall']:<15.3f} {metrics_corrected['recall']:<15.3f} {metrics_corrected['recall'] - metrics_judged['recall']:+.3f}\n")
            f.write(f"F1 Score        {metrics_judged['f1']:<15.3f} {metrics_corrected['f1']:<15.3f} {metrics_corrected['f1'] - metrics_judged['f1']:+.3f}\n")
        
        print(f"\nResults saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

