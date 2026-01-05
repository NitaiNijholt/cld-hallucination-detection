#!/usr/bin/env python3
"""
Analyze RQ1b Correction Prompt Variants

Compares correction performance across different corrector prompt variants.
Similar to analyze_prompt_variants_corrupted.py but for correction instead of judging.

Usage:
    python analyze_rq1b_correction_variants.py <run_dir>
    
Example:
    python analyze_rq1b_correction_variants.py final_runs/RQ1b_correction_experiment_final/depressive/run_1
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from neo4j import GraphDatabase

def get_edges_from_neo4j(session_id: str) -> set:
    """Fetch edges from Neo4j for a given session."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            query = """
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND t.session_id = $sid
            AND type(r) <> 'NONE'
            RETURN s.name as source, t.name as target, type(r) as type
            """
            result = session.run(query, {"sid": session_id})
            edges = set([f"{r['source']}|{r['target']}|{r['type']}" for r in result])
        return edges
    finally:
        driver.close()

def calculate_metrics(pred_edges: set, gt_edges: set) -> dict:
    """Calculate Precision, Recall, F1."""
    tp = len(pred_edges.intersection(gt_edges))
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_rq1b_correction_variants.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    
    # Load session info
    session_info_file = run_dir / "session_info.txt"
    if not session_info_file.exists():
        print(f"Error: {session_info_file} not found")
        sys.exit(1)
    
    info = {}
    with open(session_info_file, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                info[k] = v
    
    corrupted_id = info.get('corrupted_session_id')
    base_id = info.get('base_session_id')
    
    print("="*80)
    print("RQ1b ANALYSIS: CORRECTOR PROMPT VARIANTS")
    print("="*80)
    print(f"Run Directory: {run_dir}")
    print(f"Corrupted Session: {corrupted_id}")
    print(f"Base Session (GT): {base_id}")
    print()
    
    # Get base edges (ground truth)
    print("Loading ground truth edges from Neo4j...")
    base_edges = get_edges_from_neo4j(base_id)
    corrupted_edges = get_edges_from_neo4j(corrupted_id)
    print(f"Base edges (GT): {len(base_edges)}")
    print(f"Corrupted edges: {len(corrupted_edges)}")
    print()
    
    # Find all corrected Excel files
    corrected_files = sorted(run_dir.glob("corrected_*.xlsx"))
    
    if not corrected_files:
        print(f"No corrected Excel files found in {run_dir}")
        sys.exit(1)
    
    print(f"Found {len(corrected_files)} corrected files:")
    for f in corrected_files:
        print(f"  - {f.name}")
    print()
    
    # Extract variant names and session IDs from filenames
    # Expected format: corrected_<cld>_<variant>_<timestamp>.xlsx
    variants_data = []
    
    for file_path in corrected_files:
        # Try to extract variant name from filename
        # e.g., corrected_Depressive_symptoms_cot_20251116_123456.xlsx
        parts = file_path.stem.split('_')
        
        # Find variant (original, improved, cot, mechanistic)
        variant = None
        for candidate in ["original", "improved", "cot", "mechanistic", "diagnostic"]:
            if candidate in file_path.stem:
                variant = candidate
                break
        
        if not variant:
            print(f"⚠️  Could not determine variant for {file_path.name}, skipping")
            continue
        
        # Read Excel to get session_id
        try:
            params_df = pd.read_excel(file_path, sheet_name='Params')
            session_id_row = params_df[params_df['Parameter'] == 'session_id']
            if not session_id_row.empty:
                corrected_session_id = session_id_row.iloc[0]['Value']
            else:
                print(f"⚠️  No session_id found in {file_path.name}, skipping")
                continue
            
            variants_data.append({
                'variant': variant,
                'file': file_path,
                'session_id': corrected_session_id
            })
        except Exception as e:
            print(f"⚠️  Error reading {file_path.name}: {e}")
            continue
    
    if not variants_data:
        print("No valid corrected files found")
        sys.exit(1)
    
    # Calculate metrics for each variant
    print("="*80)
    print("CALCULATING METRICS FOR EACH VARIANT")
    print("="*80)
    
    results = []
    
    for variant_data in variants_data:
        variant = variant_data['variant']
        session_id = variant_data['session_id']
        
        print(f"\n{variant.upper()}:")
        print(f"  Session ID: {session_id}")
        
        # Get corrected edges from Neo4j
        corrected_edges = get_edges_from_neo4j(session_id)
        print(f"  Corrected edges: {len(corrected_edges)}")
        
        # Calculate metrics
        metrics = calculate_metrics(corrected_edges, base_edges)
        
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        
        results.append({
            'Variant': variant,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'Edges': len(corrected_edges)
        })
    
    # Calculate baseline (corrupted) metrics
    print(f"\nBASELINE (Corrupted, no correction):")
    corrupted_metrics = calculate_metrics(corrupted_edges, base_edges)
    print(f"  Precision: {corrupted_metrics['precision']:.3f}")
    print(f"  Recall: {corrupted_metrics['recall']:.3f}")
    print(f"  F1 Score: {corrupted_metrics['f1']:.3f}")
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1', ascending=False)
    
    # Add improvement vs baseline
    results_df['F1_Improvement'] = results_df['F1'] - corrupted_metrics['f1']
    results_df['Precision_Improvement'] = results_df['Precision'] - corrupted_metrics['precision']
    results_df['Recall_Improvement'] = results_df['Recall'] - corrupted_metrics['recall']
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    output_file = run_dir / "analysis" / "corrector_variants_analysis.xlsx"
    output_file.parent.mkdir(exist_ok=True)
    
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name='Results', index=False)
        
        # Add baseline sheet
        baseline_df = pd.DataFrame([{
            'Metric': 'Corrupted (Baseline)',
            'Precision': corrupted_metrics['precision'],
            'Recall': corrupted_metrics['recall'],
            'F1': corrupted_metrics['f1']
        }])
        baseline_df.to_excel(writer, sheet_name='Baseline', index=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\n🏆 Best variant: {results_df.iloc[0]['Variant']} (F1={results_df.iloc[0]['F1']:.3f})")

if __name__ == "__main__":
    main()

