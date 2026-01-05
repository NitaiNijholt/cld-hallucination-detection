#!/usr/bin/env python3
"""
Analyze RQ1b Correction Results

Calculates Precision, Recall, and F1 for:
1. Corrupted CLD vs Base CLD (Degradation)
2. Corrected CLD vs Base CLD (Recovery)

Usage:
    python analyze_rq1b_results.py <output_dir>
"""

import sys
import os
import pandas as pd
import json
from pathlib import Path
from neo4j import GraphDatabase

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_rq1b_results.py <output_dir>")
        sys.exit(1)
        
    output_dir = Path(sys.argv[1])
    session_info_file = output_dir / "session_info.txt"
    
    if not session_info_file.exists():
        print(f"Error: {session_info_file} not found")
        sys.exit(1)
        
    # Load session info
    info = {}
    with open(session_info_file, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                info[k] = v
                
    base_id = info.get('base_session_id')
    corrupted_id = info.get('corrupted_session_id')
    corrected_id = info.get('corrected_session_id')
    
    print("="*80)
    print("RQ1b ANALYSIS: CORRECTION RECOVERY")
    print("="*80)
    
    # Connect to Neo4j to fetch edges
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "bvdGGOJxqQp3Wdz")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def get_edges(session_id):
        with driver.session() as session:
            query = """
            MATCH (s:variable)-[r]->(t:variable)
            WHERE s.session_id = $sid AND t.session_id = $sid
            RETURN s.name as source, t.name as target, type(r) as type
            """
            result = session.run(query, {"sid": session_id})
            return set([f"{r['source']}|{r['target']}|{r['type']}" for r in result])

    try:
        base_edges = get_edges(base_id)
        corrupted_edges = get_edges(corrupted_id)
        corrected_edges = get_edges(corrected_id)
        
        print(f"Base Edges (GT):      {len(base_edges)}")
        print(f"Corrupted Edges:      {len(corrupted_edges)}")
        print(f"Corrected Edges:      {len(corrected_edges)}")
        print("-" * 40)
        
        def calculate_metrics(pred_edges, gt_edges):
            tp = len(pred_edges.intersection(gt_edges))
            fp = len(pred_edges - gt_edges)
            fn = len(gt_edges - pred_edges)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
            
        # 1. Corrupted vs Base
        p_corr, r_corr, f1_corr = calculate_metrics(corrupted_edges, base_edges)
        
        # 2. Corrected vs Base
        p_rec, r_rec, f1_rec = calculate_metrics(corrected_edges, base_edges)
        
        print(f"{'Metric':<15} {'Corrupted':<15} {'Corrected':<15} {'Delta':<15}")
        print("-" * 60)
        print(f"{'Precision':<15} {p_corr:<15.3f} {p_rec:<15.3f} {p_rec - p_corr:+.3f}")
        print(f"{'Recall':<15} {r_corr:<15.3f} {r_rec:<15.3f} {r_rec - r_corr:+.3f}")
        print(f"{'F1 Score':<15} {f1_corr:<15.3f} {f1_rec:<15.3f} {f1_rec - f1_corr:+.3f}")
        
        print("="*80)
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1'],
            'Corrupted': [p_corr, r_corr, f1_corr],
            'Corrected': [p_rec, r_rec, f1_rec],
            'Delta': [p_rec - p_corr, r_rec - r_corr, f1_rec - f1_corr]
        })
        results_df.to_csv(output_dir / "correction_metrics.csv", index=False)
        print(f"Metrics saved to {output_dir / 'correction_metrics.csv'}")

    finally:
        driver.close()

if __name__ == "__main__":
    main()

