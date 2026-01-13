#!/usr/bin/env python3
"""
Calculate OpenAI API costs using MAX POSSIBLE EDGES (N*(N-1)) method.

This ensures we capture costs even for runs where LLM Usage Stats weren't saved.
Uses: max_edges × avg_tokens_per_edge × runs × prompt_variants

Usage:
    python calculate_costs_max_edges.py [--output-dir <path>]

Author: Causalix.ai
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple
import ast


# OpenAI API Pricing (December 2024)
PRICING = {
    'gpt-4.1': {'input': 2.00 / 1_000_000, 'output': 8.00 / 1_000_000},
    'gpt-5-mini': {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
}

# CLD node counts (verified from data)
CLD_NODES = {
    'depressive': 15,
    'emergency_department': 35,
    'social_norms': 11,
}

# Calculate max edges for each CLD
CLD_MAX_EDGES = {cld: n * (n - 1) for cld, n in CLD_NODES.items()}

# Experiment configurations
EXPERIMENTS = {
    # RQ1a Judge experiments
    'RQ1a Corruption Citation': {
        'dir': 'RQ1a_gt_synth_citation',
        'model': 'gpt-5-mini',
        'type': 'citation',
        'prompt_variants': 3,  # baseline, cot, mechanistic
        'runs_per_cld': 3,
    },
    'RQ1a Ground Truth Citation': {
        'dir': 'RQ1a_gt_lit_citation',
        'model': 'gpt-4.1',
        'type': 'citation',
        'prompt_variants': 3,
        'runs_per_cld': 3,
    },
    'RQ1a Corruption Correctness': {
        'dir': 'RQ1a_gt_synth_correctness',
        'model': 'gpt-4.1',
        'type': 'correctness',
        'prompt_variants': 3,
        'runs_per_cld': 3,
    },
    'RQ1a Ground Truth Correctness': {
        'dir': 'RQ1a_gt_lit_correctness',
        'model': 'gpt-4.1',
        'type': 'correctness',
        'prompt_variants': 3,
        'runs_per_cld': 3,
    },
    'RQ1a Human Validation': {
        'dir': 'RQ1_human_validation_citation_judge',
        'model': 'gpt-4.1',
        'type': 'citation',
        'prompt_variants': 1,
        'runs_per_cld': 1,
        'clds': ['depressive'],  # Only one CLD for human validation
    },
    # RQ1b Corrector experiments
    'RQ1b Synth Baseline': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline',
        'model': 'gpt-4.1',
        'type': 'corrector',
        'prompt_variants': 1,
        'runs_per_cld': 3,
    },
    'RQ1b Synth CoT': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_cot',
        'model': 'gpt-4.1',
        'type': 'corrector',
        'prompt_variants': 1,
        'runs_per_cld': 3,
    },
    'RQ1b Synth Mechanistic': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic',
        'model': 'gpt-4.1',
        'type': 'corrector',
        'prompt_variants': 1,
        'runs_per_cld': 3,
    },
    'RQ1b GT Baseline': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_baseline',
        'model': 'gpt-4.1',
        'type': 'corrector',
        'prompt_variants': 1,
        'runs_per_cld': 3,
    },
    'RQ1b GT CoT': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_cot',
        'model': 'gpt-4.1',
        'type': 'corrector',
        'prompt_variants': 1,
        'runs_per_cld': 3,
    },
    'RQ1b GT Mechanistic': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_mechanistic',
        'model': 'gpt-4.1',
        'type': 'corrector',
        'prompt_variants': 1,
        'runs_per_cld': 3,
    },
}


def parse_tokens(token_str) -> Dict[str, int]:
    """Parse token totals from string or dict format."""
    if isinstance(token_str, dict):
        return token_str
    if isinstance(token_str, str):
        try:
            return ast.literal_eval(token_str)
        except:
            return {}
    return {}


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in USD."""
    pricing = PRICING.get(model, PRICING['gpt-4.1'])
    return prompt_tokens * pricing['input'] + completion_tokens * pricing['output']


def get_avg_tokens_per_edge(base_dir: Path) -> Dict[str, float]:
    """Calculate average tokens per edge for each experiment type from actual data."""
    
    tokens_per_edge = {'citation': [], 'correctness': []}
    
    for exp_name, config in EXPERIMENTS.items():
        if config['type'] not in ['citation', 'correctness']:
            continue
            
        dir_path = base_dir / config['dir']
        if not dir_path.exists():
            continue
            
        for f in dir_path.glob("**/*.xlsx"):
            if '.backup' in str(f) or 'analysis' in str(f):
                continue
            try:
                df_stats = pd.read_excel(f, sheet_name="LLM Usage Stats", engine='openpyxl')
                judge_row = df_stats[df_stats['Role'] == 'Judge']
                if len(judge_row) == 0:
                    continue
                
                tokens = parse_tokens(judge_row['Token Totals'].values[0])
                total_tokens = tokens.get('prompt_tokens', 0) + tokens.get('completion_tokens', 0)
                if total_tokens == 0:
                    continue
                
                df_edges = pd.read_excel(f, sheet_name="All Edges", engine='openpyxl')
                n_edges = len(df_edges)
                if n_edges == 0:
                    continue
                
                tpe = total_tokens / n_edges
                tokens_per_edge[config['type']].append(tpe)
            except:
                pass
    
    return {
        'citation': np.mean(tokens_per_edge['citation']) if tokens_per_edge['citation'] else 44000,
        'correctness': np.mean(tokens_per_edge['correctness']) if tokens_per_edge['correctness'] else 480,
    }


def count_actual_runs(base_dir: Path) -> Dict[str, Dict[str, int]]:
    """Count actual runs per CLD for each experiment."""
    
    run_counts = {}
    
    for exp_name, config in EXPERIMENTS.items():
        dir_path = base_dir / config['dir']
        if not dir_path.exists():
            run_counts[exp_name] = {}
            continue
            
        counts = {}
        for cld in CLD_NODES.keys():
            cld_path = dir_path / cld
            if cld_path.exists():
                runs = len(list(cld_path.glob('run_*')))
                if runs > 0:
                    counts[cld] = runs
        
        run_counts[exp_name] = counts
    
    return run_counts


def calculate_experiment_cost(config: Dict, run_counts: Dict[str, int], 
                             avg_tokens: Dict[str, float], model: str) -> Tuple[float, int]:
    """Calculate total cost for an experiment based on max edges."""
    
    exp_type = config['type']
    prompt_variants = config['prompt_variants']
    
    # Get tokens per edge
    if exp_type == 'citation':
        tpe = avg_tokens['citation']
    elif exp_type == 'correctness':
        tpe = avg_tokens['correctness']
    elif exp_type == 'corrector':
        # Corrector = correction + rejudging, use 2x correctness tokens
        tpe = avg_tokens['correctness'] * 2
    else:
        tpe = avg_tokens['correctness']
    
    total_edges = 0
    total_cost = 0
    
    # Use specific CLDs if defined, otherwise all
    clds = config.get('clds', CLD_NODES.keys())
    
    for cld in clds:
        if cld not in run_counts:
            continue
        
        n_runs = run_counts[cld]
        max_edges = CLD_MAX_EDGES[cld]
        
        # Total edges = max_edges × runs × prompt_variants
        edges = max_edges * n_runs * prompt_variants
        total_edges += edges
        
        # Calculate tokens and cost
        total_tokens = edges * tpe
        # Assume 78% prompt, 22% completion (from actual data)
        prompt_tokens = int(total_tokens * 0.78)
        completion_tokens = int(total_tokens * 0.22)
        
        cost = calculate_cost(prompt_tokens, completion_tokens, model)
        total_cost += cost
    
    return total_cost, total_edges


def main():
    parser = argparse.ArgumentParser(description='Calculate costs using max edges method')
    parser.add_argument('--data-dir', type=str, 
                        default=str(Path(__file__).parent.parent),
                        help='Path to final_runs directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as script)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 95)
    print("COST ANALYSIS USING MAX POSSIBLE EDGES (N × (N-1))")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 95)
    
    # Get average tokens per edge from actual data
    print("\n1. Calculating average tokens/edge from actual data...")
    avg_tokens = get_avg_tokens_per_edge(base_dir)
    print(f"   Citation:   {avg_tokens['citation']:.0f} tokens/edge")
    print(f"   Correctness: {avg_tokens['correctness']:.0f} tokens/edge")
    
    # Count actual runs
    print("\n2. Counting runs per CLD...")
    run_counts = count_actual_runs(base_dir)
    
    # Calculate costs
    print("\n3. Calculating costs (max_edges × runs × prompt_variants × tokens/edge)...")
    print("-" * 95)
    print(f"{'Experiment':<35} {'Model':<10} {'Edges':>10} {'Tok/Edge':>10} {'Total ($)':>12}")
    print("-" * 95)
    
    results = []
    total_cost = 0
    
    for exp_name, config in EXPERIMENTS.items():
        exp_run_counts = run_counts.get(exp_name, {})
        if not exp_run_counts:
            continue
            
        cost, edges = calculate_experiment_cost(config, exp_run_counts, avg_tokens, config['model'])
        
        exp_type = config['type']
        if exp_type == 'citation':
            tpe = avg_tokens['citation']
        elif exp_type == 'correctness':
            tpe = avg_tokens['correctness']
        else:
            tpe = avg_tokens['correctness'] * 2
        
        print(f"{exp_name:<35} {config['model']:<10} {edges:>10,} {tpe:>10,.0f} ${cost:>11,.2f}")
        
        results.append({
            'Experiment': exp_name,
            'Model': config['model'],
            'Type': exp_type,
            'Max Edges': edges,
            'Tokens/Edge': int(tpe),
            'Total Cost ($)': round(cost, 2),
        })
        
        total_cost += cost
    
    print("-" * 95)
    print(f"{'GRAND TOTAL':<35} {'':<10} {'':<10} {'':<10} ${total_cost:>11,.2f}")
    print("=" * 95)
    
    # CLD info
    print("\nCLD Max Edges:")
    for cld, n in CLD_NODES.items():
        print(f"  {cld}: {n} nodes → {CLD_MAX_EDGES[cld]} max edges")
    
    # Pricing
    print("\nPricing (OpenAI API, December 2024):")
    for model, prices in PRICING.items():
        print(f"  {model}: ${prices['input']*1e6:.2f}/1M input, ${prices['output']*1e6:.2f}/1M output")
    
    print("\nNotes:")
    print("  - Edges = max_edges × runs × prompt_variants (captures ALL possible API calls)")
    print("  - Corrector experiments: 2× correctness tokens (correction + rejudging)")
    print("  - Token/completion ratio: 78% prompt, 22% completion (from actual data)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df = pd.DataFrame(results)
    csv_path = output_dir / 'cost_max_edges.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")
    
    json_data = {
        'results': results,
        'total_cost': total_cost,
        'avg_tokens_per_edge': avg_tokens,
        'cld_nodes': CLD_NODES,
        'cld_max_edges': CLD_MAX_EDGES,
        'pricing': PRICING,
        'timestamp': datetime.now().isoformat(),
    }
    json_path = output_dir / 'cost_max_edges_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to: {json_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())







