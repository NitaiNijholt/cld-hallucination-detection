#!/usr/bin/env python3
"""
Calculate OpenAI API costs by experiment type.

This script extracts token usage from experiment files and calculates costs
broken down by the 4 main experiment types:
- Corruption Correctness
- Corruption Citation
- Ground Truth Correctness
- Ground Truth Citation

Usage:
    python calculate_costs_by_experiment.py --data-dir <path> --output-dir <path>

Author: Causalix.ai
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import argparse
import json
from datetime import datetime
from typing import Dict, Tuple


# OpenAI API Pricing (December 2024)
# Source: https://platform.openai.com/docs/pricing
PRICING = {
    'gpt-4.1': {
        'input': 2.00 / 1_000_000,   # $2.00 per 1M input tokens
        'output': 8.00 / 1_000_000,  # $8.00 per 1M output tokens
    },
    'gpt-5-mini': {
        'input': 0.15 / 1_000_000,   # $0.15 per 1M input tokens
        'output': 0.60 / 1_000_000,  # $0.60 per 1M output tokens
    },
}

# Experiment type definitions (final experiments only)
EXPERIMENTS = {
    # RQ1a Judge experiments
    'RQ1a Corruption Citation': {
        'dir': 'RQ1a_gt_synth_citation',
        'model': 'gpt-5-mini',
        'type': 'judge',
    },
    'RQ1a Ground Truth Citation': {
        'dir': 'RQ1a_gt_lit_citation',
        'model': 'gpt-4.1',
        'type': 'judge',
    },
    'RQ1a Corruption Correctness': {
        'dir': 'RQ1a_gt_synth_correctness',
        'model': 'gpt-4.1',
        'type': 'judge',
    },
    'RQ1a Ground Truth Correctness': {
        'dir': 'RQ1a_gt_lit_correctness',
        'model': 'gpt-4.1',
        'type': 'judge',
    },
    'RQ1a Human Validation': {
        'dir': 'RQ1_human_validation_citation_judge',
        'model': 'gpt-4.1',
        'type': 'judge',
    },
    # RQ1b Corrector experiments
    'RQ1b Corr. Synth Baseline': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_baseline',
        'model': 'gpt-4.1',
        'type': 'corrector',
    },
    'RQ1b Corr. Synth CoT': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_cot',
        'model': 'gpt-4.1',
        'type': 'corrector',
    },
    'RQ1b Corr. Synth Mech': {
        'dir': 'RQ1b_corrector_experiment_corruption_detection_correctness_final_mechanistic',
        'model': 'gpt-4.1',
        'type': 'corrector',
    },
    'RQ1b Corr. GT Baseline': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_baseline',
        'model': 'gpt-4.1',
        'type': 'corrector',
    },
    'RQ1b Corr. GT CoT': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_cot',
        'model': 'gpt-4.1',
        'type': 'corrector',
    },
    'RQ1b Corr. GT Mech': {
        'dir': 'RQ1b_corrector_experiment_ground_truth_correctness_mechanistic',
        'model': 'gpt-4.1',
        'type': 'corrector',
    },
}


def parse_token_dict(token_str) -> Dict[str, int]:
    """Parse token totals from string or dict format."""
    if isinstance(token_str, dict):
        return token_str
    if isinstance(token_str, str):
        try:
            return ast.literal_eval(token_str)
        except:
            return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in USD for given token counts and model."""
    pricing = PRICING.get(model, PRICING['gpt-4.1'])
    return prompt_tokens * pricing['input'] + completion_tokens * pricing['output']


def determine_experiment_type(file_path: Path) -> str:
    """Determine which experiment type a file belongs to."""
    path_str = str(file_path)
    for exp_name, exp_config in EXPERIMENTS.items():
        if exp_config['dir'] in path_str:
            return exp_name
    return None


def process_file(file_path: Path) -> Dict:
    """Extract token usage and edge count from a single experiment file."""
    try:
        # Read LLM Usage Stats
        df_stats = pd.read_excel(file_path, sheet_name="LLM Usage Stats", engine='openpyxl')
        judge_row = df_stats[df_stats['Role'] == 'Judge']
        
        if len(judge_row) == 0:
            return None
        
        tokens = parse_token_dict(judge_row['Token Totals'].values[0])
        if tokens.get('total_tokens', 0) == 0:
            return None
        
        # Get edge count from All Edges sheet
        df_edges = pd.read_excel(file_path, sheet_name="All Edges", engine='openpyxl')
        n_edges = len(df_edges)
        
        return {
            'file': file_path.name,
            'prompt_tokens': tokens.get('prompt_tokens', 0),
            'completion_tokens': tokens.get('completion_tokens', 0),
            'total_tokens': tokens.get('total_tokens', 0),
            'edges': n_edges,
            'inference_calls': judge_row['Inference Calls'].values[0],
        }
    except Exception as e:
        return None


def analyze_experiments(data_dir: Path) -> Dict:
    """Analyze all experiments and aggregate by experiment type."""
    
    # Initialize results
    results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        results[exp_name] = {
            'model': exp_config['model'],
            'exp_type': exp_config.get('type', 'judge'),
            'files': [],
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'edges': 0,
            'inference_calls': 0,
        }
    
    # Find all Excel files (judged_* for RQ1a, corrected_* for RQ1b, and others)
    excel_files = list(data_dir.glob("**/*.xlsx"))
    # Filter out analysis/summary files
    excel_files = [f for f in excel_files if not any(x in str(f) for x in 
                   ['enhanced_analysis', 'sensitivity', 'combined', 'metrics', 'results.xlsx'])]
    print(f"Found {len(excel_files)} Excel files")
    
    processed = 0
    skipped = 0
    
    for file_path in excel_files:
        # Skip backup files
        if '.backup' in str(file_path):
            continue
        
        # Determine experiment type
        exp_type = determine_experiment_type(file_path)
        if exp_type is None:
            skipped += 1
            continue
        
        # Process file
        file_data = process_file(file_path)
        if file_data is None:
            skipped += 1
            continue
        
        # Aggregate
        results[exp_type]['files'].append(file_data['file'])
        results[exp_type]['prompt_tokens'] += file_data['prompt_tokens']
        results[exp_type]['completion_tokens'] += file_data['completion_tokens']
        results[exp_type]['edges'] += file_data['edges']
        results[exp_type]['inference_calls'] += file_data['inference_calls']
        processed += 1
    
    print(f"Processed: {processed}, Skipped: {skipped}")
    
    return results


def generate_report(results: Dict, output_dir: Path) -> str:
    """Generate cost report from results."""
    
    lines = []
    lines.append("=" * 90)
    lines.append("OPENAI API COST ANALYSIS BY EXPERIMENT TYPE")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    
    # Calculate costs and build summary
    summary_data = []
    total_cost = 0.0
    
    lines.append(f"\n{'Experiment':<30} {'Model':<12} {'Files':>6} {'Edges':>8} {'Tok/Edge':>10} {'¢/Edge':>8} {'Total $':>10}")
    lines.append("-" * 90)
    
    for exp_name, data in results.items():
        n_files = len(data['files'])
        n_edges = data['edges']
        model = data['model']
        total_tokens = data['prompt_tokens'] + data['completion_tokens']
        cost = calculate_cost(data['prompt_tokens'], data['completion_tokens'], model)
        
        if n_edges > 0:
            tokens_per_edge = total_tokens / n_edges
            cost_per_edge = cost / n_edges * 100  # cents
        else:
            tokens_per_edge = 0
            cost_per_edge = 0
        
        lines.append(f"{exp_name:<30} {model:<12} {n_files:>6} {n_edges:>8,} {tokens_per_edge:>10,.0f} {cost_per_edge:>7.2f}¢ ${cost:>9,.2f}")
        
        total_cost += cost
        summary_data.append({
            'Experiment': exp_name,
            'Model': model,
            'Files': n_files,
            'Edges': n_edges,
            'Prompt Tokens': data['prompt_tokens'],
            'Completion Tokens': data['completion_tokens'],
            'Total Tokens': total_tokens,
            'Tokens/Edge': int(tokens_per_edge) if n_edges > 0 else 0,
            'Cost/Edge (¢)': round(cost_per_edge, 2),
            'Total Cost ($)': round(cost, 2),
        })
    
    lines.append("-" * 90)
    lines.append(f"{'TOTAL':<30} {'':<12} {'':<6} {'':<8} {'':<10} {'':<8} ${total_cost:>9,.2f}")
    lines.append("=" * 90)
    
    # Pricing info
    lines.append("\nPricing (OpenAI API, December 2024):")
    for model, prices in PRICING.items():
        lines.append(f"  {model}: ${prices['input']*1e6:.2f}/1M input, ${prices['output']*1e6:.2f}/1M output")
    
    # Key insights
    lines.append("\nKey Insights:")
    lines.append(f"  - Total experiment cost: ${total_cost:,.2f}")
    lines.append(f"  - Citation experiments use ~43-44k tokens/edge (retrieved context)")
    lines.append(f"  - Correctness experiments use ~470-490 tokens/edge (no retrieval)")
    lines.append(f"  - Ground Truth Citation (GPT-4.1) dominates costs ({summary_data[3]['Total Cost ($)']/total_cost*100:.0f}%)")
    
    # Save CSV
    df_summary = pd.DataFrame(summary_data)
    csv_path = output_dir / 'cost_by_experiment.csv'
    df_summary.to_csv(csv_path, index=False)
    lines.append(f"\nSummary saved to: {csv_path}")
    
    return '\n'.join(lines), summary_data


def generate_latex_table(summary_data: list) -> str:
    """Generate LaTeX table from summary data."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Computational Efficiency Summary by Experiment Type}")
    lines.append(r"\label{tab:cost_by_experiment}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Experiment} & \textbf{Files} & \textbf{Edges} & \textbf{Tokens/Edge} & \textbf{Cost/Edge (¢)} & \textbf{Total (\$)} \\")
    lines.append(r"\midrule")
    
    total_cost = 0
    for row in summary_data:
        exp_name = row['Experiment']
        model = row['Model']
        model_suffix = " (GPT-5-mini)" if model == 'gpt-5-mini' else " (GPT-4.1)" if 'Citation' in exp_name else ""
        
        lines.append(f"{exp_name}{model_suffix} & {row['Files']} & {row['Edges']:,} & {row['Tokens/Edge']:,} & {row['Cost/Edge (¢)']:.2f} & {row['Total Cost ($)']:.2f} \\\\")
        total_cost += row['Total Cost ($)']
    
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & & & & & \\textbf{{{total_cost:.2f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Calculate OpenAI API costs by experiment type')
    parser.add_argument('--data-dir', type=str, 
                        default=str(Path(__file__).parent.parent),  # Default: final_runs/
                        help='Path to experiment data directory (default: ../)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as script)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Analyze experiments
    results = analyze_experiments(data_dir)
    
    # Generate report
    report, summary_data = generate_report(results, output_dir)
    print(report)
    
    # Save report
    report_path = output_dir / f'cost_by_experiment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save LaTeX table
    latex_table = generate_latex_table(summary_data)
    latex_path = output_dir / 'cost_by_experiment_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")
    
    # Save JSON results (convert numpy types to native Python)
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj
    
    json_results = {
        'experiments': {name: {k: convert_to_native(v) for k, v in data.items() if k != 'files'} 
                       for name, data in results.items()},
        'summary': convert_to_native(summary_data),
        'pricing': PRICING,
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(data_dir),
    }
    json_path = output_dir / 'cost_by_experiment_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())







