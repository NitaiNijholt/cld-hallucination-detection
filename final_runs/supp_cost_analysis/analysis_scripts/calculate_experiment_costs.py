#!/usr/bin/env python3
"""
Calculate OpenAI API costs from experiment Excel files.

This script extracts token usage from LLM Usage Stats sheets in experiment files
and calculates costs based on OpenAI API pricing.

Usage:
    python calculate_experiment_costs.py --data-dir <path_to_experiment_data>

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
from typing import Dict, List, Tuple, Optional


# OpenAI API Pricing (December 2024)
# Source: https://platform.openai.com/docs/pricing
PRICING = {
    'gpt-4.1': {
        'input': 2.00 / 1_000_000,   # $2.00 per 1M input tokens
        'output': 8.00 / 1_000_000,  # $8.00 per 1M output tokens
    },
    'gpt-4o': {
        'input': 2.50 / 1_000_000,
        'output': 10.00 / 1_000_000,
    },
    'gpt-4o-mini': {
        'input': 0.15 / 1_000_000,
        'output': 0.60 / 1_000_000,
    },
    'gpt-5-mini': {
        'input': 0.15 / 1_000_000,   # Assumed same as gpt-4o-mini
        'output': 0.60 / 1_000_000,
    },
    'text-embedding-3-small': {
        'input': 0.02 / 1_000_000,
        'output': 0.0,
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


def extract_file_stats(file_path: Path) -> Optional[Dict]:
    """Extract token usage statistics from a single experiment file."""
    try:
        # Read LLM Usage Stats sheet
        df_stats = pd.read_excel(file_path, sheet_name="LLM Usage Stats")
        
        # Read All Edges sheet to get actual edge count
        try:
            df_edges = pd.read_excel(file_path, sheet_name="All Edges")
            n_edges = len(df_edges)
        except:
            n_edges = 0
        
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'n_edges': n_edges,  # Actual edges in CLD
            'roles': {}
        }
        
        for _, row in df_stats.iterrows():
            role = row['Role']
            tokens = parse_token_dict(row['Token Totals'])
            inference_calls = row.get('Inference Calls', 0)
            inference_time = row.get('Inference Time (s)', 0)
            
            if tokens.get('total_tokens', 0) > 0 or tokens.get('input_tokens', 0) > 0:
                result['roles'][role] = {
                    'prompt_tokens': tokens.get('prompt_tokens', tokens.get('input_tokens', 0)),
                    'completion_tokens': tokens.get('completion_tokens', 0),
                    'total_tokens': tokens.get('total_tokens', 
                                              tokens.get('prompt_tokens', 0) + tokens.get('completion_tokens', 0)),
                    'inference_calls': inference_calls,
                    'inference_time_s': inference_time,
                }
        
        return result if result['roles'] else None
        
    except Exception as e:
        return None


def determine_experiment_type(file_path: Path) -> Tuple[str, str]:
    """Determine experiment type and judge type from file path."""
    path_str = str(file_path).lower()
    
    # Experiment type
    if 'generation' in path_str or 'result_excel_path' in file_path.name.lower():
        exp_type = 'generation'
    elif 'corrector' in path_str or 'correction' in path_str:
        exp_type = 'correction'
    else:
        exp_type = 'judging'
    
    # Judge type
    if 'citation' in path_str:
        judge_type = 'citation'
    elif 'correctness' in path_str:
        judge_type = 'correctness'
    else:
        judge_type = 'unknown'
    
    return exp_type, judge_type


def determine_model(file_path: Path) -> str:
    """Determine the model used from file path."""
    path_str = str(file_path).lower()
    
    if 'gpt5mini' in path_str or 'gpt-5-mini' in path_str:
        return 'gpt-5-mini'
    elif 'gpt4o-mini' in path_str or 'gpt-4o-mini' in path_str:
        return 'gpt-4o-mini'
    else:
        return 'gpt-4.1'  # Default


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in USD for given token counts."""
    if model not in PRICING:
        model = 'gpt-4.1'  # Fallback
    
    pricing = PRICING[model]
    return (prompt_tokens * pricing['input'] + 
            completion_tokens * pricing['output'])


def process_experiment_directory(data_dir: Path) -> Dict:
    """Process all experiment files in a directory."""
    
    results = {
        'generation': {
            'gpt-4.1': {'files': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'edges': 0, 'inference_calls': 0}
        },
        'judging': {
            'correctness': {
                'gpt-4.1': {'files': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'edges': 0, 'inference_calls': 0}
            },
            'citation': {
                'gpt-4.1': {'files': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'edges': 0, 'inference_calls': 0},
                'gpt-5-mini': {'files': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'edges': 0, 'inference_calls': 0}
            }
        },
        'files_processed': [],
        'files_skipped': [],
    }
    
    # Find all Excel files
    excel_files = list(data_dir.glob("**/*.xlsx"))
    print(f"Found {len(excel_files)} Excel files")
    
    for file_path in excel_files:
        # Skip backup files
        if '.backup' in str(file_path):
            continue
        
        # Skip analysis result files
        if 'enhanced_analysis' in str(file_path) or 'sensitivity' in str(file_path).lower():
            continue
            
        stats = extract_file_stats(file_path)
        if not stats:
            results['files_skipped'].append(str(file_path))
            continue
        
        exp_type, judge_type = determine_experiment_type(file_path)
        model = determine_model(file_path)
        
        # Use n_edges from All Edges sheet (actual CLD edges)
        n_edges = stats.get('n_edges', 0)
        
        # Process based on experiment type
        if exp_type == 'generation':
            if 'Generator' in stats['roles']:
                role_data = stats['roles']['Generator']
                results['generation']['gpt-4.1']['files'] += 1
                results['generation']['gpt-4.1']['prompt_tokens'] += role_data['prompt_tokens']
                results['generation']['gpt-4.1']['completion_tokens'] += role_data['completion_tokens']
                results['generation']['gpt-4.1']['edges'] += n_edges if n_edges > 0 else role_data['inference_calls']
                results['generation']['gpt-4.1']['inference_calls'] += role_data['inference_calls']
                results['files_processed'].append({
                    'path': str(file_path),
                    'type': 'generation',
                    'tokens': role_data['total_tokens'],
                    'edges': n_edges,
                    'inference_calls': role_data['inference_calls']
                })
                
        elif exp_type == 'judging':
            if 'Judge' in stats['roles']:
                role_data = stats['roles']['Judge']
                
                # Only count files with actual token usage
                if role_data['total_tokens'] == 0:
                    results['files_skipped'].append(str(file_path))
                    continue
                
                if judge_type not in results['judging']:
                    judge_type = 'correctness'  # Fallback
                
                if model not in results['judging'][judge_type]:
                    model = 'gpt-4.1'  # Fallback
                
                # Use n_edges from All Edges sheet for per-edge metrics
                edge_count = n_edges if n_edges > 0 else role_data['inference_calls']
                
                results['judging'][judge_type][model]['files'] += 1
                results['judging'][judge_type][model]['prompt_tokens'] += role_data['prompt_tokens']
                results['judging'][judge_type][model]['completion_tokens'] += role_data['completion_tokens']
                results['judging'][judge_type][model]['edges'] += edge_count
                results['judging'][judge_type][model]['inference_calls'] += role_data['inference_calls']
                results['files_processed'].append({
                    'path': str(file_path),
                    'type': f'judging-{judge_type}',
                    'model': model,
                    'tokens': role_data['total_tokens'],
                    'edges': edge_count,
                    'inference_calls': role_data['inference_calls']
                })
    
    return results


def generate_report(results: Dict, output_dir: Path) -> str:
    """Generate cost report from results."""
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("OPENAI API COST ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    total_cost = 0.0
    summary_data = []
    
    # Generation costs
    gen_data = results['generation']['gpt-4.1']
    if gen_data['edges'] > 0:
        total_tokens = gen_data['prompt_tokens'] + gen_data['completion_tokens']
        cost = calculate_cost(gen_data['prompt_tokens'], gen_data['completion_tokens'], 'gpt-4.1')
        tokens_per_edge = total_tokens / gen_data['edges']
        cost_per_edge = cost / gen_data['edges'] * 100  # cents
        
        report_lines.append(f"\nGENERATION (GPT-4.1):")
        report_lines.append(f"  Files processed: {gen_data['files']}")
        report_lines.append(f"  Total edges: {gen_data['edges']:,}")
        report_lines.append(f"  Prompt tokens: {gen_data['prompt_tokens']:,}")
        report_lines.append(f"  Completion tokens: {gen_data['completion_tokens']:,}")
        report_lines.append(f"  Total tokens: {total_tokens:,}")
        report_lines.append(f"  Tokens per edge: {tokens_per_edge:,.0f}")
        report_lines.append(f"  Cost: ${cost:.2f}")
        report_lines.append(f"  Cost per edge: {cost_per_edge:.2f}¢")
        
        total_cost += cost
        summary_data.append({
            'Task': 'Generation (GPT-4.1)',
            'Files': gen_data['files'],
            'Edges': gen_data['edges'],
            'Tokens/Edge': int(tokens_per_edge),
            'Cost/Edge (¢)': round(cost_per_edge, 2),
            'Total Cost ($)': round(cost, 2)
        })
    
    # Judging costs
    for judge_type in ['correctness', 'citation']:
        for model in results['judging'][judge_type]:
            data = results['judging'][judge_type][model]
            if data['edges'] > 0:
                total_tokens = data['prompt_tokens'] + data['completion_tokens']
                cost = calculate_cost(data['prompt_tokens'], data['completion_tokens'], model)
                tokens_per_edge = total_tokens / data['edges']
                cost_per_edge = cost / data['edges'] * 100  # cents
                
                report_lines.append(f"\n{judge_type.upper()} JUDGING ({model.upper()}):")
                report_lines.append(f"  Files processed: {data['files']}")
                report_lines.append(f"  Total edges: {data['edges']:,}")
                report_lines.append(f"  Prompt tokens: {data['prompt_tokens']:,}")
                report_lines.append(f"  Completion tokens: {data['completion_tokens']:,}")
                report_lines.append(f"  Total tokens: {total_tokens:,}")
                report_lines.append(f"  Tokens per edge: {tokens_per_edge:,.0f}")
                report_lines.append(f"  Cost: ${cost:.2f}")
                report_lines.append(f"  Cost per edge: {cost_per_edge:.2f}¢")
                
                total_cost += cost
                summary_data.append({
                    'Task': f'{judge_type.title()} Judge ({model})',
                    'Files': data['files'],
                    'Edges': data['edges'],
                    'Tokens/Edge': int(tokens_per_edge),
                    'Cost/Edge (¢)': round(cost_per_edge, 2),
                    'Total Cost ($)': round(cost, 2)
                })
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append(f"TOTAL COST: ${total_cost:.2f}")
    report_lines.append("=" * 70)
    
    report_lines.append(f"\nFiles processed: {len(results['files_processed'])}")
    report_lines.append(f"Files skipped: {len(results['files_skipped'])}")
    
    # Save summary as CSV
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_dir / 'cost_summary.csv', index=False)
        report_lines.append(f"\nSummary saved to: {output_dir / 'cost_summary.csv'}")
    
    return '\n'.join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Calculate OpenAI API costs from experiment files')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to experiment data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: same as script)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing experiment data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process experiments
    results = process_experiment_directory(data_dir)
    
    # Generate report
    report = generate_report(results, output_dir)
    print(report)
    
    # Save report
    report_path = output_dir / f'cost_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save detailed results as JSON
    results_json = {
        'generation': results['generation'],
        'judging': results['judging'],
        'files_processed_count': len(results['files_processed']),
        'files_skipped_count': len(results['files_skipped']),
        'pricing': PRICING,
        'timestamp': datetime.now().isoformat(),
    }
    
    json_path = output_dir / 'cost_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Detailed results saved to: {json_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())







